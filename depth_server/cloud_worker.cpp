#include "cloud_worker.h"

#include "cloud_splat_encoder.h"
#include "protocol.h"

#include <da_capi.h>
#include <nvutils/logger.hpp>

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <unistd.h>

namespace {

std::atomic<int> g_cloudShutdownFlag{0};

void handleSigterm(int)
{
    g_cloudShutdownFlag.store(1);
}

bool readAll(int fd, void* buffer, size_t size)
{
    auto* bytes = static_cast<std::uint8_t*>(buffer);
    size_t offset = 0;
    while (offset < size)
    {
        const ssize_t n = ::read(fd, bytes + offset, size - offset);
        if (n == 0)
            return false;
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return false;
        }
        offset += static_cast<size_t>(n);
    }
    return true;
}

bool writeAll(int fd, const void* buffer, size_t size)
{
    const auto* bytes = static_cast<const std::uint8_t*>(buffer);
    size_t offset = 0;
    while (offset < size)
    {
        const ssize_t n = ::write(fd, bytes + offset, size - offset);
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            return false;
        }
        offset += static_cast<size_t>(n);
    }
    return true;
}

enum CloudMsgType : std::uint8_t {
    CLOUD_MSG_PROGRESS = 0,
    CLOUD_MSG_RESULT   = 1,
    CLOUD_MSG_ERROR    = 2,
};

bool sendProgress(int fd, std::uint32_t job_id, std::uint8_t stage, std::uint8_t percent)
{
    std::uint8_t type = CLOUD_MSG_PROGRESS;
    return writeAll(fd, &job_id, sizeof(job_id))
           && writeAll(fd, &type, sizeof(type))
           && writeAll(fd, &stage, sizeof(stage))
           && writeAll(fd, &percent, sizeof(percent));
}

bool sendResult(int fd, std::uint32_t job_id, std::uint32_t n_points, const std::string& path)
{
    std::uint8_t type = CLOUD_MSG_RESULT;
    const std::uint32_t path_len = static_cast<std::uint32_t>(path.size());
    return writeAll(fd, &job_id, sizeof(job_id))
           && writeAll(fd, &type, sizeof(type))
           && writeAll(fd, &n_points, sizeof(n_points))
           && writeAll(fd, &path_len, sizeof(path_len))
           && (path.empty() || writeAll(fd, path.data(), path.size()));
}

bool sendError(int fd, std::uint32_t job_id, std::uint32_t error_code, const std::string& error)
{
    std::uint8_t type = CLOUD_MSG_ERROR;
    const std::uint32_t err_len = static_cast<std::uint32_t>(error.size());
    return writeAll(fd, &job_id, sizeof(job_id))
           && writeAll(fd, &type, sizeof(type))
           && writeAll(fd, &error_code, sizeof(error_code))
           && writeAll(fd, &err_len, sizeof(err_len))
           && (error.empty() || writeAll(fd, error.data(), error.size()));
}

bool shutdownWriteEnd(int fd)
{
    // Signal EOF to the reader while ensuring all buffered data is
    // still readable. Without this the parent may see POLLHUP before
    // POLLIN and close the socket without consuming the error message.
    return ::shutdown(fd, SHUT_WR) == 0;
}

}  // namespace

void CloudWorker::run(int parent_fd, const std::string& model_path,
                      const std::string& backend, const std::string& cloud_cache_dir,
                      const std::string& work_dir)
{
    (void)work_dir;

    std::signal(SIGTERM, handleSigterm);
    std::signal(SIGINT, handleSigterm);

    da_ctx* ctx = nullptr;

    while (!g_cloudShutdownFlag.load())
    {
        std::uint32_t job_id = 0;
        if (!readAll(parent_fd, &job_id, sizeof(job_id)))
            break;

        CloudRequestOptions opts{};
        if (!readAll(parent_fd, &opts, sizeof(opts)))
            break;

        if (opts.n_frames < 2 || opts.n_frames > MAX_CLOUD_FRAMES)
        {
            LOGE("CloudWorker: invalid n_frames=%u for job %u\n", opts.n_frames, job_id);
            sendError(parent_fd, job_id, 2,
                      "n_frames must be 2.." + std::to_string(MAX_CLOUD_FRAMES));
            continue;
        }

        std::vector<std::string> frame_paths;
        frame_paths.reserve(opts.n_frames);
        bool read_ok = true;
        for (std::uint32_t i = 0; i < opts.n_frames; ++i)
        {
            std::uint32_t path_len = 0;
            if (!readAll(parent_fd, &path_len, sizeof(path_len)))
            {
                read_ok = false;
                break;
            }
            if (path_len == 0 || path_len > MAX_CLOUD_PATH_LEN)
            {
                LOGE("CloudWorker: invalid path len=%u for job %u\n", path_len, job_id);
                read_ok = false;
                break;
            }
            std::string path(path_len, '\0');
            if (!readAll(parent_fd, path.data(), path_len))
            {
                read_ok = false;
                break;
            }
            frame_paths.push_back(std::move(path));
        }

        if (!read_ok)
        {
            LOGE("CloudWorker: truncated job %u (expected %u paths)\n", job_id, opts.n_frames);
            sendError(parent_fd, job_id, 2, "truncated job data");
            continue;
        }

        if (!ctx)
        {
            ctx = (backend == "metric")
                ? da_capi_load_nested(model_path.c_str(), (model_path + "-metric").c_str(), 4)
                : da_capi_load(model_path.c_str(), 4);

            if (!ctx)
            {
                const char* err = da_capi_last_error(nullptr);
                std::string err_str = err && err[0] ? err : "failed to load model";
                LOGE("CloudWorker: failed to load model: %s\n", err_str.c_str());
                sendError(parent_fd, job_id, 5, err_str);
                shutdownWriteEnd(parent_fd);
                break;
            }

            LOGI("CloudWorker: model loaded (backend=%s)\n", backend.c_str());
            // da_capi_points_stream has its own pose-capability guard
            // (checks engine->is_mono() || engine->is_da2()). The JSON
            // from da_capi_info_json does NOT contain a "pose" key, so
            // we must NOT gate here — let the C API validate instead.
        }

        sendProgress(parent_fd, job_id, 2, 0);

        if (opts.fuse)
        {
            double vf = (opts.fuse_voxel_frac > 0.0f)
                ? static_cast<double>(opts.fuse_voxel_frac) : 0.004;
            double tm = (opts.fuse_trunc_mult > 0.0f)
                ? static_cast<double>(opts.fuse_trunc_mult) : 4.0;
            da_capi_set_fuse_params(ctx, vf, tm);
        }

        std::vector<const char*> cpaths;
        cpaths.reserve(frame_paths.size());
        for (const auto& p : frame_paths)
            cpaths.push_back(p.c_str());

        int n_out = 0;
        std::vector<int> out_counts(frame_paths.size(), 0);
        float* out_xyz = nullptr;
        unsigned char* out_rgb = nullptr;
        float* out_radius = nullptr;

        int ret = da_capi_points_stream(ctx, cpaths.data(),
            static_cast<int>(cpaths.size()),
            static_cast<int>(opts.chunk_size),
            static_cast<int>(opts.overlap),
            static_cast<double>(opts.conf_pct),
            opts.point_size,
            static_cast<int>(opts.global_budget),
            static_cast<int>(opts.icp_refine),
            static_cast<int>(opts.loop_close),
            static_cast<int>(opts.fuse),
            static_cast<int>(opts.metric),
            0.03,
            &n_out, out_counts.data(),
            &out_xyz, &out_rgb, &out_radius);

        if (ret != 0 || !out_xyz)
        {
            const char* err = da_capi_last_error(ctx);
            std::string err_str = err && err[0] ? err : "da_capi_points_stream failed";
            LOGE("CloudWorker: job %u failed: %s\n", job_id, err_str.c_str());
            sendError(parent_fd, job_id, 3, err_str);
            da_capi_free_floats(out_xyz);
            da_capi_free_bytes(out_rgb);
            da_capi_free_floats(out_radius);
            continue;
        }

        if (g_cloudShutdownFlag.load())
            break;

        sendProgress(parent_fd, job_id, 3, 0);

        std::error_code ec;
        std::filesystem::create_directories(cloud_cache_dir, ec);
        if (ec)
        {
            LOGE("CloudWorker: cannot create cache dir %s: %s\n",
                 cloud_cache_dir.c_str(), ec.message().c_str());
            sendError(parent_fd, job_id, 3, "cannot create cache dir");
            da_capi_free_floats(out_xyz);
            da_capi_free_bytes(out_rgb);
            da_capi_free_floats(out_radius);
            continue;
        }

        std::vector<std::uint8_t> splat_bytes;
        cloud_splat_encoder_encode(out_xyz, out_rgb, out_radius,
                                   static_cast<size_t>(n_out), splat_bytes);

        const std::string out_path = cloud_cache_dir + "/job_"
            + std::to_string(job_id) + ".splat";
        FILE* fp = std::fopen(out_path.c_str(), "wb");
        if (!fp)
        {
            LOGE("CloudWorker: cannot open %s for write: %s\n",
                 out_path.c_str(), std::strerror(errno));
            sendError(parent_fd, job_id, 3, "cannot open output file");
            da_capi_free_floats(out_xyz);
            da_capi_free_bytes(out_rgb);
            da_capi_free_floats(out_radius);
            continue;
        }
        const size_t written = std::fwrite(splat_bytes.data(), 1, splat_bytes.size(), fp);
        std::fclose(fp);
        if (written != splat_bytes.size())
        {
            LOGE("CloudWorker: short write to %s\n", out_path.c_str());
            sendError(parent_fd, job_id, 3, "short write");
            da_capi_free_floats(out_xyz);
            da_capi_free_bytes(out_rgb);
            da_capi_free_floats(out_radius);
            std::filesystem::remove(out_path, ec);
            continue;
        }

        da_capi_free_floats(out_xyz);
        da_capi_free_bytes(out_rgb);
        da_capi_free_floats(out_radius);

        sendProgress(parent_fd, job_id, 4, 100);
        sendResult(parent_fd, job_id, static_cast<std::uint32_t>(n_out), out_path);
        LOGI("CloudWorker: job %u done (%d points -> %s)\n",
             job_id, n_out, out_path.c_str());
    }

    if (ctx)
        da_capi_free(ctx);
    ::close(parent_fd);
    ::_exit(0);
}
