#include "splat_worker.h"

#include "freesplatter_capi_buffer.h"
#include "protocol.h"

#include <nvutils/logger.hpp>

#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#include <unistd.h>

namespace {

std::atomic<int> g_splatShutdownFlag{0};

void handleSigterm(int)
{
    g_splatShutdownFlag.store(1);
}

bool readAll(int fd, void* buffer, size_t size)
{
    auto* bytes = static_cast<std::uint8_t*>(buffer);
    size_t offset = 0;
    while(offset < size)
    {
        const ssize_t n = ::read(fd, bytes + offset, size - offset);
        if(n == 0)
        {
            return false;
        }
        if(n < 0)
        {
            if(errno == EINTR)
            {
                continue;
            }
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
    while(offset < size)
    {
        const ssize_t n = ::write(fd, bytes + offset, size - offset);
        if(n < 0)
        {
            if(errno == EINTR)
            {
                continue;
            }
            return false;
        }
        offset += static_cast<size_t>(n);
    }
    return true;
}

enum SplatMsgType : std::uint8_t {
    SPLAT_MSG_PROGRESS = 0,
    SPLAT_MSG_RESULT   = 1,
    SPLAT_MSG_ERROR    = 2,
};

bool sendProgress(int fd, std::uint32_t job_id, std::uint8_t stage, std::uint8_t percent)
{
    std::uint8_t type = SPLAT_MSG_PROGRESS;
    return writeAll(fd, &job_id, sizeof(job_id))
           && writeAll(fd, &type, sizeof(type))
           && writeAll(fd, &stage, sizeof(stage))
           && writeAll(fd, &percent, sizeof(percent));
}

bool sendResult(int fd, std::uint32_t job_id, std::uint32_t n_gaussians, const std::string& path)
{
    std::uint8_t type = SPLAT_MSG_RESULT;
    const std::uint32_t path_len = static_cast<std::uint32_t>(path.size());
    return writeAll(fd, &job_id, sizeof(job_id))
           && writeAll(fd, &type, sizeof(type))
           && writeAll(fd, &n_gaussians, sizeof(n_gaussians))
           && writeAll(fd, &path_len, sizeof(path_len))
           && (path.empty() || writeAll(fd, path.data(), path.size()));
}

bool sendError(int fd, std::uint32_t job_id, const std::string& error)
{
    std::uint8_t type = SPLAT_MSG_ERROR;
    const std::uint32_t err_len = static_cast<std::uint32_t>(error.size());
    return writeAll(fd, &job_id, sizeof(job_id))
           && writeAll(fd, &type, sizeof(type))
           && writeAll(fd, &err_len, sizeof(err_len))
           && (error.empty() || writeAll(fd, error.data(), error.size()));
}

}  // namespace

void SplatWorker::run(int parent_fd, const std::string& model_path,
                      const std::string& backend, const std::string& splat_cache_dir)
{
    std::signal(SIGTERM, handleSigterm);
    std::signal(SIGINT,  handleSigterm);

    FreeSplatterBuffer splatter;
    FreeSplatterBuffer::InitOptions opts{};
    opts.model_path = model_path;
    opts.device     = backend;
    opts.n_threads  = 4;

    if(!splatter.initialize(opts))
    {
        LOGE("SplatWorker: failed to initialize model (%s)\n", model_path.c_str());
        close(parent_fd);
        _exit(1);
    }

    LOGI("SplatWorker: model loaded (backend=%s)\n", backend.c_str());

    while(!g_splatShutdownFlag.load())
    {
        std::uint32_t job_id = 0;
        if(!readAll(parent_fd, &job_id, sizeof(job_id)))
        {
            break;
        }

        std::uint32_t n_views = 0;
        if(!readAll(parent_fd, &n_views, sizeof(n_views)))
        {
            break;
        }

        SplatRequestOptions options{};
        if(!readAll(parent_fd, &options, sizeof(options)))
        {
            break;
        }

        if(n_views == 0 || n_views > 16)
        {
            LOGE("SplatWorker: invalid n_views=%u for job %u\n", n_views, job_id);
            sendError(parent_fd, job_id, "invalid n_views");
            continue;
        }

        std::vector<std::vector<std::uint8_t>> images;
        images.reserve(n_views);
        bool read_ok = true;
        for(std::uint32_t i = 0; i < n_views; ++i)
        {
            std::uint32_t img_len = 0;
            if(!readAll(parent_fd, &img_len, sizeof(img_len)))
            {
                read_ok = false;
                break;
            }
            if(img_len == 0 || img_len > MAX_IMAGE_SIZE)
            {
                LOGE("SplatWorker: invalid image len=%u for job %u\n", img_len, job_id);
                read_ok = false;
                break;
            }
            std::vector<std::uint8_t> img(img_len);
            if(!readAll(parent_fd, img.data(), img_len))
            {
                read_ok = false;
                break;
            }
            images.push_back(std::move(img));
        }

        if(!read_ok)
        {
            break;
        }

        sendProgress(parent_fd, job_id, 1, 0);

        FreeSplatterBuffer::RunResult result = splatter.run(images);

        if(!result.error_msg.empty() || result.splat_bytes.empty())
        {
            const std::string err = result.error_msg.empty() ? "inference produced no output" : result.error_msg;
            LOGE("SplatWorker: job %u failed: %s\n", job_id, err.c_str());
            sendError(parent_fd, job_id, err);
            continue;
        }

        if(g_splatShutdownFlag.load())
        {
            break;
        }

        sendProgress(parent_fd, job_id, 3, 0);

        std::error_code ec;
        std::filesystem::create_directories(splat_cache_dir, ec);
        if(ec)
        {
            LOGE("SplatWorker: cannot create cache dir %s: %s\n", splat_cache_dir.c_str(), ec.message().c_str());
            sendError(parent_fd, job_id, "cannot create cache dir");
            continue;
        }

        const std::string out_path = splat_cache_dir + "/job_" + std::to_string(job_id) + ".splat";
        FILE* fp = std::fopen(out_path.c_str(), "wb");
        if(!fp)
        {
            LOGE("SplatWorker: cannot open %s for write: %s\n", out_path.c_str(), std::strerror(errno));
            sendError(parent_fd, job_id, "cannot open output file");
            continue;
        }
        const size_t written = std::fwrite(result.splat_bytes.data(), 1, result.splat_bytes.size(), fp);
        std::fclose(fp);
        if(written != result.splat_bytes.size())
        {
            LOGE("SplatWorker: short write to %s\n", out_path.c_str());
            sendError(parent_fd, job_id, "short write");
            continue;
        }

        sendProgress(parent_fd, job_id, 4, 100);
        sendResult(parent_fd, job_id, static_cast<std::uint32_t>(result.num_gaussians), out_path);
        LOGI("SplatWorker: job %u done (%d gaussians -> %s)\n", job_id, result.num_gaussians, out_path.c_str());
    }

    splatter.shutdown();
    close(parent_fd);
    _exit(0);
}