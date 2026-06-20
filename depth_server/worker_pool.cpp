#include "worker_pool.h"

#include "protocol.h"
#include "da_capi_buffer.h"

#include <nvutils/logger.hpp>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <poll.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

namespace {

constexpr uint32_t kWorkerReadyMagic = 0;

std::int64_t nowNs()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
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

std::vector<std::string> discoverCudaDevices(int worker_count)
{
    std::vector<std::string> devices;
    const char* visible_devices = std::getenv("CUDA_VISIBLE_DEVICES");
    if(visible_devices != nullptr && visible_devices[0] != '\0')
    {
        const std::string env_devices = visible_devices;
        size_t start = 0;
        while(start <= env_devices.size())
        {
            const size_t comma = env_devices.find(',', start);
            const size_t end = (comma == std::string::npos) ? env_devices.size() : comma;
            std::string token = env_devices.substr(start, end - start);
            token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char ch) {
                            return std::isspace(ch) != 0;
                        }),
                        token.end());
            if(!token.empty())
            {
                devices.push_back("cuda:" + token);
            }
            if(comma == std::string::npos)
            {
                break;
            }
            start = comma + 1;
        }
    }

    if(devices.empty())
    {
        for(int i = 0; i < worker_count; ++i)
        {
            devices.push_back("cuda:" + std::to_string(i));
        }
    }

    return devices;
}

float computeDepthMax(const std::vector<float>& depth_data)
{
    float max_value = 0.0f;
    for(float value : depth_data)
    {
        if(std::isfinite(value))
        {
            max_value = std::max(max_value, value);
        }
    }
    return max_value;
}

}  // namespace

WorkerPool::~WorkerPool()
{
    shutdown();
}

bool WorkerPool::initialize(const std::string& model_path, int num_workers, const std::string& backend)
{
    shutdown();

    if(model_path.empty())
    {
        LOGE("WorkerPool: model path is empty\n");
        return false;
    }

    const unsigned int hw_threads = std::thread::hardware_concurrency();
    m_modelPath = model_path;
    m_numWorkers = (num_workers > 0) ? num_workers : static_cast<int>(std::max(1u, hw_threads));
    m_nextWorker = 0;
    m_totalProcessingMs = 0.0;
    m_completedFrames = 0;
    m_pendingQueue.clear();
    m_workers.clear();
    m_workers.reserve(static_cast<size_t>(m_numWorkers));
    m_dispatchTimesNs.clear();
    m_dispatchTimesNs.reserve(static_cast<size_t>(m_numWorkers));

    std::vector<std::string> devices;
    if(backend == "cuda")
    {
        devices = discoverCudaDevices(m_numWorkers);
    }
    else
    {
        devices.assign(static_cast<size_t>(m_numWorkers), "cpu");
    }

    for(int worker_id = 0; worker_id < m_numWorkers; ++worker_id)
    {
        const std::string& device = devices[static_cast<size_t>(worker_id) % devices.size()];
        WorkerProcess worker{};
        if(!spawnWorker(worker_id, device, worker))
        {
            LOGE("WorkerPool: failed to spawn worker %d\n", worker_id);
            shutdown();
            return false;
        }
        m_workers.push_back(std::move(worker));
        m_dispatchTimesNs.push_back(0);
    }

    LOGI("WorkerPool: initialized %d workers (backend=%s)\n", m_numWorkers, backend.c_str());
    return true;
}

bool WorkerPool::spawnWorker(int id, const std::string& device, WorkerProcess& worker_out)
{
    int sv[2];
    if(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == -1)
    {
        LOGE("WorkerPool: socketpair failed: %s\n", std::strerror(errno));
        return false;
    }

    const pid_t pid = fork();
    if(pid == -1)
    {
        LOGE("WorkerPool: fork failed: %s\n", std::strerror(errno));
        close(sv[0]);
        close(sv[1]);
        return false;
    }

    if(pid == 0)
    {
        close(sv[0]);

        if(device != "cpu")
        {
            setenv("DA_DEVICE", device.c_str(), 1);
        }

        const int n_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency() / 2));
        da_ctx* ctx = da_capi_load(m_modelPath.c_str(), n_threads);
        if(!ctx)
        {
            LOGE("Worker %d: failed to load model\n", id);
            close(sv[1]);
            _exit(1);
        }

        LOGI("Worker %d: model loaded (device=%s)\n", id, device.c_str());

        const uint32_t ready_msg = kWorkerReadyMagic;
        if(!writeAll(sv[1], &ready_msg, sizeof(ready_msg)))
        {
            LOGE("Worker %d: failed to send ready signal\n", id);
            da_capi_free(ctx);
            close(sv[1]);
            _exit(1);
        }

        while(true)
        {
            FrameHeader header{};
            if(!readAll(sv[1], &header, sizeof(header)))
            {
                break;
            }

            if(!validateHeader(&header, sizeof(header)))
            {
                LOGE("Worker %d: invalid header received\n", id);
                break;
            }

            if(header.message_type == MSG_SHUTDOWN)
            {
                break;
            }

            if(header.message_type != MSG_FRAME_REQUEST)
            {
                LOGW("Worker %d: ignoring unexpected message type %s\n", id,
                     getMessageTypeName(header.message_type));
                const size_t payload_bytes = computePayloadSize(&header);
                std::vector<std::uint8_t> skip(payload_bytes);
                if(payload_bytes != 0 && !readAll(sv[1], skip.data(), payload_bytes))
                {
                    break;
                }
                continue;
            }

            FrameRequestPayload payload{};
            if(!readAll(sv[1], &payload, sizeof(payload)))
            {
                break;
            }

            const size_t rgb_size = static_cast<size_t>(payload.width) * static_cast<size_t>(payload.height) * 3;
            std::vector<std::uint8_t> rgb(rgb_size);
            if(rgb_size != 0 && !readAll(sv[1], rgb.data(), rgb_size))
            {
                break;
            }

            DepthResult result = processFrame(rgb.data(), payload.width, payload.height, ctx);
            if(!result.depth)
            {
                LOGE("Worker %d: frame timestamp %u processing failed\n", id, payload.timestamp_ms);
                continue;
            }

            const size_t depth_size = static_cast<size_t>(result.w) * static_cast<size_t>(result.h);
            std::vector<float> depth(depth_size);
            std::memcpy(depth.data(), result.depth, depth_size * sizeof(float));

            DepthResponsePayload resp_payload{};
            resp_payload.width = static_cast<uint32_t>(result.w);
            resp_payload.height = static_cast<uint32_t>(result.h);
            resp_payload.timestamp_ms = payload.timestamp_ms;
            resp_payload.scale = 1.0f;
            resp_payload.bias = 0.0f;
            resp_payload.z_max = computeDepthMax(depth);

            FrameHeader resp_header{};
            resp_header.magic = TCP_DEPTH_MAGIC;
            resp_header.frame_length = static_cast<uint32_t>(sizeof(FrameHeader) + sizeof(DepthResponsePayload)
                                                             + depth_size * sizeof(float));
            resp_header.message_type = MSG_DEPTH_RESPONSE;

            const bool ok = writeAll(sv[1], &resp_header, sizeof(resp_header))
                            && writeAll(sv[1], &resp_payload, sizeof(resp_payload))
                            && (depth.empty() || writeAll(sv[1], depth.data(), depth_size * sizeof(float)));

            da_capi_free_floats(result.depth);

            if(!ok)
            {
                LOGE("Worker %d: failed to send depth response\n", id);
                break;
            }
        }

        da_capi_free(ctx);
        close(sv[1]);
        _exit(0);
    }

    close(sv[1]);

    uint32_t ready = 0;
    if(!readAll(sv[0], &ready, sizeof(ready)))
    {
        LOGE("WorkerPool: worker %d failed to signal readiness\n", id);
        close(sv[0]);
        int status = 0;
        waitpid(pid, &status, 0);
        return false;
    }

    worker_out = WorkerProcess{};
    worker_out.id = id;
    worker_out.pid = pid;
    worker_out.device = device;
    worker_out.socket_fd = sv[0];
    worker_out.busy = false;
    worker_out.current_frame = 0;
    worker_out.pending_depth_fd = -1;

    LOGI("WorkerPool: worker %d started (pid=%d, device=%s)\n", id, static_cast<int>(pid), device.c_str());
    return true;
}

int WorkerPool::submitFrame(uint32_t frame_index,
                            uint32_t timestamp_ms,
                            const uint8_t* rgb_data,
                            uint32_t width,
                            uint32_t height)
{
    std::scoped_lock lock(m_mutex);
    if(rgb_data == nullptr || width == 0 || height == 0)
    {
        LOGE("WorkerPool: invalid frame %u submission\n", frame_index);
        return -1;
    }

    PendingFrame frame{};
    frame.frame_index = frame_index;
    frame.timestamp_ms = timestamp_ms;
    frame.width = width;
    frame.height = height;
    frame.rgb.assign(rgb_data, rgb_data + static_cast<size_t>(width) * static_cast<size_t>(height) * 3);

    if(dispatchFrame(frame) != 0)
    {
        m_pendingQueue.push_back(std::move(frame));
        LOGD("WorkerPool: queued frame %u (queue=%zu)\n", frame_index, m_pendingQueue.size());
    }

    return static_cast<int>(frame_index);
}

int WorkerPool::dispatchFrame(const PendingFrame& frame)
{
    if(m_workers.empty())
    {
        LOGE("WorkerPool: no workers available for frame %u\n", frame.frame_index);
        return -1;
    }

    const size_t worker_count = m_workers.size();
    for(size_t attempt = 0; attempt < worker_count; ++attempt)
    {
        const size_t worker_idx = (m_nextWorker + attempt) % worker_count;
        WorkerProcess& worker = m_workers[worker_idx];
        if(worker.socket_fd < 0 || worker.busy)
        {
            continue;
        }

        FrameHeader header{};
        header.magic = TCP_DEPTH_MAGIC;
        header.frame_length = static_cast<uint32_t>(sizeof(FrameHeader) + sizeof(FrameRequestPayload) + frame.rgb.size());
        header.message_type = MSG_FRAME_REQUEST;

        FrameRequestPayload payload{};
        payload.width = frame.width;
        payload.height = frame.height;
        payload.timestamp_ms = frame.timestamp_ms;

        const bool ok = writeAll(worker.socket_fd, &header, sizeof(header))
                        && writeAll(worker.socket_fd, &payload, sizeof(payload))
                        && (frame.rgb.empty() || writeAll(worker.socket_fd, frame.rgb.data(), frame.rgb.size()));
        if(!ok)
        {
            LOGE("WorkerPool: failed to dispatch frame %u to worker %d: %s\n", frame.frame_index, worker.id,
                 std::strerror(errno));
            close(worker.socket_fd);
            worker.socket_fd = -1;
            worker.busy = false;
            worker.current_frame = 0;
            continue;
        }

        worker.busy = true;
        worker.current_frame = frame.frame_index;
        if(worker_idx < m_dispatchTimesNs.size())
        {
            m_dispatchTimesNs[worker_idx] = nowNs();
        }
        m_nextWorker = (worker_idx + 1) % worker_count;

        LOGD("WorkerPool: dispatched frame %u to worker %d\n", frame.frame_index, worker.id);
        return 0;
    }

    return -1;
}

bool WorkerPool::pollResult(int& frame_index,
                            std::vector<float>& depth_data,
                            int& out_w,
                            int& out_h,
                            float& scale,
                            float& bias,
                            float& z_max)
{
    std::scoped_lock lock(m_mutex);
    for(size_t worker_idx = 0; worker_idx < m_workers.size(); ++worker_idx)
    {
        WorkerProcess& worker = m_workers[worker_idx];
        if(worker.socket_fd < 0 || !worker.busy)
        {
            continue;
        }

        pollfd pfd{};
        pfd.fd = worker.socket_fd;
        pfd.events = POLLIN;

        const int poll_rc = poll(&pfd, 1, 0);
        if(poll_rc < 0)
        {
            if(errno == EINTR)
            {
                continue;
            }
            LOGE("WorkerPool: poll failed for worker %d: %s\n", worker.id, std::strerror(errno));
            continue;
        }
        if(poll_rc == 0 || (pfd.revents & POLLIN) == 0)
        {
            continue;
        }

        const bool ok = readResult(static_cast<int>(worker_idx), frame_index, depth_data, out_w, out_h, scale, bias, z_max);
        worker.busy = false;
        worker.current_frame = 0;
        if(worker_idx < m_dispatchTimesNs.size() && m_dispatchTimesNs[worker_idx] != 0)
        {
            const double elapsed_ms = static_cast<double>(nowNs() - m_dispatchTimesNs[worker_idx]) / 1000000.0;
            m_totalProcessingMs += elapsed_ms;
            ++m_completedFrames;
            m_dispatchTimesNs[worker_idx] = 0;
        }

        if(!m_pendingQueue.empty())
        {
            PendingFrame next_frame = std::move(m_pendingQueue.front());
            m_pendingQueue.pop_front();
            if(dispatchFrame(next_frame) != 0)
            {
                m_pendingQueue.push_front(std::move(next_frame));
            }
        }

        return ok;
    }

    return false;
}

bool WorkerPool::readResult(int worker_idx,
                            int& frame_index,
                            std::vector<float>& depth_data,
                            int& out_w,
                            int& out_h,
                            float& scale,
                            float& bias,
                            float& z_max)
{
    if(worker_idx < 0 || static_cast<size_t>(worker_idx) >= m_workers.size())
    {
        return false;
    }

    WorkerProcess& worker = m_workers[static_cast<size_t>(worker_idx)];
    FrameHeader header{};
    if(!readAll(worker.socket_fd, &header, sizeof(header)))
    {
        LOGE("WorkerPool: failed reading response header from worker %d\n", worker.id);
        return false;
    }
    if(!validateHeader(&header, sizeof(header)) || header.message_type != MSG_DEPTH_RESPONSE)
    {
        LOGE("WorkerPool: invalid response header from worker %d (type=%s)\n", worker.id,
             getMessageTypeName(header.message_type));
        return false;
    }

    DepthResponsePayload payload{};
    if(!readAll(worker.socket_fd, &payload, sizeof(payload)))
    {
        LOGE("WorkerPool: failed reading response payload from worker %d\n", worker.id);
        return false;
    }

    const size_t depth_count = static_cast<size_t>(payload.width) * static_cast<size_t>(payload.height);
    depth_data.resize(depth_count);
    if(depth_count != 0 && !readAll(worker.socket_fd, depth_data.data(), depth_count * sizeof(float)))
    {
        LOGE("WorkerPool: failed reading depth buffer from worker %d\n", worker.id);
        depth_data.clear();
        return false;
    }

    frame_index = static_cast<int>(worker.current_frame);
    out_w = static_cast<int>(payload.width);
    out_h = static_cast<int>(payload.height);
    scale = payload.scale;
    bias = payload.bias;
    z_max = payload.z_max;
    return true;
}

void WorkerPool::shutdown()
{
    std::scoped_lock lock(m_mutex);
    for(WorkerProcess& worker : m_workers)
    {
        if(worker.socket_fd >= 0)
        {
            FrameHeader header{};
            header.magic = TCP_DEPTH_MAGIC;
            header.frame_length = sizeof(FrameHeader);
            header.message_type = MSG_SHUTDOWN;
            if(!writeAll(worker.socket_fd, &header, sizeof(header)))
            {
                LOGD("WorkerPool: shutdown send failed for worker %d\n", worker.id);
            }
            close(worker.socket_fd);
            worker.socket_fd = -1;
        }
    }

    for(WorkerProcess& worker : m_workers)
    {
        if(worker.pid > 0)
        {
            int status = 0;
            if(waitpid(worker.pid, &status, 0) < 0)
            {
                LOGE("WorkerPool: waitpid failed for worker %d: %s\n", worker.id, std::strerror(errno));
            }
        }
    }

    m_workers.clear();
    m_pendingQueue.clear();
    m_dispatchTimesNs.clear();
    m_nextWorker = 0;
    m_numWorkers = 0;
    m_totalProcessingMs = 0.0;
    m_completedFrames = 0;
}

int WorkerPool::activeWorkers() const
{
    std::scoped_lock lock(m_mutex);
    int active = 0;
    for(const WorkerProcess& worker : m_workers)
    {
        if(worker.socket_fd >= 0)
        {
            ++active;
        }
    }
    return active;
}

size_t WorkerPool::workerCount() const
{
    std::scoped_lock lock(m_mutex);
    return m_workers.size();
}

pid_t WorkerPool::workerPid(size_t index) const
{
    std::scoped_lock lock(m_mutex);
    if(index >= m_workers.size())
    {
        return -1;
    }
    return m_workers[index].pid;
}

std::string WorkerPool::workerDevice(size_t index) const
{
    std::scoped_lock lock(m_mutex);
    if(index >= m_workers.size())
    {
        return {};
    }
    return m_workers[index].device;
}

bool WorkerPool::restartWorker(size_t index)
{
    std::scoped_lock lock(m_mutex);
    if(index >= m_workers.size())
    {
        return false;
    }

    WorkerProcess& worker = m_workers[index];
    const int old_pid = worker.pid;
    const std::string device = worker.device;

    if(worker.socket_fd >= 0)
    {
        close(worker.socket_fd);
        worker.socket_fd = -1;
    }
    worker.busy = false;
    worker.current_frame = 0;
    worker.pending_depth_fd = -1;

    WorkerProcess replacement{};
    if(!spawnWorker(worker.id, device, replacement))
    {
        worker.pid = -1;
        return false;
    }

    m_workers[index] = std::move(replacement);
    LOGI("WorkerPool: restarted worker %d (old pid=%d, new pid=%d, device=%s)\n", worker.id, old_pid,
         static_cast<int>(m_workers[index].pid), device.c_str());
    return true;
}

void WorkerPool::retireWorker(size_t index)
{
    std::scoped_lock lock(m_mutex);
    if(index >= m_workers.size())
    {
        return;
    }

    WorkerProcess& worker = m_workers[index];
    if(worker.socket_fd >= 0)
    {
        close(worker.socket_fd);
        worker.socket_fd = -1;
    }
    worker.pid = -1;
    worker.busy = false;
    worker.current_frame = 0;
    worker.pending_depth_fd = -1;
}

int WorkerPool::queueDepth() const
{
    std::scoped_lock lock(m_mutex);
    return static_cast<int>(m_pendingQueue.size());
}

float WorkerPool::avgProcessingMs() const
{
    std::scoped_lock lock(m_mutex);
    if(m_completedFrames == 0)
    {
        return 0.0f;
    }
    return static_cast<float>(m_totalProcessingMs / static_cast<double>(m_completedFrames));
}

std::vector<int> WorkerPool::getWorkerLoads() const
{
    std::scoped_lock lock(m_mutex);
    std::vector<int> loads;
    loads.reserve(m_workers.size());
    for(const WorkerProcess& worker : m_workers)
    {
        loads.push_back(worker.busy ? 1 : 0);
    }
    return loads;
}
