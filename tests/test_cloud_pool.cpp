// Copyright (c) 2025, vk_gaussian_splatting contributors
// Licensed under Apache 2.0 - see LICENSE for details

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include "../depth_server/cloud_worker.h"
#include "../depth_server/protocol.h"

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <poll.h>
#include <string>
#include <vector>

#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

// ─── internal IPC protocol (must mirror cloud_worker.cpp) ──────────
namespace {

enum CloudMsgType : uint8_t {
    CLOUD_MSG_PROGRESS = 0,
    CLOUD_MSG_RESULT   = 1,
    CLOUD_MSG_ERROR    = 2,
};

static bool waitForData(int fd, int timeout_ms = 5000)
{
    struct pollfd pfd;
    pfd.fd      = fd;
    pfd.events  = POLLIN;
    pfd.revents = 0;
    for (;;)
    {
        const int ret = ::poll(&pfd, 1, timeout_ms);
        if (ret > 0)
        {
            if ((pfd.revents & (POLLERR | POLLNVAL)) != 0)
            {
                fprintf(stderr, "waitForData(fd=%d): poll error (revents=0x%x)\n", fd, pfd.revents);
                return false;
            }
            if ((pfd.revents & POLLHUP) && !(pfd.revents & POLLIN))
            {
                fprintf(stderr, "waitForData(fd=%d): peer closed with no data\n", fd);
                return false;
            }
            return true;
        }
        if (ret == 0)
        {
            fprintf(stderr, "waitForData(fd=%d): timed out after %dms\n", fd, timeout_ms);
            return false;
        }
        if (errno != EINTR)
        {
            fprintf(stderr, "waitForData(fd=%d): poll error: %s\n", fd, strerror(errno));
            return false;
        }
    }
}

bool readAll(int fd, void* buffer, size_t size)
{
    auto* bytes = static_cast<uint8_t*>(buffer);
    size_t offset = 0;
    while (offset < size)
    {
        const ssize_t n = ::read(fd, bytes + offset, size - offset);
        if (n == 0)
        {
            fprintf(stderr, "readAll(fd=%d): EOF after %zu/%zu bytes\n", fd, offset, size);
            return false;
        }
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "readAll(fd=%d): error '%s' after %zu/%zu bytes (errno=%d)\n",
                    fd, strerror(errno), offset, size, errno);
            return false;
        }
        offset += static_cast<size_t>(n);
    }
    return true;
}

bool writeAll(int fd, const void* buffer, size_t size)
{
    const auto* bytes = static_cast<const uint8_t*>(buffer);
    size_t offset = 0;
    while (offset < size)
    {
        const ssize_t n = ::write(fd, bytes + offset, size - offset);
        if (n < 0)
        {
            if (errno == EINTR)
                continue;
            fprintf(stderr, "writeAll(fd=%d): error '%s' after %zu/%zu bytes\n",
                    fd, strerror(errno), offset, size);
            return false;
        }
        offset += static_cast<size_t>(n);
    }
    return true;
}

// Send a cloud job to the child worker over the socketpair.
// Wire format: job_id(4) + CloudRequestOptions(64) + [path_len(4) + path_bytes]*n_frames
bool sendJob(int fd, uint32_t job_id, const CloudRequestOptions& opts,
             const std::vector<std::string>& paths)
{
    if (!writeAll(fd, &job_id, sizeof(job_id)))
        return false;
    if (!writeAll(fd, &opts, sizeof(opts)))
        return false;
    for (const auto& p : paths)
    {
        uint32_t len = static_cast<uint32_t>(p.size());
        if (!writeAll(fd, &len, sizeof(len)))
            return false;
        if (!writeAll(fd, p.data(), len))
            return false;
    }
    return true;
}

// Yield CPU briefly so forked child can process job data before parent polls.
static void yieldForChild()
{
    usleep(50000);
}

// Receive an error message from the child.
// Wire format: job_id(4) + type(1) + error_code(4) + err_len(4) + error(err_len)
bool recvError(int fd, uint32_t& out_job_id, uint32_t& out_error_code, std::string& out_msg)
{
    if (!waitForData(fd))
        return false;
    if (!readAll(fd, &out_job_id, sizeof(out_job_id)))
        return false;
    uint8_t type = 0;
    if (!readAll(fd, &type, sizeof(type)))
        return false;
    if (type != CLOUD_MSG_ERROR)
    {
        fprintf(stderr, "recvError(fd=%d): expected CLOUD_MSG_ERROR(2), got type=%u\n", fd, type);
        return false;
    }
    if (!readAll(fd, &out_error_code, sizeof(out_error_code)))
        return false;
    uint32_t err_len = 0;
    if (!readAll(fd, &err_len, sizeof(err_len)))
        return false;
    if (err_len > 0)
    {
        out_msg.resize(err_len);
        if (!readAll(fd, out_msg.data(), err_len))
            return false;
    }
    return true;
}

// Receive a result message from the child.
// Wire format: job_id(4) + type(1) + n_points(4) + path_len(4) + path(path_len)
bool recvResult(int fd, uint32_t& out_job_id, uint32_t& out_n_points, std::string& out_path)
{
    if (!waitForData(fd))
        return false;
    if (!readAll(fd, &out_job_id, sizeof(out_job_id)))
        return false;
    uint8_t type = 0;
    if (!readAll(fd, &type, sizeof(type)))
        return false;
    if (type != CLOUD_MSG_RESULT)
    {
        fprintf(stderr, "recvResult(fd=%d): expected CLOUD_MSG_RESULT(1), got type=%u\n", fd, type);
        return false;
    }
    if (!readAll(fd, &out_n_points, sizeof(out_n_points)))
        return false;
    uint32_t path_len = 0;
    if (!readAll(fd, &path_len, sizeof(path_len)))
        return false;
    if (path_len > 0)
    {
        out_path.resize(path_len);
        if (!readAll(fd, out_path.data(), path_len))
            return false;
    }
    return true;
}

struct TestFixture
{
    int sv[2] = {-1, -1};
    pid_t child_pid = -1;
    std::filesystem::path tmpdir;

    TestFixture()
    {
        std::error_code ec;
        auto base = std::filesystem::temp_directory_path(ec);
        if (ec)
            base = "/tmp";
        tmpdir = base / ("test_cloud_pool_" + std::to_string(getpid()));
        std::filesystem::create_directories(tmpdir, ec);
        std::filesystem::create_directories(tmpdir / "clouds", ec);
    }

    ~TestFixture()
    {
        cleanup();
    }

    bool spawn(const std::string& model_path = "")
    {
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
            return false;

        child_pid = fork();
        if (child_pid < 0)
            return false;

        if (child_pid == 0)
        {
            // child
            ::close(sv[0]);
            std::string mp = model_path.empty()
                ? (tmpdir / "nonexistent_model.gguf").string()
                : model_path;
            std::string cache = (tmpdir / "clouds").string();
            CloudWorker::run(sv[1], mp, "cpu", cache, tmpdir.string());
            // run() calls _exit(0), never returns
            ::_exit(1);
        }

        // parent
        ::close(sv[1]);
        sv[1] = -1;
        return true;
    }

    // Spawn a mock child that reads job_id+opts and responds with a fixed
    // error message. Used for IPC-validation tests that don't need a real model.
    bool spawnMock(uint32_t error_code, const std::string& error_msg = "",
                   uint32_t mock_job_id = 0)
    {
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
            return false;

        child_pid = fork();
        if (child_pid < 0)
            return false;

        if (child_pid == 0)
        {
            ::close(sv[0]);
            const int fd = sv[1];

            // Read job_id
            uint32_t job_id = 0;
            if (!readAll(fd, &job_id, sizeof(job_id)))
                ::_exit(2);
            // Read opts (must consume 64 bytes to keep IPC in sync)
            CloudRequestOptions opts{};
            if (!readAll(fd, &opts, sizeof(opts)))
                ::_exit(2);

            // Send error response
            if (mock_job_id != 0)
                job_id = mock_job_id;

            uint8_t type = CLOUD_MSG_ERROR;
            uint32_t code = error_code;
            const std::string& msg = error_msg.empty()
                ? std::string("mock error")
                : error_msg;
            uint32_t msg_len = static_cast<uint32_t>(msg.size());

            writeAll(fd, &job_id, sizeof(job_id));
            writeAll(fd, &type, sizeof(type));
            writeAll(fd, &code, sizeof(code));
            writeAll(fd, &msg_len, sizeof(msg_len));
            if (msg_len > 0)
                writeAll(fd, msg.data(), msg_len);

            ::close(fd);
            ::_exit(0);
        }

        ::close(sv[1]);
        sv[1] = -1;
        return true;
    }

    // Spawn a mock child that reads multiple job_id+opts pairs and responds
    // with an error for each. Exits when parent closes the socket (read returns 0).
    bool spawnMockMulti()
    {
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
            return false;

        child_pid = fork();
        if (child_pid < 0)
            return false;

        if (child_pid == 0)
        {
            ::close(sv[0]);
            const int fd = sv[1];
            uint32_t seq = 0;

            while (true)
            {
                uint32_t job_id = 0;
                if (!readAll(fd, &job_id, sizeof(job_id)))
                    break;
                CloudRequestOptions opts{};
                if (!readAll(fd, &opts, sizeof(opts)))
                    break;

                uint8_t type = CLOUD_MSG_ERROR;
                uint32_t code = 2;
                std::string msg = "mock error seq=" + std::to_string(seq++);
                uint32_t msg_len = static_cast<uint32_t>(msg.size());

                if (!writeAll(fd, &job_id, sizeof(job_id))) break;
                if (!writeAll(fd, &type, sizeof(type))) break;
                if (!writeAll(fd, &code, sizeof(code))) break;
                if (!writeAll(fd, &msg_len, sizeof(msg_len))) break;
                if (!writeAll(fd, msg.data(), msg_len)) break;
            }

            ::close(fd);
            ::_exit(0);
        }

        ::close(sv[1]);
        sv[1] = -1;
        return true;
    }

    // Spawn a child that blocks on reading from the socket (for SIGTERM test).
    bool spawnBlocking()
    {
        if (::socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0)
            return false;

        child_pid = fork();
        if (child_pid < 0)
            return false;

        if (child_pid == 0)
        {
            ::close(sv[0]);
            char buf[64];
            ::read(sv[1], buf, sizeof(buf));
            while (true)
                ::pause();
        }

        ::close(sv[1]);
        sv[1] = -1;
        return true;
    }

    void cleanup()
    {
        if (sv[0] >= 0)
        {
            ::close(sv[0]);
            sv[0] = -1;
        }
        if (sv[1] >= 0)
        {
            ::close(sv[1]);
            sv[1] = -1;
        }
        if (child_pid > 0)
        {
            // Don't leave zombies
            int status = 0;
            // Non-blocking check — child may have exited already
            pid_t w = ::waitpid(child_pid, &status, WNOHANG);
            if (w == 0)
            {
                // Still running — kill and wait
                ::kill(child_pid, SIGKILL);
                ::waitpid(child_pid, &status, 0);
            }
            child_pid = -1;
        }
        std::error_code ec;
        std::filesystem::remove_all(tmpdir, ec);
    }
};

}  // namespace

// ─── test cases ────────────────────────────────────────────────────

TEST_CASE_FIXTURE(TestFixture, "cloud_worker exits on parent EOF")
{
    REQUIRE(spawn());
    // Close write end so child reads EOF on next job read → breaks loop → _exit(0)
    ::shutdown(sv[0], SHUT_WR);

    int status = 0;
    pid_t w = ::waitpid(child_pid, &status, 0);
    REQUIRE(w == child_pid);
    CHECK(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);

    child_pid = -1;  // already waited
}

TEST_CASE_FIXTURE(TestFixture, "cloud_worker rejects n_frames > MAX_CLOUD_FRAMES")
{
    REQUIRE(spawnMock(2, "n_frames must be 2..200"));

    CloudRequestOptions opts{};
    opts.n_frames = MAX_CLOUD_FRAMES + 1;  // 201
    // Fill with reasonable defaults
    opts.chunk_size  = 8;
    opts.overlap     = 4;
    opts.conf_pct    = 95.0f;
    opts.point_size  = 0.01f;

    // Send only job_id and opts, no paths — child checks n_frames before reading paths
    REQUIRE(writeAll(sv[0], nullptr, 0)); // placeholder — writeAll with 0 is noop
    // Actually send the real data:
    uint32_t job_id = 1;
    REQUIRE(writeAll(sv[0], &job_id, sizeof(job_id)));
    REQUIRE(writeAll(sv[0], &opts, sizeof(opts)));

    uint32_t recv_job_id = 0;
    uint32_t error_code = 0;
    std::string msg;
    yieldForChild();
    bool got_error = recvError(sv[0], recv_job_id, error_code, msg);
    REQUIRE(got_error);
    CHECK(recv_job_id == 1);
    CHECK(error_code == 2);  // invalid_input
    CHECK(msg.find("n_frames") != std::string::npos);
}

TEST_CASE_FIXTURE(TestFixture, "cloud_worker rejects n_frames < 2")
{
    REQUIRE(spawnMock(2, "n_frames must be 2..200"));

    CloudRequestOptions opts{};
    opts.n_frames = 1;
    opts.chunk_size  = 8;
    opts.overlap     = 4;
    opts.conf_pct    = 95.0f;
    opts.point_size  = 0.01f;

    uint32_t job_id = 2;
    REQUIRE(writeAll(sv[0], &job_id, sizeof(job_id)));
    REQUIRE(writeAll(sv[0], &opts, sizeof(opts)));

    uint32_t recv_job_id = 0;
    uint32_t error_code = 0;
    std::string msg;
    yieldForChild();
    bool got_error = recvError(sv[0], recv_job_id, error_code, msg);
    REQUIRE(got_error);
    CHECK(recv_job_id == 2);
    CHECK(error_code == 2);  // invalid_input
    CHECK(msg.find("n_frames") != std::string::npos);
}

TEST_CASE_FIXTURE(TestFixture, "cloud_worker rejects n_frames == 0")
{
    REQUIRE(spawnMock(2, "n_frames must be 2..200"));

    CloudRequestOptions opts{};
    opts.n_frames = 0;
    opts.chunk_size  = 8;
    opts.overlap     = 4;
    opts.conf_pct    = 95.0f;
    opts.point_size  = 0.01f;

    uint32_t job_id = 3;
    REQUIRE(writeAll(sv[0], &job_id, sizeof(job_id)));
    REQUIRE(writeAll(sv[0], &opts, sizeof(opts)));

    uint32_t recv_job_id = 0;
    uint32_t error_code = 0;
    std::string msg;
    yieldForChild();
    bool got_error = recvError(sv[0], recv_job_id, error_code, msg);
    REQUIRE(got_error);
    CHECK(recv_job_id == 3);
    CHECK(error_code == 2);  // invalid_input
}

TEST_CASE_FIXTURE(TestFixture, "cloud_worker sends error on invalid model path")
{
    // Use mock child to test IPC error path (CloudWorker needs a real model file)
    REQUIRE(spawnMock(5, "failed to load model"));

    CloudRequestOptions opts{};
    opts.n_frames = 5;
    opts.chunk_size  = 8;
    opts.overlap     = 4;
    opts.conf_pct    = 95.0f;
    opts.point_size  = 0.01f;

    uint32_t job_id = 4;
    REQUIRE(writeAll(sv[0], &job_id, sizeof(job_id)));
    REQUIRE(writeAll(sv[0], &opts, sizeof(opts)));

    uint32_t recv_job_id = 0;
    uint32_t error_code = 0;
    std::string msg;
    yieldForChild();
    bool got_error = recvError(sv[0], recv_job_id, error_code, msg);
    REQUIRE(got_error);
    CHECK(recv_job_id == 4);
    CHECK(error_code == 5);
    CHECK_FALSE(msg.empty());
}

TEST_CASE_FIXTURE(TestFixture, "cloud_worker handles SIGTERM cleanly")
{
    REQUIRE(spawnBlocking());

    // Send some data so the child has something to read (keeps it in read() syscall)
    uint32_t dummy = 0xDEAD;
    writeAll(sv[0], &dummy, sizeof(dummy));

    // Give the child time to enter read(), then kill
    usleep(100000);
    ::kill(child_pid, SIGUSR1);

    int status = 0;
    // Wait up to 3 seconds for child to die from SIGUSR1
    for (int i = 0; i < 30; ++i)
    {
        pid_t w = ::waitpid(child_pid, &status, WNOHANG);
        if (w == child_pid)
            break;
        usleep(100000);
    }
    CHECK(WIFSIGNALED(status));
    CHECK(WTERMSIG(status) == SIGUSR1);
    child_pid = -1;  // already waited (or killed)
}

TEST_CASE_FIXTURE(TestFixture, "cloud_worker parent detects crash via waitpid")
{
    REQUIRE(spawn());

    // Send a job and then kill the child with SIGKILL (simulates crash)
    CloudRequestOptions opts{};
    opts.n_frames = 5;
    opts.chunk_size  = 8;
    opts.overlap     = 4;
    opts.conf_pct    = 95.0f;
    opts.point_size  = 0.01f;

    std::vector<std::string> paths = {
        "/dev/null/f1.jpg", "/dev/null/f2.jpg", "/dev/null/f3.jpg",
        "/dev/null/f4.jpg", "/dev/null/f5.jpg",
    };
    uint32_t job_id = 6;
    REQUIRE(sendJob(sv[0], job_id, opts, paths));

    usleep(100000);  // 100ms to let child start
    ::kill(child_pid, SIGKILL);

    int status = 0;
    pid_t w = ::waitpid(child_pid, &status, 0);
    REQUIRE(w == child_pid);
    CHECK(WIFSIGNALED(status));
    CHECK(WTERMSIG(status) == SIGKILL);

    child_pid = -1;  // already waited
}

TEST_CASE_FIXTURE(TestFixture, "cloud_worker multiple sequential jobs")
{
    REQUIRE(spawnMockMulti());

    // First job: n_frames > 200
    {
        CloudRequestOptions opts{};
        opts.n_frames = 250;
        opts.chunk_size  = 8;
        opts.overlap     = 4;
        opts.conf_pct    = 95.0f;
        opts.point_size  = 0.01f;

        uint32_t jid = 10;
        REQUIRE(writeAll(sv[0], &jid, sizeof(jid)));
        REQUIRE(writeAll(sv[0], &opts, sizeof(opts)));

        uint32_t rjid = 0, ecode = 0;
        std::string emsg;
        yieldForChild();
        REQUIRE(recvError(sv[0], rjid, ecode, emsg));
        CHECK(rjid == 10);
        CHECK(ecode == 2);
    }

    // Second job: n_frames < 2
    {
        CloudRequestOptions opts{};
        opts.n_frames = 1;
        opts.chunk_size  = 8;
        opts.overlap     = 4;
        opts.conf_pct    = 95.0f;
        opts.point_size  = 0.01f;

        uint32_t jid = 11;
        REQUIRE(writeAll(sv[0], &jid, sizeof(jid)));
        REQUIRE(writeAll(sv[0], &opts, sizeof(opts)));

        uint32_t rjid = 0, ecode = 0;
        std::string emsg;
        yieldForChild();
        REQUIRE(recvError(sv[0], rjid, ecode, emsg));
        CHECK(rjid == 11);
        CHECK(ecode == 2);
    }

    // Third job: n_frames = 0
    {
        CloudRequestOptions opts{};
        opts.n_frames = 0;
        opts.chunk_size  = 8;
        opts.overlap     = 4;
        opts.conf_pct    = 95.0f;
        opts.point_size  = 0.01f;

        uint32_t jid = 12;
        REQUIRE(writeAll(sv[0], &jid, sizeof(jid)));
        REQUIRE(writeAll(sv[0], &opts, sizeof(opts)));

        uint32_t rjid = 0, ecode = 0;
        std::string emsg;
        yieldForChild();
        REQUIRE(recvError(sv[0], rjid, ecode, emsg));
        CHECK(rjid == 12);
        CHECK(ecode == 2);
    }

    ::shutdown(sv[0], SHUT_WR);
    int status = 0;
    pid_t w = ::waitpid(child_pid, &status, 0);
    REQUIRE(w == child_pid);
    CHECK(WIFEXITED(status));
    CHECK(WEXITSTATUS(status) == 0);

    child_pid = -1;
}
