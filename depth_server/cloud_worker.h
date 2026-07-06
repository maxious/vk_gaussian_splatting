#pragma once

#include <string>

class CloudWorker {
public:
    static void run(int parent_fd, const std::string& model_path,
                    const std::string& backend, const std::string& cloud_cache_dir,
                    const std::string& work_dir);
};
