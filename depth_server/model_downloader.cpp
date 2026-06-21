#include "model_downloader.h"
#include <nvutils/logger.hpp>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

std::string getModelCacheDir()
{
    const char* home = getenv("HOME");
    if(!home) home = getenv("USERPROFILE");
    if(!home) return "/tmp/depth_server_cache";
    return std::string(home) + "/.cache/depth_server";
}

void printAvailableModels()
{
    LOGI("Available HuggingFace models (repo:filename):\n");
    LOGI("  mudler/depth-anything.cpp-gguf:depth-anything-base-q4_k.gguf (~99 MB, fast)\n");
    LOGI("  mudler/depth-anything.cpp-gguf:depth-anything-base-q8_0.gguf (~142 MB, accurate)\n");
    LOGI("  mudler/depth-anything.cpp-gguf:depth-anything-base-f16.gguf (~233 MB, GPU)\n");
    LOGI("  mudler/depth-anything.cpp-gguf:depth-anything-large-f32.gguf\n");
    LOGI("  mudler/depth-anything.cpp-gguf:depth-anything-metric-large-f32.gguf\n");
    LOGI("  mudler/depth-anything.cpp-gguf:depth-anything-nested-anyview.gguf\n");
    LOGI("  mudler/depth-anything.cpp-gguf:depth-anything-nested-metric.gguf\n");
}

static bool looksLikeHfReference(const std::string& s)
{
    // A HF reference contains exactly one colon separating repo_id from filename.
    // The repo_id has form "owner/name" (one slash, no slashes in filename part).
    size_t colon = s.find(':');
    if(colon == std::string::npos || colon == 0 || colon == s.size() - 1)
        return false;
    size_t slash = s.find('/');
    if(slash == std::string::npos || slash >= colon)
        return false;
    if(s.find('/', slash + 1) != std::string::npos)
        return false;  // repo_id has more than one slash
    if(s.find('\\') != std::string::npos)
        return false;
    return true;
}

std::string resolveModelPath(const std::string& modelRef)
{
    if(modelRef.empty())
    {
        LOGE("Model reference is empty\n");
        return "";
    }

    if(fs::exists(modelRef))
    {
        LOGI("Model: using local file %s\n", modelRef.c_str());
        return modelRef;
    }

    std::string localPath = fs::current_path().string() + "/" + modelRef;
    if(fs::exists(localPath))
    {
        LOGI("Model: using local file %s\n", localPath.c_str());
        return localPath;
    }

    if(!looksLikeHfReference(modelRef))
    {
        LOGE("Model not found and not a HF reference (use repo_id:filename): %s\n", modelRef.c_str());
        return "";
    }

    size_t      colon   = modelRef.find(':');
    std::string repoId  = modelRef.substr(0, colon);
    std::string fileName = modelRef.substr(colon + 1);

    std::string cacheDir  = getModelCacheDir() + "/" + repoId;
    std::string cachePath = cacheDir + "/" + fileName;

    if(fs::exists(cachePath))
    {
        LOGI("Model: using cached %s (%zu MB)\n", cachePath.c_str(), fs::file_size(cachePath) / 1024 / 1024);
        return cachePath;
    }

    std::error_code ec;
    fs::create_directories(cacheDir, ec);
    if(ec)
    {
        LOGE("Model: failed to create cache dir %s: %s\n", cacheDir.c_str(), ec.message().c_str());
        return "";
    }

    std::string url = "https://huggingface.co/" + repoId + "/resolve/main/" + fileName;
    LOGI("Model: downloading %s -> %s\n", url.c_str(), cachePath.c_str());

    std::string tmpPath = cachePath + ".tmp";
    std::string cmd     = "curl -fL --progress-bar -o '" + tmpPath + "' '" + url + "' 2>&1";
    LOGI("Model: running: %s\n", cmd.c_str());
    int ret = system(cmd.c_str());

    if(ret != 0 || !fs::exists(tmpPath))
    {
        LOGE("Model: download failed (curl exit %d)\n", ret);
        std::error_code rmec;
        fs::remove(tmpPath, rmec);
        return "";
    }

    std::error_code renec;
    fs::rename(tmpPath, cachePath, renec);
    if(renec)
    {
        LOGE("Model: failed to rename %s -> %s: %s\n", tmpPath.c_str(), cachePath.c_str(), renec.message().c_str());
        return "";
    }

    size_t sizeMB = fs::file_size(cachePath) / 1024 / 1024;
    LOGI("Model: downloaded %s (%zu MB)\n", fileName.c_str(), sizeMB);
    return cachePath;
}
