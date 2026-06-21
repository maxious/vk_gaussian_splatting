#pragma once
#include <string>

// Resolve a model path/reference to a local file path.
// If path is a local file that exists, returns it as-is.
// If path is a HuggingFace reference "repo_id:filename", downloads and caches.
// Returns empty string on failure.
std::string resolveModelPath(const std::string& modelRef);

// Get cache directory (~/.cache/depth_server/)
std::string getModelCacheDir();

// Print the list of known available HF models (curated list).
void printAvailableModels();
