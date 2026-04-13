#ifndef _SUPERSPLAT_CLIENT_H_
#define _SUPERSPLAT_CLIENT_H_

#include <string>
#include <vector>
#include <functional>
#include <thread>
#include <mutex>
#include <atomic>

namespace vk_viewer {

class SupersplatClient
{
public:
  struct Scene
  {
    int         id;
    std::string title;
    std::string description;
    std::string thumbnailUrl;
    std::string viewUrl;  // https://superspl.at/view?id=...
    std::string author;
    int         views;
    int         likes;
    long long   size = 0;
  };

  using SceneListCallback  = std::function<void(const std::vector<Scene>& scenes)>;
  using ThumbnailCallback  = std::function<void(const std::vector<uint8_t>& data, int width, int height, int channels)>;
  using ContentUrlCallback = std::function<void(const std::string& contentUrl, bool success)>;

  SupersplatClient();
  ~SupersplatClient();

  void               fetchSceneList(const std::string& search, SceneListCallback callback);
  void               fetchThumbnail(const std::string& url, ThumbnailCallback callback);
  void               resolveContentUrl(const std::string& viewUrl, ContentUrlCallback callback);
  static bool        isSuperSplatUrl(const std::string& url);
  static std::string normalizeUrl(const std::string& url);
  static bool        resolveContentUrlSync(const std::string& viewUrl, std::string& outContentUrl);

private:
  void threadFunc(std::function<void()> task);

  // Helper for WinHTTP GET request (returns true on success, populates response)
  static bool httpGet(const std::string& url, std::vector<uint8_t>& response);

private:
  std::mutex               m_threadMutex;
  std::vector<std::thread> m_threads;
};

}  // namespace vk_viewer

#endif  // _SUPERSPLAT_CLIENT_H_
