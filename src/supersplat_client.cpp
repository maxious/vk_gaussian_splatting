#include "supersplat_client.h"

#include <nvutils/logger.hpp>
#include <tinygltf/json.hpp>
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <webp/decode.h>
#include <cstring>
#include <algorithm>

#include <ixwebsocket/IXHttpClient.h>
#include <ixwebsocket/IXNetSystem.h>

namespace vk_viewer {

SupersplatClient::SupersplatClient()
{
  ix::initNetSystem();
}

SupersplatClient::~SupersplatClient()
{
  for(auto& t : m_threads)
  {
    if(t.joinable())
      t.join();
  }
}

void SupersplatClient::threadFunc(std::function<void()> task)
{
  task();
}

void SupersplatClient::fetchSceneList(const std::string& search, SceneListCallback callback)
{
  std::string url = "https://playcanvas.com/api/splats/explore?skip=0&limit=32&sort=starred&order=-1&time=month";
  if(!search.empty())
  {
    // Simple URL encoding for space
    std::string encodedSearch = search;
    size_t      pos           = 0;
    while((pos = encodedSearch.find(' ', pos)) != std::string::npos)
    {
      encodedSearch.replace(pos, 1, "%20");
      pos += 3;
    }
    url += "&search=" + encodedSearch;
  }

  auto task = [this, url, callback]() {
    std::vector<uint8_t> responseData;
    if(httpGet(url, responseData))
    {
      try
      {
        auto               json = nlohmann::json::parse(responseData.begin(), responseData.end());
        std::vector<Scene> scenes;

        if(json.contains("result") && json["result"].is_array())
        {
          for(const auto& item : json["result"])
          {
            Scene scene;
            scene.id          = item.value("id", 0);
            scene.title       = item.value("title", "Untitled");
            scene.description = item.value("description", "");
            scene.views       = item.value("views", 0);
            scene.likes       = item.value("starred", 0);
            scene.size        = item.value("size", 0LL);

            if(item.contains("thumbnails"))
            {
              auto thumbs = item["thumbnails"];
              // Prefer 'm' size, fallback to 's', 'l', 'xl'
              if(thumbs.contains("m"))
                scene.thumbnailUrl = thumbs["m"];
              else if(thumbs.contains("s"))
                scene.thumbnailUrl = thumbs["s"];
              else if(thumbs.contains("l"))
                scene.thumbnailUrl = thumbs["l"];
              else if(thumbs.contains("xl"))
                scene.thumbnailUrl = thumbs["xl"];
            }

            if(item.contains("url"))
            {
              // item["url"] is like "/view?id=..."
              scene.viewUrl = "https://superspl.at" + item.value("url", "");
            }

            if(item.contains("user"))
            {
              scene.author = item["user"].value("username", "Unknown");
            }

            scenes.push_back(scene);
          }
        }

        if(callback)
          callback(scenes);
      }
      catch(const std::exception& e)
      {
        LOGE("SupersplatClient: JSON parse error: %s\n", e.what());
        if(callback)
          callback({});
      }
    }
    else
    {
      LOGE("SupersplatClient: Failed to fetch scene list\n");
      if(callback)
        callback({});
    }
  };

  std::lock_guard<std::mutex> lock(m_threadMutex);
  m_threads.emplace_back(task);
}

void SupersplatClient::fetchThumbnail(const std::string& url, ThumbnailCallback callback)
{
  auto task = [this, url, callback]() {
    std::vector<uint8_t> responseData;
    if(httpGet(url, responseData))
    {
      int w, h, c;
      stbi_set_flip_vertically_on_load(0);  // Don't flip for UI

      bool   isWebP = false;
      size_t dotPos = url.find_last_of(".");
      if(dotPos != std::string::npos)
      {
        std::string ext = url.substr(dotPos);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if(ext == ".webp")
          isWebP = true;
      }

      unsigned char* data = nullptr;
      if(!isWebP)
      {
        data = stbi_load_from_memory(responseData.data(), static_cast<int>(responseData.size()), &w, &h, &c, 4);  // Force 4 channels (RGBA)
      }

      if(!data)
      {
        // Try decoding with WebPDecodeRGBA if stbi failed or if it was identified as WebP
        uint8_t* webpData = WebPDecodeRGBA(responseData.data(), responseData.size(), &w, &h);
        if(webpData)
        {
          std::vector<uint8_t> imageData(webpData, webpData + (w * h * 4));
          WebPFree(webpData);
          if(callback)
            callback(imageData, w, h, 4);
          return;
        }
      }

      if(data)
      {
        std::vector<uint8_t> imageData(data, data + (w * h * 4));
        stbi_image_free(data);
        if(callback)
          callback(imageData, w, h, 4);
      }
      else
      {
        LOGE("SupersplatClient: Failed to decode image: %s\n", url.c_str());
      }
    }
    else
    {
      LOGE("SupersplatClient: Failed to fetch thumbnail: %s\n", url.c_str());
    }
  };

  std::lock_guard<std::mutex> lock(m_threadMutex);
  m_threads.emplace_back(task);
}

bool SupersplatClient::httpGet(const std::string& url, std::vector<uint8_t>& response)
{
  ix::HttpClient httpClient;
  auto           args = httpClient.createRequest(url, ix::HttpClient::kGet);

  args->compress = true;

  auto res = httpClient.get(url, args);

  if(res->errorCode != ix::HttpErrorCode::Ok)
  {
    LOGE("SupersplatClient: HTTP error: %s\n", res->errorMsg.c_str());
    return false;
  }

  if(res->statusCode != 200)
  {
    LOGE("SupersplatClient: HTTP status error: %d\n", res->statusCode);
    return false;
  }

  response.assign(res->body.begin(), res->body.end());
  return true;
}

bool SupersplatClient::isSuperSplatUrl(const std::string& url)
{
  if(url.find("superspl.at") == std::string::npos)
  {
    return false;
  }

  if(url.find("/s?id=") != std::string::npos)
  {
    return true;
  }
  if(url.find("/view?id=") != std::string::npos)
  {
    return true;
  }
  if(url.find("/scene/") != std::string::npos)
  {
    return true;
  }
  return false;
}

std::string SupersplatClient::normalizeUrl(const std::string& url)
{
  std::string normalized = url;

  size_t scenePos = normalized.find("/scene/");
  if(scenePos != std::string::npos)
  {
    std::string id     = normalized.substr(scenePos + 7);
    size_t      endPos = id.find_first_of("?&");
    if(endPos != std::string::npos)
    {
      id = id.substr(0, endPos);
    }
    return "https://superspl.at/s?id=" + id;
  }

  if(normalized.find("/view?id=") != std::string::npos)
  {
    size_t      idPos  = normalized.find("/view?id=");
    std::string id     = normalized.substr(idPos + 9);
    size_t      endPos = id.find_first_of("?&");
    if(endPos != std::string::npos)
    {
      id = id.substr(0, endPos);
    }
    return "https://superspl.at/s?id=" + id;
  }

  return normalized;
}

void SupersplatClient::resolveContentUrl(const std::string& viewUrl, ContentUrlCallback callback)
{
  auto task = [this, viewUrl, callback]() {
    std::string normalizedUrl = normalizeUrl(viewUrl);
    LOGI("Resolving SuperSplat URL: %s\n", normalizedUrl.c_str());

    std::vector<uint8_t> responseData;
    if(!httpGet(normalizedUrl, responseData))
    {
      LOGE("SupersplatClient: Failed to fetch SuperSplat page\n");
      if(callback)
        callback("", false);
      return;
    }

    std::string html(responseData.begin(), responseData.end());

    const char* patterns[] = {R"(const\s+contentUrl\s*=\s*['"]([^'"]+)['"])", R"(data-content-url\s*=\s*["']([^"']+)["'])",
                              R"(window\.contentUrl\s*=\s*['"]([^'"]+)['"])"};

    std::string contentUrl;
    for(const auto& pattern : patterns)
    {
      size_t      patternLen = 0;
      const char* p          = pattern;
      while(*p)
      {
        if(*p != '(' && *p != '[')
          patternLen++;
        p++;
      }

      size_t matchStart = html.find("contentUrl");
      if(matchStart != std::string::npos)
      {
        size_t valueStart = html.find('"', matchStart);
        if(valueStart == std::string::npos)
          valueStart = html.find('\'', matchStart);
        if(valueStart != std::string::npos)
        {
          valueStart++;
          size_t valueEnd = html.find('"', valueStart);
          if(valueEnd == std::string::npos)
            valueEnd = html.find('\'', valueStart);
          if(valueEnd != std::string::npos)
          {
            contentUrl = html.substr(valueStart, valueEnd - valueStart);
            break;
          }
        }
      }
    }

    if(contentUrl.empty())
    {
      LOGE("SupersplatClient: Could not find content URL in page (URL format may have changed)\n");
      if(callback)
        callback("", false);
      return;
    }

    if(callback)
      callback(contentUrl, true);
  };

  std::lock_guard<std::mutex> lock(m_threadMutex);
  m_threads.emplace_back(task);
}

bool SupersplatClient::resolveContentUrlSync(const std::string& viewUrl, std::string& outContentUrl)
{
  std::string normalizedUrl = normalizeUrl(viewUrl);
  LOGI("Resolving SuperSplat URL: %s\n", normalizedUrl.c_str());

  std::vector<uint8_t> responseData;
  if(!httpGet(normalizedUrl, responseData))
  {
    LOGE("SupersplatClient: Failed to fetch SuperSplat page\n");
    return false;
  }

  std::string html(responseData.begin(), responseData.end());

  size_t searchStart = html.find("contentUrl");
  if(searchStart == std::string::npos)
  {
    LOGE("SupersplatClient: Could not find contentUrl in page\n");
    return false;
  }

  size_t valueStart = html.find('"', searchStart);
  if(valueStart == std::string::npos)
    valueStart = html.find('\'', searchStart);
  if(valueStart == std::string::npos)
  {
    LOGE("SupersplatClient: Could not parse contentUrl value\n");
    return false;
  }
  valueStart++;

  size_t valueEnd = html.find('"', valueStart);
  if(valueEnd == std::string::npos)
    valueEnd = html.find('\'', valueStart);
  if(valueEnd == std::string::npos)
  {
    LOGE("SupersplatClient: Could not find end of contentUrl\n");
    return false;
  }

  outContentUrl = html.substr(valueStart, valueEnd - valueStart);
  LOGI("Resolved to: %s\n", outContentUrl.c_str());
  return true;
}

}  // namespace vk_viewer
