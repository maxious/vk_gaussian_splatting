#include "supersplat_client.h"

#include <nvutils/logger.hpp>
#include <tinygltf/json.hpp>
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>
#include <webp/decode.h>

#ifdef _WIN32
#include <windows.h>
#include <winhttp.h>
#pragma comment(lib, "winhttp.lib")
#endif

namespace vk_gaussian_splatting {

SupersplatClient::SupersplatClient()
{
}

SupersplatClient::~SupersplatClient()
{
    for (auto& t : m_threads)
    {
        if (t.joinable())
            t.join();
    }
}

void SupersplatClient::threadFunc(std::function<void()> task)
{
    task();
    // In a real thread pool, we'd reuse threads. Here we just let them finish.
    // Detaching or joining is tricky with this simple design, but for now we'll assume the client lives long enough.
}

void SupersplatClient::fetchSceneList(const std::string& search, SceneListCallback callback)
{
    std::string url = "https://playcanvas.com/api/splats/explore?skip=0&limit=32&sort=starred&order=-1&time=month";
    if (!search.empty())
    {
        // Simple URL encoding for space
        std::string encodedSearch = search;
        size_t pos = 0;
        while ((pos = encodedSearch.find(' ', pos)) != std::string::npos) {
            encodedSearch.replace(pos, 1, "%20");
            pos += 3;
        }
        url += "&search=" + encodedSearch;
    }

    auto task = [this, url, callback]() {
        std::vector<uint8_t> responseData;
        if (httpGet(url, responseData))
        {
            try
            {
                auto json = nlohmann::json::parse(responseData.begin(), responseData.end());
                std::vector<Scene> scenes;

                if (json.contains("result") && json["result"].is_array())
                {
                    for (const auto& item : json["result"])
                    {
                        Scene scene;
                        scene.id = item.value("id", 0);
                        scene.title = item.value("title", "Untitled");
                        scene.description = item.value("description", "");
                        scene.views = item.value("views", 0);
                        scene.likes = item.value("starred", 0);
                        scene.size = item.value("size", 0LL);

                        if (item.contains("thumbnails"))
                        {
                            auto thumbs = item["thumbnails"];
                            // Prefer 'm' size, fallback to 's', 'l', 'xl'
                            if (thumbs.contains("m")) scene.thumbnailUrl = thumbs["m"];
                            else if (thumbs.contains("s")) scene.thumbnailUrl = thumbs["s"];
                            else if (thumbs.contains("l")) scene.thumbnailUrl = thumbs["l"];
                            else if (thumbs.contains("xl")) scene.thumbnailUrl = thumbs["xl"];
                        }

                        if (item.contains("url"))
                        {
                             // item["url"] is like "/view?id=..."
                             scene.viewUrl = "https://superspl.at" + item.value("url", "");
                        }

                        if (item.contains("user"))
                        {
                            scene.author = item["user"].value("username", "Unknown");
                        }

                        scenes.push_back(scene);
                    }
                }
                
                if (callback) callback(scenes);
            }
            catch (const std::exception& e)
            {
                LOGE("SupersplatClient: JSON parse error: %s\n", e.what());
                if (callback) callback({});
            }
        }
        else
        {
            LOGE("SupersplatClient: Failed to fetch scene list\n");
            if (callback) callback({});
        }
    };

    std::lock_guard<std::mutex> lock(m_threadMutex);
    m_threads.emplace_back(task);
}

void SupersplatClient::fetchThumbnail(const std::string& url, ThumbnailCallback callback)
{
    auto task = [this, url, callback]() {
        std::vector<uint8_t> responseData;
        if (httpGet(url, responseData))
        {
            int w, h, c;
            stbi_set_flip_vertically_on_load(0); // Don't flip for UI
            
            bool isWebP = false;
            size_t dotPos = url.find_last_of(".");
            if (dotPos != std::string::npos) {
                std::string ext = url.substr(dotPos);
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".webp") isWebP = true;
            }

            unsigned char* data = nullptr;
            if (!isWebP)
            {
                data = stbi_load_from_memory(responseData.data(), static_cast<int>(responseData.size()), &w, &h, &c, 4); // Force 4 channels (RGBA)
            }

            if (!data)
            {
                // Try decoding with WebPDecodeRGBA if stbi failed or if it was identified as WebP
                uint8_t* webpData = WebPDecodeRGBA(responseData.data(), responseData.size(), &w, &h);
                if (webpData)
                {
                    std::vector<uint8_t> imageData(webpData, webpData + (w * h * 4));
                    WebPFree(webpData);
                    if (callback) callback(imageData, w, h, 4);
                    return;
                }
            }
            
            if (data)
            {
                std::vector<uint8_t> imageData(data, data + (w * h * 4));
                stbi_image_free(data);
                if (callback) callback(imageData, w, h, 4);
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
#ifdef _WIN32
    URL_COMPONENTS urlComp;
    ZeroMemory(&urlComp, sizeof(urlComp));
    urlComp.dwStructSize = sizeof(urlComp);

    wchar_t hostName[256] = {0};
    wchar_t urlPath[2048] = {0};
    urlComp.lpszHostName = hostName;
    urlComp.dwHostNameLength = sizeof(hostName) / sizeof(wchar_t);
    urlComp.lpszUrlPath = urlPath;
    urlComp.dwUrlPathLength = sizeof(urlPath) / sizeof(wchar_t);

    std::wstring wideUrl(url.begin(), url.end());

    if (!WinHttpCrackUrl(wideUrl.c_str(), static_cast<DWORD>(wideUrl.length()), 0, &urlComp))
    {
        return false;
    }

    HINTERNET hSession = WinHttpOpen(L"SupersplatClient/1.0",
                                     WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                                     WINHTTP_NO_PROXY_NAME,
                                     WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession) return false;

    HINTERNET hConnect = WinHttpConnect(hSession, hostName, urlComp.nPort, 0);
    if (!hConnect)
    {
        WinHttpCloseHandle(hSession);
        return false;
    }

    DWORD dwFlags = (urlComp.nScheme == INTERNET_SCHEME_HTTPS) ? WINHTTP_FLAG_SECURE : 0;
    HINTERNET hRequest = WinHttpOpenRequest(hConnect, L"GET", urlPath,
                                            NULL, WINHTTP_NO_REFERER,
                                            WINHTTP_DEFAULT_ACCEPT_TYPES,
                                            dwFlags);
    if (!hRequest)
    {
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    if (!WinHttpSendRequest(hRequest, WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                            WINHTTP_NO_REQUEST_DATA, 0, 0, 0))
    {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    if (!WinHttpReceiveResponse(hRequest, NULL))
    {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    DWORD dwStatusCode = 0;
    DWORD dwSize = sizeof(dwStatusCode);
    WinHttpQueryHeaders(hRequest, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                        WINHTTP_HEADER_NAME_BY_INDEX, &dwStatusCode, &dwSize, WINHTTP_NO_HEADER_INDEX);

    if (dwStatusCode != 200)
    {
        WinHttpCloseHandle(hRequest);
        WinHttpCloseHandle(hConnect);
        WinHttpCloseHandle(hSession);
        return false;
    }

    DWORD dwSizeAvail = 0;
    DWORD dwDownloaded = 0;
    std::vector<char> buffer(8192);

    do
    {
        dwSizeAvail = 0;
        if (!WinHttpQueryDataAvailable(hRequest, &dwSizeAvail)) break;

        if (dwSizeAvail > 0)
        {
            if (dwSizeAvail > buffer.size()) buffer.resize(dwSizeAvail);

            if (WinHttpReadData(hRequest, buffer.data(), dwSizeAvail, &dwDownloaded))
            {
                response.insert(response.end(), buffer.data(), buffer.data() + dwDownloaded);
            }
            else
            {
                break;
            }
        }
    } while (dwSizeAvail > 0);

    WinHttpCloseHandle(hRequest);
    WinHttpCloseHandle(hConnect);
    WinHttpCloseHandle(hSession);
    return true;
#else
    LOGE("SupersplatClient: Only supported on Windows\n");
    return false;
#endif
}

} // namespace vk_gaussian_splatting
