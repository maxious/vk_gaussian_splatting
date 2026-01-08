#include "gltf_loader.h"
#include <nvutils/logger.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <webp/decode.h>
#include <cstring>
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

// Helper to decode image data - tries stbi first, falls back to WebP
static bool decodeImageData(const uint8_t* data, size_t size, TextureData& texData)
{
  // Try stbi first (handles PNG, JPEG, BMP, etc.)
  int w, h, c;
  uint8_t* pixels = stbi_load_from_memory(data, static_cast<int>(size), &w, &h, &c, 4);
  if(pixels)
  {
    texData.width = static_cast<uint32_t>(w);
    texData.height = static_cast<uint32_t>(h);
    texData.channels = 4;
    texData.pixels.assign(pixels, pixels + w * h * 4);
    stbi_image_free(pixels);
    return true;
  }

  // Try WebP decoder
  int webpWidth, webpHeight;
  if(WebPGetInfo(data, size, &webpWidth, &webpHeight))
  {
    pixels = WebPDecodeRGBA(data, size, &webpWidth, &webpHeight);
    if(pixels)
    {
      texData.width = static_cast<uint32_t>(webpWidth);
      texData.height = static_cast<uint32_t>(webpHeight);
      texData.channels = 4;
      texData.pixels.assign(pixels, pixels + webpWidth * webpHeight * 4);
      WebPFree(pixels);
      return true;
    }
  }

  return false;
}

bool GltfLoader::load(const std::filesystem::path& filepath)
{
  this->filename = filepath;

  LOGI("Loading GLTF: %s\n", filepath.string().c_str());

  fastgltf::Parser parser(fastgltf::Extensions::KHR_mesh_quantization |
                          fastgltf::Extensions::KHR_texture_transform |
                          fastgltf::Extensions::KHR_materials_unlit |
                          fastgltf::Extensions::KHR_materials_emissive_strength |
                          fastgltf::Extensions::KHR_lights_punctual |
                          fastgltf::Extensions::EXT_texture_webp);

  auto data = fastgltf::MappedGltfFile::FromPath(filepath);
  if(data.error() != fastgltf::Error::None)
  {
    LOGE("Failed to open GLTF file: %s\n", filepath.string().c_str());
    return false;
  }

  auto asset = parser.loadGltf(data.get(), filepath.parent_path(),
                               fastgltf::Options::LoadExternalBuffers | fastgltf::Options::LoadExternalImages);

  if(asset.error() != fastgltf::Error::None)
  {
    LOGE("Failed to parse GLTF: %s\n", fastgltf::getErrorMessage(asset.error()).data());
    return false;
  }

  LOGI("  Meshes: %zu, Materials: %zu, Images: %zu, Textures: %zu\n",
       asset->meshes.size(), asset->materials.size(), asset->images.size(), asset->textures.size());

  // Load images into texture data
  m_textureData.reserve(asset->images.size());
  for(size_t i = 0; i < asset->images.size(); ++i)
  {
    const auto& image = asset->images[i];
    TextureData texData;

    std::visit(fastgltf::visitor{
      [](auto& arg) {},
      [&](const fastgltf::sources::URI& uri) {
        std::filesystem::path imagePath = filepath.parent_path() / uri.uri.path();
        // First try to load as file with stbi
        int w, h, c;
        stbi_set_flip_vertically_on_load(false);
        uint8_t* data = stbi_load(imagePath.string().c_str(), &w, &h, &c, 4);
        if(data)
        {
          texData.width = static_cast<uint32_t>(w);
          texData.height = static_cast<uint32_t>(h);
          texData.channels = 4;
          texData.pixels.assign(data, data + w * h * 4);
          stbi_image_free(data);
          LOGI("  Loaded texture %zu from URI: %s (%dx%d)\n", i, imagePath.string().c_str(), w, h);
        }
        else
        {
          // Try loading file and decoding as WebP
          FILE* f = fopen(imagePath.string().c_str(), "rb");
          if(f)
          {
            fseek(f, 0, SEEK_END);
            size_t size = ftell(f);
            fseek(f, 0, SEEK_SET);
            std::vector<uint8_t> fileData(size);
            fread(fileData.data(), 1, size, f);
            fclose(f);
            if(decodeImageData(fileData.data(), size, texData))
            {
              LOGI("  Loaded WebP texture %zu from URI: %s (%dx%d)\n", i, imagePath.string().c_str(), texData.width, texData.height);
            }
            else
            {
              LOGW("  Failed to decode texture from URI: %s\n", imagePath.string().c_str());
            }
          }
          else
          {
            LOGW("  Failed to load texture from URI: %s\n", imagePath.string().c_str());
          }
        }
      },
      [&](const fastgltf::sources::Vector& vec) {
        if(decodeImageData(reinterpret_cast<const uint8_t*>(vec.bytes.data()), vec.bytes.size(), texData))
        {
          LOGI("  Loaded embedded texture %zu (%dx%d)\n", i, texData.width, texData.height);
        }
      },
      [&](const fastgltf::sources::BufferView& view) {
        const auto& bufferView = asset->bufferViews[view.bufferViewIndex];
        const auto& buffer = asset->buffers[bufferView.bufferIndex];
        std::visit(fastgltf::visitor{
          [](auto& arg) {},
          [&](const fastgltf::sources::Vector& vec) {
            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(vec.bytes.data()) + bufferView.byteOffset;
            if(decodeImageData(ptr, bufferView.byteLength, texData))
            {
              LOGI("  Loaded buffer texture %zu (%dx%d)\n", i, texData.width, texData.height);
            }
          },
          [&](const fastgltf::sources::Array& arr) {
            const uint8_t* ptr = reinterpret_cast<const uint8_t*>(arr.bytes.data()) + bufferView.byteOffset;
            if(decodeImageData(ptr, bufferView.byteLength, texData))
            {
              LOGI("  Loaded buffer texture %zu (%dx%d)\n", i, texData.width, texData.height);
            }
          }
        }, buffer.data);
      },
      [&](const fastgltf::sources::Array& arr) {
        if(decodeImageData(reinterpret_cast<const uint8_t*>(arr.bytes.data()), arr.bytes.size(), texData))
        {
          LOGI("  Loaded array texture %zu (%dx%d)\n", i, texData.width, texData.height);
        }
      }
    }, image.data);

    m_textureData.push_back(std::move(texData));
  }

  // Load materials
  m_materials.reserve(asset->materials.size());
  m_matNames.reserve(asset->materials.size());

  for(const auto& mat : asset->materials)
  {
    ObjMaterial m;
    m.ambient  = glm::vec4(0.1f, 0.1f, 0.1f, 0.0f);
    m.specular = glm::vec4(0.5f, 0.5f, 0.5f, 0.0f);
    m.transmittance = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
    m.dissolve  = 1.0f;
    m.ior       = 1.5f;
    m.shininess = 10.0f;
    m.illum     = 1;
    m.textureID = -1;

    const auto& pbr = mat.pbrData;
    m.diffuse = glm::vec4(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2], 0.0f);

    // Check for base color texture
    if(pbr.baseColorTexture.has_value())
    {
      size_t texIndex = pbr.baseColorTexture->textureIndex;
      if(texIndex < asset->textures.size())
      {
        const auto& tex = asset->textures[texIndex];
        // Check standard imageIndex first, then WebP extension
        if(tex.imageIndex.has_value())
        {
          m.textureID = static_cast<int>(*tex.imageIndex);
          LOGI("  Material '%s' uses texture %d\n", mat.name.c_str(), m.textureID);
        }
        else if(tex.webpImageIndex.has_value())
        {
          m.textureID = static_cast<int>(*tex.webpImageIndex);
          LOGI("  Material '%s' uses WebP texture %d\n", mat.name.c_str(), m.textureID);
        }
      }
    }

    m.emission = glm::vec4(mat.emissiveFactor.x(), mat.emissiveFactor.y(), mat.emissiveFactor.z(), 0.0f);

    m_materials.push_back(m);
    m_matNames.push_back(std::string(mat.name));
  }

  if(m_materials.empty())
  {
    m_materials.push_back(ObjMaterial());
    m_matNames.push_back("Default");
  }

  // Load meshes
  for(const auto& mesh : asset->meshes)
  {
    for(auto it = mesh.primitives.begin(); it != mesh.primitives.end(); ++it)
    {
      // Only handle triangle primitives
      if(it->type != fastgltf::PrimitiveType::Triangles)
      {
        LOGW("  Skipping non-triangle primitive (type %d)\n", static_cast<int>(it->type));
        continue;
      }

      int matId = it->materialIndex.has_value() ? static_cast<int>(*it->materialIndex) : 0;

      // Get position accessor
      auto* positionIt = it->findAttribute("POSITION");
      if(positionIt == it->attributes.end())
        continue;

      const auto& posAcc = asset->accessors[positionIt->accessorIndex];
      size_t vertexCount = posAcc.count;

      // Get normal accessor (optional)
      auto* normIt = it->findAttribute("NORMAL");

      size_t indexOffset = m_vertices.size();
      m_vertices.reserve(m_vertices.size() + vertexCount);

      // Pre-allocate vertices
      for(size_t i = 0; i < vertexCount; ++i)
      {
        m_vertices.push_back(ObjVertex{});
      }

      // Extract positions
      fastgltf::iterateAccessorWithIndex<glm::vec3>(asset.get(), posAcc,
        [&](glm::vec3 pos, size_t idx) {
          m_vertices[indexOffset + idx].pos = pos;
        });

      // Extract normals if available
      if(normIt != it->attributes.end())
      {
        const auto& normAcc = asset->accessors[normIt->accessorIndex];
        fastgltf::iterateAccessorWithIndex<glm::vec3>(asset.get(), normAcc,
          [&](glm::vec3 norm, size_t idx) {
            m_vertices[indexOffset + idx].nrm = norm;
          });
      }

      // Extract texture coordinates if available
      auto* texcoordIt = it->findAttribute("TEXCOORD_0");
      if(texcoordIt != it->attributes.end())
      {
        const auto& uvAcc = asset->accessors[texcoordIt->accessorIndex];
        fastgltf::iterateAccessorWithIndex<glm::vec2>(asset.get(), uvAcc,
          [&](glm::vec2 uv, size_t idx) {
            m_vertices[indexOffset + idx].texCoord = uv;
          });
      }

      // Extract indices
      if(it->indicesAccessor.has_value())
      {
        const auto& indAcc = asset->accessors[*it->indicesAccessor];
        size_t indexCount = indAcc.count;
        m_indices.reserve(m_indices.size() + indexCount);

        fastgltf::iterateAccessor<uint32_t>(asset.get(), indAcc,
          [&](uint32_t index) {
            m_indices.push_back(static_cast<uint32_t>(indexOffset + index));
          });

        // Material indices per triangle
        for(size_t i = 0; i < indexCount / 3; i++)
        {
          m_matIndices.push_back(matId);
        }
      }
    }
  }

  LOGI("  Loaded: %zu vertices, %zu indices\n", m_vertices.size(), m_indices.size());

  // Debug: print first few vertices
  if(!m_vertices.empty())
  {
    LOGI("  First vertex: pos=(%.3f,%.3f,%.3f) nrm=(%.3f,%.3f,%.3f) uv=(%.3f,%.3f)\n",
         m_vertices[0].pos.x, m_vertices[0].pos.y, m_vertices[0].pos.z,
         m_vertices[0].nrm.x, m_vertices[0].nrm.y, m_vertices[0].nrm.z,
         m_vertices[0].texCoord.x, m_vertices[0].texCoord.y);
    if(m_vertices.size() > 1000)
    {
      LOGI("  Vertex 1000: pos=(%.3f,%.3f,%.3f) nrm=(%.3f,%.3f,%.3f) uv=(%.3f,%.3f)\n",
           m_vertices[1000].pos.x, m_vertices[1000].pos.y, m_vertices[1000].pos.z,
           m_vertices[1000].nrm.x, m_vertices[1000].nrm.y, m_vertices[1000].nrm.z,
           m_vertices[1000].texCoord.x, m_vertices[1000].texCoord.y);
    }
  }

  return !m_vertices.empty();
}
