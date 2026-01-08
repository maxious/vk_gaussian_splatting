#include "gltf_loader.h"
#include <nvutils/logger.hpp>
#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/glm_element_traits.hpp>
#include <webp/decode.h>
#include <cstring>

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

  LOGI("  Meshes: %zu, Materials: %zu, Images: %zu\n",
       asset->meshes.size(), asset->materials.size(), asset->images.size());

  // Load materials
  m_materials.reserve(asset->materials.size());
  m_matNames.reserve(asset->materials.size());

  for(const auto& mat : asset->materials)
  {
    ObjMaterial m;
    m.ambient  = glm::vec3(0.1f);
    m.specular = glm::vec3(0.5f);
    m.transmittance = glm::vec3(0.0f);
    m.dissolve  = 1.0f;
    m.ior       = 1.5f;
    m.shininess = 10.0f;
    m.illum     = 1;

    const auto& pbr = mat.pbrData;
    m.diffuse = glm::vec3(pbr.baseColorFactor[0], pbr.baseColorFactor[1], pbr.baseColorFactor[2]);

    m.emission = glm::vec3(mat.emissiveFactor.x(), mat.emissiveFactor.y(), mat.emissiveFactor.z());

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

  return !m_vertices.empty();
}
