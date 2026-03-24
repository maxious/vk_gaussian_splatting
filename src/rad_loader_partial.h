/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef _RAD_LOADER_PARTIAL_H_
#define _RAD_LOADER_PARTIAL_H_

#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace vk_viewer {
namespace rad_partial {

// Get the maximum number of chunks to decode from environment variable.
// Returns 0 (unlimited) if not set or invalid.
inline uint32_t getMaxChunksFromEnv()
{
  const char* envVal = std::getenv("VK_GS_RAD_PARTIAL_MAX_CHUNKS");
  if(!envVal)
    return 0;

  char*    endptr = nullptr;
  long     val    = std::strtol(envVal, &endptr, 10);
  uint32_t result = (val > 0) ? static_cast<uint32_t>(val) : 0;
  return result;
}

}  // namespace rad_partial
}  // namespace vk_viewer

#endif
