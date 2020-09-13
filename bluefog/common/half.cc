// Copyright 2018 Uber Technologies, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "half.h"

namespace bluefog {
namespace common {

// float16 custom data type summation operation.
void float16_sum(void* invec, void* inoutvec, int* len,
                 MPI_Datatype* datatype) {
  // cast invec and inoutvec to your float16 type
  auto* in = (unsigned short*)invec;
  auto* inout = (unsigned short*)inoutvec;

  int i = 0;
  for (; i < *len; ++i) {
    float in_float;
    float inout_float;
    HalfBits2Float(in + i, &in_float);
    HalfBits2Float(inout + i, &inout_float);
    inout_float += in_float;
    Float2HalfBits(&inout_float, inout + i);
  }
}

} // namespace common
} // namespace horovod
