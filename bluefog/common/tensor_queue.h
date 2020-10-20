// Modifications copyright (C) 2020 Bluefog Team. All Rights Reserved.
// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
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
// ==============================================================================

#ifndef BLUEFOG_COMMON_TENSOR_QUEUE_H
#define BLUEFOG_COMMON_TENSOR_QUEUE_H

#include <iostream>
#include <mutex>
#include <queue>

#include "common.h"
#include "message.h"

namespace bluefog {
namespace common {

class TensorQueue {
 public:
  TensorQueue() = default;
  TensorQueue(const TensorQueue&) = delete;
  Status AddToTensorQueue(TensorTableEntry& e, Request& message);

  void FinalizeTensorQueue(std::vector<StatusCallback>& callbacks_buffer);

  void GetTensorEntriesFromResponse(const Response& response,
                                    std::vector<TensorTableEntry>& entries);

  TensorTableEntry GetTensorEntriesFromRequestDirectly(const Request& request);

  const TensorTableEntry& GetTensorEntry(const std::string& tensor_name) const;

  void PopMessagesFromQueue(std::deque<Request>& message_queue_buffer);

  void PushMessageToQueue(Request& message);

  // Used when setting Topology, which require the tensor queue should be empty always.
  inline void LockTensorQueue() { mutex_.lock(); }
  inline void UnlockTensorQueue() { mutex_.unlock(); }
  inline size_t size() { return message_queue_.size(); }

 protected:
  // Tensors waiting to be processed.
  // Key is based upon the message name since tensor_name in table entry for win ops
  // is for window and we need to add "win_put."/"win_create." before it in message.
  std::unordered_map<std::string, TensorTableEntry> tensor_table_;

  // Queue of MPI requests waiting to be sent to the coordinator node.
  std::queue<Request> message_queue_;

  // A mutex that needs to be used whenever operations on message queue are
  // done.
  mutable std::mutex mutex_;
};

// Encapsulates the process of creating and destroying fusion buffers as the requested
// threshold is changed.
class FusionBufferManager {
 public:
  // Initializes a buffer of the given threshold size if not already cached.
  //
  // Args:
  //  threshold: Size of the buffer in bytes.
  //  device: Device ID to associate the buffer.
  //  context: Framework used to create the buffer and associate it.
  //  on_start_init: Callback on starting buffer initialization.
  //  on_end_init: Callback on completing buffer initialization.
  Status InitializeBuffer(int64_t threshold, int device,
                          std::shared_ptr<OpContext> context,
                          std::function<void()> on_start_init,
                          std::function<void()> on_end_init);

  // Returns the buffer associated with the given device and framework, or null.
  std::shared_ptr<PersistentBuffer> GetBuffer(int device);

 private:
  // Memory buffers for Tensor Fusion.  They are keyed by device ID.
  std::unordered_map<int, std::pair<std::shared_ptr<PersistentBuffer>, int64_t>>
      tensor_fusion_buffers_;
};

}  // namespace common
}  // namespace bluefog

#endif  // BLUEFOG_COMMON_TENSOR_QUEUE_H
