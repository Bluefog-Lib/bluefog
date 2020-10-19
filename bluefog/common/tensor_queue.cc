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

#include "tensor_queue.h"

#include <assert.h>

namespace bluefog {
namespace common {

// Add a TensorTableEntry as well as its message to the queue.
Status TensorQueue::AddToTensorQueue(TensorTableEntry& e, Request& message) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (tensor_table_.find(e.tensor_name) != tensor_table_.end()) {
    return DUPLICATE_NAME_ERROR;
  }
  const std::string& name = message.tensor_name();
  tensor_table_.emplace(name, std::move(e));
  message_queue_.push(message);
  return Status::OK();
}

// Put callbacks for each tensor in the callback buffer and clear tensor queue
void TensorQueue::FinalizeTensorQueue(
    std::vector<StatusCallback>& callbacks_buffer) {
  std::lock_guard<std::mutex> guard(mutex_);
  for (auto& e : tensor_table_) {
    callbacks_buffer.emplace_back(e.second.callback);
  }
  tensor_table_.clear();
  while (!message_queue_.empty()) {
    message_queue_.pop();
  }
}

// Parse tensor names from response and generate a vector of corresponding
// tensor entries.
void TensorQueue::GetTensorEntriesFromResponse(
    const Response& response, std::vector<TensorTableEntry>& entries) {
  // Reserve to save re-allocation costs, as we know the size before.
  // entries may not be empty due to win_ops is processed at first.
  entries.reserve(entries.size() + response.tensor_names().size());
  {
    // Lock on the tensor table.
    std::lock_guard<std::mutex> guard(mutex_);
    for (auto& name : response.tensor_names()) {
      auto iter = tensor_table_.find(name);
      assert(iter != tensor_table_.end());

      assert(response.response_type() == Response::ALLREDUCE ||
             response.response_type() == Response::ALLGATHER ||
             response.response_type() == Response::BROADCAST ||
             response.response_type() == Response::NEIGHBOR_ALLGATHER ||
             response.response_type() == Response::NEIGHBOR_ALLREDUCE ||
             response.response_type() == Response::WIN_CREATE ||
             response.response_type() == Response::WIN_FREE ||
             response.response_type() == Response::ERROR);

      if (response.response_type() == Response::ERROR) {
        auto& e = iter->second;
        e.callback(Status::PreconditionError(response.error_message()));
      } else {
        entries.push_back(std::move(iter->second));
      }

      // Clear the tensor table of this tensor.
      tensor_table_.erase(iter);
    }
  }
}

// It should be used for no-coordinate request operator only.
TensorTableEntry TensorQueue::GetTensorEntriesFromRequestDirectly(
    const Request& request) {
  std::lock_guard<std::mutex> guard(mutex_);
  const std::string& name = request.tensor_name();

  auto iter = tensor_table_.find(name);
  assert(iter != tensor_table_.end());

  TensorTableEntry e = iter->second;
  // Clear the tensor table of this tensor.
  tensor_table_.erase(iter);
  return e;
}

// Get tensor entry given a tensor name
const TensorTableEntry& TensorQueue::GetTensorEntry(
    const std::string& tensor_name) const {
  // Lock on the tensor table.
  std::lock_guard<std::mutex> guard(mutex_);
  auto& iter = tensor_table_.at(tensor_name);

  return iter;
}

// Pop out all the messages from the queue
void TensorQueue::PopMessagesFromQueue(
    std::deque<Request>& message_queue_buffer) {
  std::lock_guard<std::mutex> guard(mutex_);
  while (!message_queue_.empty()) {
    Request message = message_queue_.front();
    message_queue_.pop();
    message_queue_buffer.push_back(std::move(message));
  }
}

// Push a message to massage queue
void TensorQueue::PushMessageToQueue(Request& message) {
  std::lock_guard<std::mutex> guard(mutex_);
  message_queue_.push(std::move(message));
}

Status FusionBufferManager::InitializeBuffer(
    int64_t threshold, int device, std::shared_ptr<OpContext> context,
    std::function<void()> on_start_init, std::function<void()> on_end_init) {
  auto& elem = tensor_fusion_buffers_[device];
  auto& buffer = elem.first;
  int64_t& size = elem.second;
  if (size != threshold) {
    buffer.reset();
    size = 0;
  }

  if (buffer == nullptr) {
    on_start_init();
    size = threshold;
  
    // Lazily allocate persistent buffer for Tensor Fusion and keep it
    // forever per device.
    Status status = context->AllocatePersistent(threshold, &buffer);
    on_end_init();

    return status;
  }

  return Status::OK();
}

std::shared_ptr<PersistentBuffer> FusionBufferManager::GetBuffer(int device) {
  return tensor_fusion_buffers_[device].first;
}

}  // namespace common
}  // namespace bluefog