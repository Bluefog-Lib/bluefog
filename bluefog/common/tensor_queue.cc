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

#include "logging.h"
#include "tensor_queue.h"

#include <assert.h>

namespace bluefog {
namespace common {

// Add a TensorTableEntry as well as its message to the queue.
Status TensorQueue::AddToTensorQueue(TensorTableEntry& e) {
  std::lock_guard<std::mutex> guard(mutex_);
  message_queue_.push(std::move(e));
  return Status::OK();
}

// Put callbacks for each tensor in the callback buffer and clear tensor queue
void TensorQueue::FinalizeTensorQueue(
    std::vector<StatusCallback>& callbacks_buffer) {
  std::lock_guard<std::mutex> guard(mutex_);
  while (!message_queue_.empty()) {
    TensorTableEntry message = message_queue_.front();
    BFLOG(TRACE) << "Message " << message.tensor_name << " is still in the queue after shut down.";
    message_queue_.pop();
    callbacks_buffer.emplace_back(message.callback);
  }
}

// Pop out the front messages from the queue
TensorTableEntry TensorQueue::PopMessagesFromQueue() {
  std::lock_guard<std::mutex> guard(mutex_);
  if (!message_queue_.empty()) {
    TensorTableEntry message = message_queue_.front();
    message_queue_.pop();
    return message;
  }
  throw std::length_error("Tensor Queue is empty. Cannot pop meesage.");
}


}  // namespace common
}  // namespace bluefog