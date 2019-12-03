#ifndef BLUEFOG_COMMON_TENSOR_QUEUE_H
#define BLUEFOG_COMMON_TENSOR_QUEUE_H

#include <iostream>
#include <mutex>
#include <queue>

#include "common.h"

namespace bluefog {
namespace common {

class TensorQueue {
 public:
  TensorQueue() = default;
  TensorQueue(const TensorQueue&) = delete;
  Status AddToTensorQueue(TensorTableEntry& e);

  void FinalizeTensorQueue(std::vector<StatusCallback>& callbacks_buffer);

  TensorTableEntry PopMessagesFromQueue();

  void PushMessageToQueue(TensorTableEntry& message);

 protected:
  // Queue of MPI requests waiting to be performed.
  std::queue<TensorTableEntry> message_queue_;

  // A mutex that needs to be used whenever operations on message queue are
  // done.
  mutable std::mutex mutex_;
};

}  // namespace common
}  // namespace bluefog

#endif  // BLUEFOG_COMMON_TENSOR_QUEUE_H
