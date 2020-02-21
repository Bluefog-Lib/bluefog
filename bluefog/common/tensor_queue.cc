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