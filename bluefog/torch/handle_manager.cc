#include "handle_manager.h"

namespace bluefog {
namespace torch {

int HandleManager::AllocateHandle() {
  int handle = last_handle_.fetch_add(1) + 1;
  std::lock_guard<std::mutex> guard(mutex_);
  results_[handle] = nullptr;
  return handle;
}

void HandleManager::MarkDone(int handle, const Status& status) {
  std::lock_guard<std::mutex> guard(mutex_);
  results_[handle] = std::make_shared<Status>(status);
}

bool HandleManager::PollHandle(int handle) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (results_.find(handle) == results_.end()) {
    throw std::invalid_argument("Handle " + std::to_string(handle) +
                                " was not created or has been cleared.");
  }
  return results_[handle] != nullptr;
}

std::shared_ptr<Status> HandleManager::ReleaseHandle(int handle) {
  std::lock_guard<std::mutex> guard(mutex_);
  if (results_.find(handle) == results_.end()) {
    throw std::invalid_argument("Handle " + std::to_string(handle) +
                                " was not created or has been cleared.");
  }
  auto status = results_[handle];
  results_.erase(handle);
  return status;
}

}  // namespace torch
}  // namespace bluefog