#ifndef BLUEFOG_TORCH_HANDLE_MANAGER_H
#define BLUEFOG_TORCH_HANDLE_MANAGER_H

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "../common/common.h"

namespace bluefog {
namespace torch {

class HandleManager {
 public:
  int AllocateHandle();
  void MarkDone(int handle, const common::Status& status);
  bool PollHandle(int handle);
  std::shared_ptr<common::Status> ReleaseHandle(int handle);

 private:
  std::atomic_int last_handle_;
  std::unordered_map<int, std::shared_ptr<common::Status>> results_;
  std::mutex mutex_;
};

}  // namespace torch
}  // namespace bluefog

#endif  // BLUEFOG_TORCH_HANDLE_MANAGER_H
