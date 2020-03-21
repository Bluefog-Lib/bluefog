#ifndef BLUEFOG_COMMON_CUDA_UTIL_H
#define BLUEFOG_COMMON_CUDA_UTIL_H

namespace bluefog {
namespace common {

class with_device {
public:
  with_device(int device);
  ~with_device();

private:
  int restore_device_ = CPU_DEVICE_ID;
};

}
}

#endif // BLUEFOG_COMMON_CUDA_UTIL_H