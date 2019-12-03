#ifndef BLUEFOG_TORCH_CUDA_UTIL_H
#define BLUEFOG_TORCH_CUDA_UTIL_H

namespace bluefog {
namespace torch {

class with_device {
public:
  with_device(int device);
  ~with_device();

private:
  int restore_device_ = CPU_DEVICE_ID;
};

}
}

#endif // BLUEFOG_TORCH_CUDA_UTIL_H