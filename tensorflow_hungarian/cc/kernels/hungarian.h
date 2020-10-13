// kernel_example.h
#ifndef KERNEL_HUNGARIAN_H_
#define KERNEL_HUNGARIAN_H_

#include "tensorflow/core/framework/op.h"

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct HungarianFunctor {
  void operator()(const Device& d,
                  const int32 size_n,
                  const int32 size_m,
                  const T* costs,
                  int32* assignments);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_HUNGARIAN_H_
