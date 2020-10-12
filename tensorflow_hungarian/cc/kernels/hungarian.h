// kernel_example.h
#ifndef KERNEL_HUNGARIAN_H_
#define KERNEL_HUNGARIAN_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct HungarianFunctor {
  void operator()(const OpKernelContext* context,
                  const Device& d,
                  int32 batch_size,
                  int32 size_n,
                  int32 size_m,
                  const T* in,
                  int32* out);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_HUNGARIAN_H_
