// kernel_example.h
#ifndef KERNEL_HUNGARIAN_H_
#define KERNEL_HUNGARIAN_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct HungarianFunctor {
  void operator()(const OpKernelContext* context,
                  const Device& d,
                  int batch_size,
                  int size_n,
                  int size_m,
                  const T* in,
                  T* out);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_HUNGARIAN_H_
