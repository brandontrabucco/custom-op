// kernel_example.h
#ifndef KERNEL_HUNGARIAN_H_
#define KERNEL_HUNGARIAN_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct HungarianFunctor {
    void operator()(const Device& d,
                    int size_n,
                    int size_m,
                    const T* costs,
                    int* assignments);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_HUNGARIAN_H_
