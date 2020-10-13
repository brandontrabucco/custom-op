/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "time_two.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"
#include <algorithm>
#include <vector>
#include <cmath>

namespace tensorflow {

using shape_inference::Shape;
using shape_inference::Dimension;
using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <typename T>
static int augmenting_path(int nc,
                           std::vector<T>& cost,
                           std::vector<T>& u,
                           std::vector<T>& v,
                           std::vector<int>& path,
                           std::vector<int>& row4col,
                           std::vector<T>& shortestPathCosts,
                           int i,
                           std::vector<bool>& SR,
                           std::vector<bool>& SC,
                           T* p_minVal) {

    T infinity = (T) INFINITY;
    T minVal = 0;

    // Crouse's pseudocode uses set complements to keep track of remaining
    // nodes.  Here we use a vector, as it is more efficient in C++.
    int num_remaining = nc;
    std::vector<int> remaining(nc);
    for (int it = 0; it < nc; it++) {
        // Filling this up in reverse order ensures that the solution of a
        // constant cost matrix is the identity matrix (c.f. #11602).
        remaining[it] = nc - it - 1;
    }

    std::fill(SR.begin(), SR.end(), false);
    std::fill(SC.begin(), SC.end(), false);
    std::fill(shortestPathCosts.begin(), shortestPathCosts.end(), infinity);

    // find shortest augmenting path
    int sink = -1;
    while (sink == -1) {

        int index = -1;
        T lowest = infinity;
        SR[i] = true;

        for (int it = 0; it < num_remaining; it++) {
                int j = remaining[it];

            T r = minVal + cost[i * nc + j] - u[i] - v[j];
            if (r < shortestPathCosts[j]) {
                path[j] = i;
                shortestPathCosts[j] = r;
            }

            // When multiple nodes have the minimum cost, we select one which
            // gives us a new sink node. This is particularly important for
            // integer cost matrices with small co-efficients.
            if (shortestPathCosts[j] < lowest ||
                (shortestPathCosts[j] == lowest && row4col[j] == -1)) {
                lowest = shortestPathCosts[j];
                index = it;
            }
        }

        minVal = lowest;
        int j = remaining[index];
        if (minVal == infinity) { // infeasible cost matrix
            return -1;
        }

        if (row4col[j] == -1) {
            sink = j;
        } else {
            i = row4col[j];
        }

        SC[j] = true;
        remaining[index] = remaining[--num_remaining];
        remaining.resize(num_remaining);
    }

    *p_minVal = minVal;
    return sink;

}

template <typename T>
static int solve(int nr,
                 int nc,
                 const T* input_cost,
                 int* output_col4row) {

    // build a non-negative cost matrix
    std::vector<T> cost(nr * nc);
    T minval = *std::min_element(input_cost, input_cost + nr * nc);
    for (int i = 0; i < nr * nc; i++) {
        cost[i] = input_cost[i] - minval;
    }

    // initialize variables
    std::vector<T> u(nr, 0);
    std::vector<T> v(nc, 0);
    std::vector<T> shortestPathCosts(nc);
    std::vector<int> path(nc, -1);
    std::vector<int> col4row(nr, -1);
    std::vector<int> row4col(nc, -1);
    std::vector<bool> SR(nr);
    std::vector<bool> SC(nc);

    // iteratively build the solution
    for (int curRow = 0; curRow < nr; curRow++) {

        T minVal;
        int sink = augmenting_path<T>(
            nc, cost, u, v, path, row4col,
            shortestPathCosts, curRow, SR, SC, &minVal);
        if (sink < 0) {
            return -1;
        }

        // update dual variables
        u[curRow] += minVal;
        for (int i = 0; i < nr; i++) {
            if (SR[i] && i != curRow) {
                u[i] += minVal - shortestPathCosts[col4row[i]];
            }
        }

        for (int j = 0; j < nc; j++) {
            if (SC[j]) {
                v[j] -= minVal - shortestPathCosts[j];
            }
        }

        // augment previous solution
        int j = sink;
        while (1) {
            int i = path[j];
            row4col[j] = i;
            std::swap(col4row[i], j);
            if (i == curRow) {
                break;
            }
        }
    }

    for (int i = 0; i < nr; i++) {
        output_col4row[i] = col4row[i];
    }

    return 0;

}

// CPU specialization of actual computation.
template <typename T>
struct TimeTwoFunctor<CPUDevice, T> {

    void operator()(const CPUDevice& d,
                    int size_n,
                    int size_m,
                    const T* costs,
                    int* assignments) {

        solve<T>(size_n, size_m, costs, assignments);

    }

};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class TimeTwoOp : public OpKernel {

public:

    explicit TimeTwoOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {

        // Grab the input tensor
        const Tensor& costs_tensor = context->input(0);

        // Create an output tensor that is null
        Tensor* assignments_tensor = NULL;

        // determine the shape of the output
        std::vector<int64> shape;
        for (int i = 0; i < costs_tensor.shape().dims(); ++i) {
            shape.push_back(costs_tensor.shape().dim_size(i));
        }

        // allocate space for the output tensor
        OP_REQUIRES_OK(context, context->allocate_output(
            0, TensorShape({shape[0], shape[1]}), &assignments_tensor));

        // check if the operation is too large
        OP_REQUIRES(context, costs_tensor.NumElements() <= tensorflow::kint32max,
                    errors::InvalidArgument("Too many elements in tensor"));

        // prepare shared variables for each shard
        int batch_size = static_cast<int>(shape[0]);
        int size_n = static_cast<int>(shape[1]);
        int size_m = static_cast<int>(shape[2]);
        const T* costs = costs_tensor.flat<T>().data();
        int* assignments = assignments_tensor->flat<int>().data();

        // implementation of the hungarian algorithm in c++
        auto sharded_function = [
                &context,
                &size_n,
                &size_m,
                &costs,
                &assignments](int start, int limit) {

            // batch-wise sharded function
            for (int i = start; i < limit; i++) {

                // launch the device generalized operation functor
                TimeTwoFunctor<Device, T>()(
                    context->eigen_device<Device>(),
                    size_n,
                    size_m, // below we view into costs and assignments
                    costs + i * size_n * size_m,
                    assignments + i * size_n);

            }

        };

        // get a handle to the cpu device threads
        auto pool = context->device()->tensorflow_cpu_worker_threads();

        // computational cost of generating a single kernel operation
        const int64 op_cost = 10000 * size_n * size_n * size_m;

        // launch a sharded batch function
        Shard(pool->num_threads, pool->workers,
              batch_size, op_cost, sharded_function);

    }

};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TimeTwo").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      TimeTwoOp<CPUDevice, T>);
REGISTER_CPU(float);
REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  extern template struct TimeTwoFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("TimeTwo").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      TimeTwoOp<GPUDevice, T>);
REGISTER_GPU(float);
REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow
