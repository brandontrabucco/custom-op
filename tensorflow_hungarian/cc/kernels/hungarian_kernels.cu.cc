/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "hungarian.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include <cmath>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/swap.h>
#include <thrust/replace.h>
#include <thrust/functional.h>

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

__global__ void kernel_3(const int nc,
                         int* remaining) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < nc; i += blockDim.x * gridDim.x) {

        remaining[i] = nc - i - 1;

    }

}

template <typename T>
__global__ void kernel_4(const int num_remaining,
                         const int nc,
                         const int i,
                         const T minVal,
                         const int* remaining,
                         const int* row4col,
                         const T* cost,
                         const T* u,
                         const T* v,
                         int* path,
                         int* index,
                         T* shortestPathCosts,
                         T* lowest) {

    for (int it = blockIdx.x * blockDim.x + threadIdx.x;
            it < num_remaining; it += blockDim.x * gridDim.x) {

        int j = remaining[it];
        T r = minVal + cost[i * nc + j] - u[i] - v[j];

        if (r < shortestPathCosts[j]) {
            path[j] = i;
            shortestPathCosts[j] = r;
        }

        // When multiple nodes have the minimum cost, we select one which
        // gives us a new sink node. This is particularly important for
        // integer cost matrices with small co-efficients.
        if (shortestPathCosts[j] < *lowest ||
            (shortestPathCosts[j] == *lowest && row4col[j] == -1)) {

            atomicExch(lowest, shortestPathCosts[j]);
            atomicExch(index, it);

        }

    }

}

template <typename T>
static int augmenting_path(const GPUDevice& d,
                           int nc,
                           thrust::device_vector<T>& cost,
                           T* cost_raw_pointer,
                           thrust::device_vector<T>& u,
                           T* u_raw_pointer,
                           thrust::device_vector<T>& v,
                           T* v_raw_pointer,
                           thrust::device_vector<int>& path,
                           int* path_raw_pointer,
                           thrust::device_vector<int>& row4col,
                           int* row4col_raw_pointer,
                           thrust::device_vector<T>& shortestPathCosts,
                           T* shortestPathCosts_raw_pointer,
                           int i,
                           thrust::device_vector<bool>& SR,
                           bool* SR_raw_pointer,
                           thrust::device_vector<bool>& SC,
                           bool* SC_raw_pointer,
                           T* p_minVal) {

    T infinity = (T) INFINITY;
    T minVal = 0;

    // Crouse's pseudocode uses set complements to keep track of remaining
    // nodes.  Here we use a vector, as it is more efficient in C++.
    int num_remaining = nc;
    thrust::device_vector<int> remaining(nc);
    int* remaining_raw_pointer = thrust::raw_pointer_cast(&remaining[0]);

    // Filling this up in reverse order ensures that the solution of a
    // constant cost matrix is the identity matrix (c.f. #11602).
    kernel_3<<<1024, 20, 0, d.stream()>>>(nc, remaining_raw_pointer);

    // synchronize the device
    cudaDeviceSynchronize();

    thrust::fill(SR.begin(), SR.end(), false);
    thrust::fill(SC.begin(), SC.end(), false);
    thrust::fill(shortestPathCosts.begin(), shortestPathCosts.end(), infinity);

    // find shortest augmenting path
    int sink = -1;
    while (sink == -1) {

        int index = -1;
        T lowest = infinity;
        SR[i] = true;

        int* index_raw_pointer;
        cudaMalloc(&index_raw_pointer, sizeof(int));

        T* lowest_raw_pointer;
        cudaMalloc(&lowest_raw_pointer, sizeof(T));

        cudaMemcpy(index_raw_pointer, &index, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(lowest_raw_pointer, &lowest, sizeof(T), cudaMemcpyHostToDevice);

        kernel_4<T><<<1024, 20, 0, d.stream()>>>(
            num_remaining,
            nc,
            i,
            minVal,
            remaining_raw_pointer,
            row4col_raw_pointer,
            cost_raw_pointer,
            u_raw_pointer,
            v_raw_pointer,
            path_raw_pointer,
            index_raw_pointer,
            shortestPathCosts_raw_pointer,
            lowest_raw_pointer);

        // synchronize the device
        cudaDeviceSynchronize();

        cudaMemcpy(&index, index_raw_pointer, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&lowest, lowest_raw_pointer, sizeof(T), cudaMemcpyDeviceToHost);

        cudaFree(index_raw_pointer);
        cudaFree(lowest_raw_pointer);

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
__global__ void kernel_0(const int nr,
                         const int nc,
                         const T* minVal,
                         T* cost) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < nr * nc; i += blockDim.x * gridDim.x) {

        cost[i] = cost[i] - *minVal;

    }

}

template <typename T>
__global__ void kernel_1(const int nr,
                         const int curRow,
                         const T minVal,
                         const T* shortestPathCosts,
                         const int* col4row,
                         const bool* SR,
                         T* u) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
            i < nr; i += blockDim.x * gridDim.x) {

        if (SR[i] && i != curRow) {
            u[i] += minVal - shortestPathCosts[col4row[i]];
        }

        else if (i == curRow) {
            u[curRow] += minVal;
        }

    }

}

template <typename T>
__global__ void kernel_2(const int nc,
                         const T minVal,
                         const T* shortestPathCosts,
                         const bool* SC,
                         T* v) {

    for (int j = blockIdx.x * blockDim.x + threadIdx.x;
            j < nc; j += blockDim.x * gridDim.x) {

            if (SC[j]) {
                v[j] -= minVal - shortestPathCosts[j];
            }

    }

}

template <typename T>
static int solve(const GPUDevice& d,
                 int nr,
                 int nc,
                 const T* input_cost_raw_pointer,
                 int* output_col4row) {

    // allocate a new cost matrix
    thrust::device_vector<T> cost(nr * nc);
    T* cost_raw_pointer = thrust::raw_pointer_cast(&cost[0]);
    cudaMemcpy(cost_raw_pointer, input_cost_raw_pointer,
               nr * nc * sizeof(T), cudaMemcpyDeviceToDevice);

    // find a pointer to the minimum element of the cost
    T* min_val_raw_pointer = thrust::raw_pointer_cast(
        &thrust::min_element(cost.begin(), cost.begin() + nr * nc)[0]);

    // subtract minimum element
    kernel_0<T><<<1024, 20, 0, d.stream()>>>(
        nr, nc, min_val_raw_pointer, cost_raw_pointer);

    // synchronize the device
    cudaDeviceSynchronize();

    // initialize variables
    thrust::device_vector<T> u(nr, 0);
    T* u_raw_pointer = thrust::raw_pointer_cast(&u[0]);

    // initialize variables
    thrust::device_vector<T> v(nc, 0);
    T* v_raw_pointer = thrust::raw_pointer_cast(&v[0]);

    // initialize variables
    thrust::device_vector<T> shortestPathCosts(nc);
    T* shortestPathCosts_raw_pointer = thrust::raw_pointer_cast(&shortestPathCosts[0]);

    // initialize variables
    thrust::device_vector<int> path(nc, -1);
    int* path_raw_pointer = thrust::raw_pointer_cast(&path[0]);

    // initialize variables
    thrust::device_vector<int> col4row(nr, -1);
    int* col4row_raw_pointer = thrust::raw_pointer_cast(&col4row[0]);

    // initialize variables
    thrust::device_vector<int> row4col(nc, -1);
    int* row4col_raw_pointer = thrust::raw_pointer_cast(&row4col[0]);

    // initialize variables
    thrust::device_vector<bool> SR(nr);
    bool* SR_raw_pointer = thrust::raw_pointer_cast(&SR[0]);

    // initialize variables
    thrust::device_vector<bool> SC(nc);
    bool* SC_raw_pointer = thrust::raw_pointer_cast(&SC[0]);

    // iteratively build the solution
    for (int curRow = 0; curRow < nr; curRow++) {

        T minVal;
        int sink = augmenting_path<T>(
            d,
            nc,
            cost,
            cost_raw_pointer,
            u,
            u_raw_pointer,
            v,
            v_raw_pointer,
            path,
            path_raw_pointer,
            row4col,
            row4col_raw_pointer,
            shortestPathCosts,
            shortestPathCosts_raw_pointer,
            curRow,
            SR,
            SR_raw_pointer,
            SC,
            SC_raw_pointer,
            &minVal);

        if (sink < 0) {
            return -1;
        }

        // subtract minimum element
        kernel_1<T><<<1024, 20, 0, d.stream()>>>(
            nr,
            curRow,
            minVal,
            shortestPathCosts_raw_pointer,
            col4row_raw_pointer,
            SR_raw_pointer,
            u_raw_pointer);

        // subtract minimum element
        kernel_2<T><<<1024, 20, 0, d.stream()>>>(
            nc,
            minVal,
            shortestPathCosts_raw_pointer,
            SC_raw_pointer,
            v_raw_pointer);

        // wait until the cuda kernel finishes
        cudaDeviceSynchronize();

        // augment previous solution
        int j = sink;
        while (1) {

            int i = path[j];
            row4col[j] = i;

            int k = j;
            j = col4row[i];
            col4row[i] = k;

            if (i == curRow) {
                break;
            }

        }

    }

    cudaMemcpy(output_col4row,
               col4row_raw_pointer,
               sizeof(int) * nr,
               cudaMemcpyDeviceToDevice);

    return 0;

}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct HungarianFunctor<GPUDevice, T> {

    void operator()(const GPUDevice& d,
                    int size_n,
                    int size_m,
                    const T* costs,
                    int* assignments) {

        solve<T>(d, size_n, size_m, costs, assignments);

    }

};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct HungarianFunctor<GPUDevice, float>;
template struct HungarianFunctor<GPUDevice, int32>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
