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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "cublas_v2.h"
#include <cmath>
#include <limits>

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void resize(const int32 size_n,
                       const int32 size_m,
                       const int32 size,
                       int32* device_max,
                       const T* costs,
                       T* square_costs) {

    // calculate the maximum number of thread calls to make
    int32 n = size * size;

    // run each thread at least once
    for (int32 i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n; i += blockDim.x * gridDim.x) {

        // calculate the position in the destination tensor
        int32 col = i % size;
        int32 row = (i / size) % size;

        // if the position represents a padded cell
        if ((row >= size_n || col >= size_m) && NULL != device_max) {

            // copy the max value of that row
            square_costs[i] = costs[device_max[0]];

        }

        // if the position represents a cell to copy
        else {

            // calculate the position in the source tensor and copy
            square_costs[i] = costs[col + row * size_m];

        }

    }

}

template <typename T>
__global__ void replace_infinities(const int32 size,
                                   int32* device_max,
                                   T infinity,
                                   const T* costs,
                                   T* square_costs) {

    // this case only occurs when all values are infinite.
    if (max == infinity) {
        max = 0;
    }

    // a value higher than the maximum value present in the matrix.
    else {
        max++;
    }

    // calculate the maximum number of thread calls to make
    int32 n = size * size;

    // run each thread at least once
    for (int32 i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n; i += blockDim.x * gridDim.x) {

        // replace the matrix entries that are infinity
        if (infinity == square_costs[i]) {
            square_costs[i] = costs[device_max[0]];
        }

    }

}

template <typename T>
__global__ void minimize_along_direction(const int32 size,
                                         int32* device_min,
                                         bool cols_fixed,
                                         T* square_costs) {

    // calculate the maximum number of thread calls to make
    int32 n = size * size;

    // run each thread at least once
    for (int32 i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n; i += blockDim.x * gridDim.x) {

        // calculate the position in the destination tensor
        int32 col = i % size;
        int32 row = (i / size) % size;

        // determine how to index into pos: are rows or columns fixed
        int32 pos = cols_fixed ? row : col;

        // replace the matrix entries that are infinity
        if (0 < device_min[pos]) {
            square_costs[i] -= square_costs[device_min[pos]];
        }

    }

}

static constexpr int NORMAL = 0;
static constexpr int STAR   = 1;
static constexpr int PRIME  = 2;

template <typename T>
__global__ void star_mask(const int32 size,
                          int32* masks,
                          int32* masks_buffer,
                          T* square_costs) {

    // calculate the maximum number of thread calls to make
    int32 n = size * size;

    // run each thread at least once
    for (int32 i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n; i += blockDim.x * gridDim.x) {

        // calculate the position in the destination tensor
        int32 col = i % size;
        int32 row = (i / size) % size;

        // value to assign to this location in masks
        masks_buffer[i] = (square_costs[i] == 0) ? STAR : masks[i];

        // wait until all threads have gotten this far
        __syncthreads();

        // look at the preceding rows and check if STAR is present
        for (int32 j = 0; j < row; j++) {

            // calculate the position of that element
            int32 pos = col + (j) * size;

            // if there is a STAR then deactivate this cell
            if (masks_buffer[pos] == STAR) masks_buffer[i] = masks[i];

            // if there is a STAR then deactivate this cell
            if (masks[pos] == STAR) masks_buffer[i] = masks[i];

        }

        // wait until all threads have gotten this far
        __syncthreads();

        // look at the preceding cols and check if STAR is present
        for (int32 j = 0; j < col; j++) {

            // calculate the position of that element
            int32 pos = (j) + row * size;

            // if there is a STAR then deactivate this cell
            if (masks_buffer[pos] == STAR) masks_buffer[i] = masks[i];

        }

        // we are in the clear and we can assign the mask value
        masks[i] = masks_buffer[i];

    }

}

void step1(int32* states,
           const int32 size,
           int32* masks,
           bool* row_masks,
           bool* col_masks,
           T* square_costs) {

    // allocate memory for a square costs matrix
    int32* masks_buffer;
    cudaMalloc((void**)&masks_buffer, sizeof(int32) * size * size);

    // fill the resized matrix with the original matrix values
    star_mask<T><<<32, 256>>>(size,
                              masks,
                              masks_buffer,
                              square_costs);

    // allow for every parallel operation on the GPU to finish
    cudaDeviceSynchronize();

    // remove the memory allocated for calculating the buffer
    cudaFree(masks_buffer);

    // determine which states to move to if not finished
    states[0] = (states[0] == 0) ? 0 : 2;

}

template <typename T>
__global__ void count_mask(int32 size,
                           int32* masks,
                           bool* col_masks,
                           int32* counts) {

    // calculate the maximum number of thread calls to make
    int32 n = size * size;

    // run each thread at least once
    for (int32 i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n; i += blockDim.x * gridDim.x) {

        // calculate the position in the destination tensor
        int32 col = i % size;

        // active when an entry in masks is STAR
        if (masks[i] == STAR) {

            // increment the cover count variable
            atomicAdd(counts, 1);

            // and set the col mask to true at this col
            col_masks[col] = true

        }

    }

}

void step2(int32* states,
           const int32 size,
           int32* masks,
           bool* row_masks,
           bool* col_masks,
           T* square_costs) {

    // allocate memory for a square costs matrix
    int32* counts;
    cudaMalloc((void**)&counts, sizeof(int32));

    // fill the resized matrix with the original matrix values
    count_mask<T><<<32, 256>>>(size,
                               masks,
                               col_masks,
                               counts);

    // allow for every parallel operation on the GPU to finish
    cudaDeviceSynchronize();

    // malloc space on the heap for the cpu cover count
    int32* counts_cpu = (int32*) malloc(sizeof(int32));

    // copy the cover count to cpu to control program flow
    cudaMemcpy(counts_cpu,
               counts,
               sizeof(int32),
               cudaMemcpyDeviceToHost);

    // remove the memory allocated for calculating the buffer
    cudaFree(count);

    // determine which states to move to if not finished
    states[0] = (states[0] == 0) ? 0 : (counts_cpu[0] >= size) ? 0 : 3;

    // remove the memory allocated for counts
    free(counts_cpu);

}

void step3(int32* states,
           const int32 size,
           int32* masks,
           bool* row_masks,
           bool* col_masks,
           T* square_costs) {

    // pass

}

void step4(int32* states,
           const int32 size,
           int32* masks,
           bool* row_masks,
           bool* col_masks,
           T* square_costs) {

    // pass

}

void step5(int32* states,
           int32 size,
           int32* masks,
           bool* row_masks,
           bool* col_masks,
           T* square_costs) {

    // pass

}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct HungarianFunctor<GPUDevice, T> {

    void operator()(const GPUDevice& d,
                    const int32 size_n,
                    const int32 size_m,
                    const T* costs,
                    int32* assignments) {

        // choose the dimension that is largest
        const int32 size = std::max(size_n, size_m);

        // initialize a library resource handle
        cublasHandle_t handle;
        cublasCreate(&handle);

        /*
         *
         *
         * CONVERT THE MATRIX INTO A SQUARE FORMAT
         *
         *
         */

        // create a device variable to store the batch-wise max
        int32* device_max;

        // allocate space for the batch-wise max on the GPU
        cudaMalloc((void**)&device_max, sizeof(int32));

        // the size of each batch-wise matrix in costs
        int32 n = size_n * size_m;

        // launch several parallel max operations
        cublasIsamax(handle, n, costs, 1, device_max + i);

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        // allocate memory for a square costs matrix
        T* square_costs;
        cudaMalloc((void**)&square_costs, sizeof(T) * size * size);

        // fill the resized matrix with the original matrix values
        resize<T><<<32, 256>>>(size_n,
                               size_m,
                               size,
                               device_max,
                               costs,
                               square_costs);

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        /*
         *
         *
         * REPLACE INFINITE ENTRIES IN THE MATRIX
         *
         *
         */

        // calculate the value of infinity
        const T infinity = std::numeric_limits<T>::infinity();

        // replace all infinities with the max value in the matrix
        replace_infinities<T><<<32, 256>>>(size,
                                           device_max,
                                           costs,
                                           infinity,
                                           square_costs);

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        // remove the memory allocated for calculating the max
        cudaFree(device_max);

        /*
         *
         *
         * NORMALIZE THE MATRIX ALONG ROWS AND COLUMNS
         *
         *
         */

        // create a device variable to store the batch-wise max
        int32* device_min;

        // allocate space for the batch-wise max on the GPU
        cudaMalloc((void**)&device_min, sizeof(int32) * size);

        bool cols_fixed = size_n >= size_m;

        // launch several parallel min operations
        for (int32 j = 0; j < size; j++) {

            // stride iterates over rows when cols_fixed is true
            // and over cols when cols_fixed is false
            cublasIsamin(
                handle,
                size,
                square_costs + (cols_fixed ? 1 : size) * j,
                cols_fixed ? size : 1, // how many cells apart is the next element
                device_min + i * size + j);

        }

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        // replace all infinities with the max value in the matrix
        minimize_along_direction<T><<<32, 256>>>(size,
                                                 device_min,
                                                 cols_fixed,
                                                 square_costs);

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        // flip the variable and normalize in the opposite direction
        cols_fixed = !cols_fixed;

        // launch several parallel min operations
        for (int32 j = 0; j < size; j++) {

            // stride iterates over rows when cols_fixed is true
            // and over cols when cols_fixed is false
            cublasIsamin(
                handle,
                size,
                square_costs + (cols_fixed ? 1 : size) * j,
                cols_fixed ? size : 1, // how many cells apart is the next element
                device_min + i * size + j);

        }

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        // replace all infinities with the max value in the matrix
        minimize_along_direction<T><<<32, 256>>>(size,
                                                 device_min,
                                                 cols_fixed,
                                                 square_costs);

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        // remove the memory allocated for calculating the min
        cudaFree(device_min);

        /*
         *
         *
         * ENTER THE PRIMARY ALGORITHM LOOP
         *
         *
         */

        // allocate space for several mask tensors
        int32* masks;
        cudaMalloc((void**)&masks, sizeof(int32) * size * size);
        cudaMemset(masks, 0x0, sizeof(int32) * size * size);

        // allocate space for several mask tensors and set to zero
        bool* row_masks;
        cudaMalloc((void**)&row_masks, sizeof(bool) * size);
        cudaMemset(row_masks, 0x0, sizeof(bool) * size);

        // allocate space for several mask tensors and set to zero
        bool* col_masks;
        cudaMalloc((void**)&col_masks, sizeof(bool) * size);
        cudaMemset(col_masks, 0x0, sizeof(bool) * size);

        // allocate space for a loop state variable
        int32* states = (int32*) malloc(sizeof(int32));
        memset(states, 0x1, sizeof(int32));

        /*
         *
         *
         * UPDATE THE SOLUTION IN A LOOP
         *
         *
         */

        do {

            // update the solution

        } while (states[i] != 0);

        // free the state variable once all batch elements are done
        free(states);

        // remove the memory allocated for calculating the masks
        cudaFree(masks);

        // remove the memory allocated for calculating the masks
        cudaFree(row_masks);

        // remove the memory allocated for calculating the masks
        cudaFree(col_masks);

        // we are done to remove the library handle
        cublasDestroy(handle);

    }

};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct HungarianFunctor<GPUDevice, float>;
template struct HungarianFunctor<GPUDevice, int32>;
}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
