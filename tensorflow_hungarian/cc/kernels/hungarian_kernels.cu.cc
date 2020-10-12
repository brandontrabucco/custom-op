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
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "cublas_v2.h"
#include <cmath>
#include <limits>

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void resize(int32 batch_size,
                       int32 size_n,
                       int32 size_m,
                       int32 size,
                       int32* device_max,
                       const T* costs,
                       T* square_costs) {

    // calculate the maximum number of thread calls to make
    int32 n = batch_size * size * size;

    // run each thread at least once
    for (int32 i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n; i += blockDim.x * gridDim.x) {

        // calculate the position in the destination tensor
        int32 col = i % size;
        int32 row = (i / size) % size;
        int32 batch = i / (size * size);

        // if the position represents a padded cell
        if (batch < batch_size
                && (row >= size_n || col >= size_m)
                && NULL != device_max) {

            // copy the max value of that row
            square_costs[i] = costs[device_max[batch]];

        }

        // if the position represents a cell to copy
        else if (batch < batch_size) {

            // calculate the position in the source tensor and copy
            pos = col + row * size_m + batch * size_m * size_n
            square_costs[i] = costs[pos];

        }

    }

}

template <typename T>
__global__ void replace_infinities(int32 batch_size,
                                   int32 size,
                                   int32* device_max,
                                   T infinity,
                                   const T* costs,
                                   T* square_costs) {

    // calculate the maximum number of thread calls to make
    int32 n = batch_size * size * size;

    // this case only occurs when all values are infinite.
    if (max == infinity) {
        max = 0;
    }

    // a value higher than the maximum value present in the matrix.
    else {
        max++;
    }

    // run each thread at least once
    for (int32 i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n; i += blockDim.x * gridDim.x) {

        // calculate the position in the destination tensor
        int32 batch = i / (size * size);

        // replace the matrix entries that are infinity
        if (infinity == square_costs[i]) {
            square_costs[i] = costs[device_max[batch]];
        }

    }

}

template <typename T>
__global__ void minimize_along_direction(int32 batch_size,
                                         int32 size,
                                         int32* device_min,
                                         bool cols_fixed,
                                         T* square_costs) {

    // calculate the maximum number of thread calls to make
    int32 n = batch_size * size * size;

    // run each thread at least once
    for (int32 i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n; i += blockDim.x * gridDim.x) {

        // calculate the position in the destination tensor
        int32 col = i % size;
        int32 row = (i / size) % size;
        int32 batch = i / (size * size);

        // determine how to index into pos: are rows or columns fixed
        int32 pos = batch * size + (cols_fixed ? row : col);

        // replace the matrix entries that are infinity
        if (0 < device_min[pos]) {
            square_costs[i] -= square_costs[device_min[pos]];
        }

    }

}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct HungarianFunctor<GPUDevice, T> {

    void operator()(const OpKernelContext* context,
                    const GPUDevice& d,
                    int32 batch_size,
                    int32 size_n,
                    int32 size_m,
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
        cudaMalloc((void**)&device_max, sizeof(int32) * batch_size);

        // the size of each batch-wise matrix in costs
        int32 n = size_n * size_m;

        // launch several parallel max operations
        for (int32 i = 0; i < batch_size; i++) {
            cublasIsamax(handle, n, costs + n * i, 1, device_max + i);
        }

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        // allocate memory for a square costs matrix
        T* square_costs;
        cudaMalloc((void**)&square_costs, sizeof(T) * batch_size * size * size);

        // fill the resized matrix with the original matrix values
        resize<T><<<32, 256>>>(batch_size,
                               size_n,
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
        replace_infinities<T><<<32, 256>>>(batch_size,
                                           size,
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
        cudaMalloc((void**)&device_min, sizeof(int32) * batch_size * size);

        bool cols_fixed = size_n >= size_m;

        // launch several parallel min operations
        for (int32 i = 0; i < batch_size; i++) {
            for (int32 j = 0; j < size; j++) {

                // stride iterates over rows when cols_fixed is true
                // and over cols when cols_fixed is false
                cublasIsamin(
                    handle,
                    size,
                    square_costs + size * size * i + (cols_fixed ? 1 : size) * j,
                    cols_fixed ? size : 1, // how many cells apart is the next element
                    device_min + i * size + j);

            }
        }

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        // replace all infinities with the max value in the matrix
        minimize_along_direction<T><<<32, 256>>>(batch_size,
                                                 size,
                                                 device_min,
                                                 cols_fixed,
                                                 square_costs);

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        // flip the variable and normalize in the opposite direction
        cols_fixed = !cols_fixed;

        // launch several parallel min operations
        for (int32 i = 0; i < batch_size; i++) {
            for (int32 j = 0; j < size; j++) {

                // stride iterates over rows when cols_fixed is true
                // and over cols when cols_fixed is false
                cublasIsamin(
                    handle,
                    size,
                    square_costs + size * size * i + (cols_fixed ? 1 : size) * j,
                    cols_fixed ? size : 1, // how many cells apart is the next element
                    device_min + i * size + j);

            }
        }

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        // replace all infinities with the max value in the matrix
        minimize_along_direction<T><<<32, 256>>>(batch_size,
                                                 size,
                                                 device_min,
                                                 cols_fixed,
                                                 square_costs);

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        // remove the memory allocated for calculating the max
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
        cudaMalloc((void**)&masks, sizeof(int32) * batch_size * size * size);

        // allocate space for several mask tensors and set to zero
        bool* row_masks;
        cudaMalloc((void**)&row_masks, sizeof(bool) * batch_size * size);
        cudaMemset(row_masks, 0x0, sizeof(bool) * batch_size * size);

        // allocate space for several mask tensors and set to zero
        bool* col_masks;
        cudaMalloc((void**)&col_masks, sizeof(bool) * batch_size * size);
        cudaMemset(col_masks, 0x0, sizeof(bool) * batch_size * size);



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
