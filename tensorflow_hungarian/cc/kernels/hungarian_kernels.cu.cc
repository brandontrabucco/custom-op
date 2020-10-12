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
                                   const T infinity,
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

template <typename T>
void step1(cublasHandle_t handle,
           int32& state,
           const int32 size,
           int32* masks,
           bool* row_masks,
           bool* col_masks,
           T* square_costs) {

    // allocate memory for a square costs matrix
    int32* masks_buffer;
    cudaMalloc((void**)&masks_buffer, sizeof(int32) * size * size);

    // fill the resized matrix with the original matrix values
    star_mask<T><<<32, 256>>>(
        size, masks, masks_buffer, square_costs);

    // allow for every parallel operation on the GPU to finish
    cudaDeviceSynchronize();

    // remove the memory allocated for calculating the buffer
    cudaFree(masks_buffer);

    // determine which state to move to if not finished
    *state = (*state == 0) ? 0 : 2;

}

__global__ void count_mask(const int32 size,
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
            atomicExch(col_masks + col, 1);

        }

    }

}

template <typename T>
void step2(cublasHandle_t handle,
           int32& state,
           const int32 size,
           int32* masks,
           bool* row_masks,
           bool* col_masks,
           T* square_costs) {

    // allocate memory for a the cover counts on the gpu
    int32* counts;
    cudaMalloc((void**)&counts, sizeof(int32));
    cudaMemset(counts, 0x0, sizeof(int32));

    // fill the resized matrix with the original matrix values
    count_mask<T><<<32, 256>>>(
        size, masks, col_masks, counts);

    // allow for every parallel operation on the GPU to finish
    cudaDeviceSynchronize();

    // malloc space on the heap for the cpu cover count
    int32 counts_cpu = 0;

    // copy the cover count to cpu to control program flow
    cudaMemcpy(
        &counts_cpu, counts, sizeof(int32), cudaMemcpyDeviceToHost);

    // remove the memory allocated for calculating the buffer
    cudaFree(count);

    // determine which state to move to if not finished
    *state = (*state == 0) ? 0 : (counts_cpu >= size) ? 0 : 3;

}

template <typename T>
__global__ void find_uncovered(const int32 size,
                               bool* row_masks,
                               bool* col_masks,
                               T* square_costs,
                               bool* uncovered_buffer,
                               const T key) {

    // calculate the maximum number of thread calls to make
    int32 n = size * size;

    // run each thread at least once
    for (int32 i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n; i += blockDim.x * gridDim.x) {

        // calculate the position in the destination tensor
        int32 col = i % size;
        int32 row = (i / size) % size;

        // check which entries satisfy the conditions
        uncovered_buffer[i] = (!row_mask[row] &&
                               !row_mask[row] &&
                               (square_costs[i] == key)) ? 1 : 0;

    }

}

__global__ void find_star(const int32 size,
                          int32* masks,
                          bool* uncovered_buffer,
                          const int32 row) {

    // calculate the maximum number of thread calls to make
    int32 n = size * size;

    // run each thread at least once
    for (int32 i = blockIdx.x * blockDim.x + threadIdx.x;
            i < n; i += blockDim.x * gridDim.x) {

        // check which entries satisfy the conditions
        uncovered_buffer[i] = (masks[i] == STAR)) ? 1 : 0;

    }

}

void step3(cublasHandle_t handle,
           int32& state,
           const int32 size,
           int32* masks,
           bool* row_masks,
           bool* col_masks,
           T* square_costs) {

    // allocate memory for a the buffer on the gpu
    bool* uncovered_buffer;
    cudaMalloc((void**)&uncovered_buffer, sizeof(bool) * size * size);

    // fill the resized matrix with the original matrix values
    find_uncovered<T><<<32, 256>>>(
        size, row_masks, col_masks, square_costs, uncovered_buffer, 0);

    // allow for every parallel operation on the GPU to finish
    cudaDeviceSynchronize();

    // location of the maximal element in memory
    int32 result = 0;

    // launch several parallel max operations
    cublasIsamax(handle, size * size, uncovered_buffer, 1, &result);

    // allow for every parallel operation on the GPU to finish
    cudaDeviceSynchronize();

    // allocate storage for the result of the argmax
    bool value = false;

    // copy the value at the argmax location to the cpu
    cudaMemcpy(&value,
               uncovered_buffer + result,
               sizeof(bool),
               cudaMemcpyDeviceToHost);

    if (value) {

        // copy the prime indicator into the mask
        int32 temporary_prime = PRIME;
        cudaMemcpy(masks + result,
                   &temporary_prime,
                   sizeof(int32),
                   cudaMemcpyHostToDevice);

    }

    else {

        // otherwise move onto state 5
        *state = 5;
        return;

    }

    // location of the maximal element in memory
    int32 row = result / size;

    // fill the resized matrix with the original matrix values
    find_star<T><<<32, 256>>>(
        size, masks, uncovered_buffer, row);

    // allow for every parallel operation on the GPU to finish
    cudaDeviceSynchronize();

    // location of the maximal element in memory
    int32 col = 0;

    // launch several parallel max operations
    cublasIsamax(handle, size, uncovered_buffer + row * size, 1, &col);

    // allow for every parallel operation on the GPU to finish
    cudaDeviceSynchronize();

    // remove the memory allocated for calculating the buffer
    cudaFree(uncovered_buffer);

    // allocate storage for the result of the argmax
    bool value = false;

    // copy the value at the argmax location to the cpu
    cudaMemcpy(&value,
               uncovered_buffer + row * size + col,
               sizeof(bool),
               cudaMemcpyDeviceToHost);

    if (value) {

        // copy the target value onto the GPU
        bool row_value = true;
        cudaMemcpy(row_mask + row,
                   &row_value,
                   sizeof(bool),
                   cudaMemcpyHostToDevice);

        // copy the target value onto the GPU
        bool col_value = false;
        cudaMemcpy(col_mask + col,
                   &col_value,
                   sizeof(bool),
                   cudaMemcpyHostToDevice);

        // move onto state 3
        *state = 3;
        return;

    }

    // move onto state 4
    *state = 4;

}

void step4(cublasHandle_t handle,
           int32& state,
           const int32 size,
           int32* masks,
           bool* row_masks,
           bool* col_masks,
           T* square_costs) {

    // pass

}

void step5(cublasHandle_t handle,
           int32& state,
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
        resize<T><<<32, 256>>>(
            size_n, size_m, size, device_max, costs, square_costs);

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
        replace_infinities<T><<<32, 256>>>(
            size, device_max, costs, infinity, square_costs);

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
                cols_fixed ? size : 1,
                device_min + j);

        }

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        // replace all infinities with the max value in the matrix
        minimize_along_direction<T><<<32, 256>>>(
            size, device_min, cols_fixed, square_costs);

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
                cols_fixed ? size : 1,
                device_min + j);

        }

        // allow for every parallel operation on the GPU to finish
        cudaDeviceSynchronize();

        // replace all infinities with the max value in the matrix
        minimize_along_direction<T><<<32, 256>>>(
            size, device_min, cols_fixed, square_costs);

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
        int32 state = 1;

        /*
         *
         *
         * UPDATE THE SOLUTION IN A LOOP
         *
         *
         */

        while (state != 0) {

            // update the solution
            switch ( state ) {
            case 1:
                step1(handle, state, size, masks, row_masks, col_masks, square_costs);
                break;
            case 2:
                step2(handle, state, size, masks, row_masks, col_masks, square_costs);
                break;
            case 3:
                step3(handle, state, size, masks, row_masks, col_masks, square_costs);
                break;
            case 4:
                step4(handle, state, size, masks, row_masks, col_masks, square_costs);
                break;
            case 5:
                step5(handle, state, size, masks, row_masks, col_masks, square_costs);
                break;
            }

        }

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
