CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

HUNGARIAN_SRCS = tensorflow_hungarian/cc/kernels/hungarian_kernels.cc $(wildcard tensorflow_hungarian/cc/kernels/*.h) $(wildcard tensorflow_hungarian/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

HUNGARIAN_GPU_ONLY_TARGET_LIB = tensorflow_hungarian/python/ops/_hungarian_ops.cu.o
HUNGARIAN_TARGET_LIB = tensorflow_hungarian/python/ops/_hungarian_ops.so

# hungarian op for GPU
hungarian_gpu_only: $(HUNGARIAN_GPU_ONLY_TARGET_LIB)

$(HUNGARIAN_GPU_ONLY_TARGET_LIB): tensorflow_hungarian/cc/kernels/hungarian_kernels.cu.cc
	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

hungarian_op: $(HUNGARIAN_TARGET_LIB)
$(HUNGARIAN_TARGET_LIB): $(HUNGARIAN_SRCS) $(HUNGARIAN_GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart

hungarian_test: tensorflow_hungarian/python/ops/hungarian_ops_test.py tensorflow_hungarian/python/ops/hungarian_ops.py $(HUNGARIAN_TARGET_LIB)
	$(PYTHON_BIN_PATH) tensorflow_hungarian/python/ops/hungarian_ops_test.py

clean:
	rm -f $(ZERO_OUT_TARGET_LIB) $(TIME_TWO_GPU_ONLY_TARGET_LIB) $(TIME_TWO_TARGET_LIB) $(HUNGARIAN_GPU_ONLY_TARGET_LIB) $(HUNGARIAN_TARGET_LIB)
