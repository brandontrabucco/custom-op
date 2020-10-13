# TensorFlow Hungarian

Install a docker image to build the op library.

```bash
docker pull tensorflow/tensorflow:custom-op-gpu-ubuntu16
docker run --runtime=nvidia --privileged  -it -v ${PWD}:/working_dir -w /working_dir  tensorflow/tensorflow:custom-op-gpu-ubuntu16
```

Compile the op library using bazel and install it.

```bash
./configure.sh
bazel build build_pip_pkg
bazel-bin/build_pip_pkg artifacts
pip3 install artifacts/*.whl
```

Test the op library.

```bash
python3 tensorflow-hungarian/tensorflow_hungarian/python/ops/hungarian_ops_test.py
```

Import the library in a python script.

```python
import tensorflow as tf
import tensorflow_hungarian
print(tensorflow_hungarian.hungarian([[[1,2], [3,4]]]))
```
