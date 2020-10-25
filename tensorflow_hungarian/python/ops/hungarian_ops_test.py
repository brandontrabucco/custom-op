# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for hungarian ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy.optimize import linear_sum_assignment
import numpy as np
import tensorflow as tf
import time

from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
try:
    from tensorflow_hungarian.python.ops import hungarian_ops
except ImportError:
    from . import hungarian_ops


class HungarianTest(test.TestCase):

    @test_util.run_gpu_only
    def test_hungarian(self):
        with self.test_session():

            costs = np.array([[1, 2], [3, 4]]).astype(np.int32)
            a_real, b_real = linear_sum_assignment(costs)

            with ops.device("/gpu:0"):

                a = np.arange(np.min(costs.shape))
                b = hungarian_ops.hungarian(costs[np.newaxis]).numpy()[0]

            self.assertAllClose(a, a_real)
            self.assertAllClose(b, b_real)


class LargeHungarianTest(test.TestCase):

    @test_util.run_gpu_only
    def test_large_hungarian(self):
        with self.test_session():

            costs = np.random.randint(100, size=[32, 32]).astype(np.int32)
            a_real, b_real = linear_sum_assignment(costs)

            with ops.device("/gpu:0"):

                a = np.arange(np.min(costs.shape))
                b = hungarian_ops.hungarian(costs[np.newaxis]).numpy()[0]

            self.assertAllClose(a, a_real)
            self.assertAllClose(b, b_real, msg="GPU does not match")


class SpeedHungarianTest(test.TestCase):

    @test_util.run_gpu_only
    def test_speed_hungarian(self):
        with self.test_session():

            costs = np.random.randint(100, size=[64, 128, 128]).astype(np.int32)

            start_python = time.time()
            a_real, b_real = linear_sum_assignment(costs[0])
            end_python = time.time()

            with ops.device("/gpu:0"):
                inputs = tf.constant(costs)
                inputs = inputs * tf.ones_like(inputs)
                inputs = inputs + tf.zeros_like(inputs)

            start_gpu = time.time()
            with ops.device("/gpu:0"):
                out_gpu = hungarian_ops.hungarian(inputs)

            with ops.device("/gpu:0"):
                out_gpu = tf.identity(out_gpu)
                out_gpu = out_gpu * tf.ones_like(out_gpu)
                out_gpu = out_gpu + tf.zeros_like(out_gpu)
            end_gpu = time.time()

            start_cpu = time.time()
            with ops.device("/cpu:0"):
                out_cpu = hungarian_ops.hungarian(inputs)

            with ops.device("/gpu:0"):
                out_cpu = tf.identity(out_cpu)
                out_cpu = out_cpu * tf.ones_like(out_cpu)
                out_cpu = out_cpu + tf.zeros_like(out_cpu)
            end_cpu = time.time()

            b_gpu = out_gpu.numpy()[0]
            b_cpu = out_cpu.numpy()[0]

            py_time = end_python - start_python
            cpu_time = end_cpu - start_cpu
            gpu_time = end_gpu - start_gpu

            self.assertAllClose(b_cpu, b_real, msg="CPU does not match")
            self.assertTrue(cpu_time > gpu_time,
                            msg="GPU is {}% slower than {} sec ({} sec in python)".format(
                                (gpu_time - cpu_time) / cpu_time, cpu_time, py_time))


if __name__ == '__main__':
    test.main()
