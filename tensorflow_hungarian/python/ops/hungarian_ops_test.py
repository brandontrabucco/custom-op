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

from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
try:
    from tensorflow_hungarian.python.ops import hungarian_ops
except ImportError:
    from . import hungarian_ops


class HungarianTest(test.TestCase):

    @test_util.run_gpu_only
    def testHungarian(self):
        with self.test_session():

            costs = np.array([[1, 2], [3, 4]])
            a_real, b_real = linear_sum_assignment(costs)

            with ops.device("/gpu:0"):

                a = np.arange(np.min(costs.shape))
                b = hungarian_ops.hungarian(costs[np.newaxis]).numpy()[0]

            self.assertAllClose(a, a_real)
            self.assertAllClose(b, b_real)


if __name__ == '__main__':
    test.main()
