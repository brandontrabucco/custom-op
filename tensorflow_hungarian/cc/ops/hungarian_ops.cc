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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using shape_inference::Shape;
using shape_inference::Dimension;
using shape_inference::DimensionHandle;
using shape_inference::ShapeHandle;
using std::vector;

REGISTER_OP("Hungarian")
    .Attr("T: {int32, float, double}")
    .Input("costs: T")
    .Output("assignments: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // access the input to the operation
        ShapeHandle input = c->input(0);

        // if the input has an unknown shape so does the op
        if (!c->RankKnown(input)) {
            c->set_output(0, c->UnknownShape());
            return Status::OK();
        }

        // otherwise, get the rank of the input tensor
        const int input_rank = c->Rank(input);

        // then, get the first input_rank - 1 dimensions of the rank
        std::vector<DimensionHandle> dims;
        for (int i = 0; i < input_rank - 1; ++i) {
            dims.emplace_back(c->Dim(input, i));
        }

        // and set the output shape to the those dimensions
        c->set_output(0, c->MakeShape(dims));
        return Status::OK();
    });
