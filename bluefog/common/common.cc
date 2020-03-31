// Modifications copyright (C) 2020 Bluefog Team. All Rights Reserved.
// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#include "common.h"

#include <cassert>
#include <sstream>

namespace bluefog {
namespace common {

const std::string& DataType_Name(DataType value) {
  switch (value) {
    case DataType::BLUEFOG_UINT8:
      static const std::string uint8("uint8");
      return uint8;
    case DataType::BLUEFOG_INT8:
      static const std::string int8("int8");
      return int8;
    case DataType::BLUEFOG_UINT16:
      static const std::string uint16("uint16");
      return uint16;
    case DataType::BLUEFOG_INT16:
      static const std::string int16("int16");
      return int16;
    case DataType::BLUEFOG_INT32:
      static const std::string int32("int32");
      return int32;
    case DataType::BLUEFOG_INT64:
      static const std::string int64("int64");
      return int64;
    case DataType::BLUEFOG_FLOAT16:
      static const std::string float16("float16");
      return float16;
    case DataType::BLUEFOG_FLOAT32:
      static const std::string float32("float32");
      return float32;
    case DataType::BLUEFOG_FLOAT64:
      static const std::string float64("float64");
      return float64;
    case DataType::BLUEFOG_BOOL:
      static const std::string bool_("bool");
      return bool_;
    default:
      static const std::string unknown("<unknown>");
      return unknown;
  }
}

Status::Status() = default;

Status::Status(StatusType type, std::string reason) {
  type_ = type;
  reason_ = reason;
}

Status Status::OK() { return Status(); }

Status Status::UnknownError(std::string message) {
  return Status(StatusType::UNKNOWN_ERROR, message);
}

Status Status::PreconditionError(std::string message) {
  return Status(StatusType::PRECONDITION_ERROR, message);
}

Status Status::Aborted(std::string message) {
  return Status(StatusType::ABORTED, message);
}

Status Status::InvalidArgument(std::string message) {
  return Status(StatusType::INVALID_ARGUMENT, message);
}

Status Status::InProgress() { return Status(StatusType::IN_PROGRESS, ""); }

bool Status::ok() const { return type_ == StatusType::OK; }

bool Status::in_progress() const { return type_ == StatusType::IN_PROGRESS; }

StatusType Status::type() const { return type_; }

const std::string& Status::reason() const { return reason_; }

void TensorShape::AddDim(int64_t dim) { shape_.push_back(dim); }

void TensorShape::AppendShape(TensorShape& other) {
  for (auto dim : other.shape_) {
    shape_.push_back(dim);
  }
}

const std::string TensorShape::DebugString() const {
  std::stringstream args;
  args << "[";
  for (auto it = shape_.begin(); it != shape_.end(); ++it) {
    if (it != shape_.begin()) {
      args << ", ";
    }
    args << *it;
  }
  args << "]";
  return args.str();
}

int TensorShape::dims() const { return (int)shape_.size(); }

int64_t TensorShape::dim_size(int idx) const {
  assert(idx >= 0);
  assert(idx < shape_.size());
  return shape_[idx];
}

int64_t TensorShape::num_elements() const {
  int64_t result = 1;
  for (auto dim : shape_) {
    result *= dim;
  }
  return result;
}

const std::vector<int64_t>& TensorShape::to_vector() const { return shape_; }

}  // namespace common
}  // namespace bluefog
