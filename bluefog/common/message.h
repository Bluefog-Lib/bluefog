// Modifications copyright (C) 2020 Bluefog Team. All Rights Reserved.
// Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
// Copyright 2016 The TensorFlow Authors.
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

#ifndef BLUEFOG_MESSAGE_H
#define BLUEFOG_MESSAGE_H

#include <string>
#include <vector>

#include "common.h"

namespace bluefog {
namespace common {

// A Request is a message sent from a rank greater than zero to the
// coordinator (rank zero), informing the coordinator of an operation that
// the rank wants to do and the tensor that it wants to apply the operation to.
class Request {
 public:
  enum RequestType {
    UNKNOWN = 0,
    ALLREDUCE = 1,
    ALLGATHER = 2,
    BROADCAST = 3,
    NEIGHBOR_ALLREDUCE = 4,
    NEIGHBOR_ALLGATHER = 5,
    WIN_PUT = 6,
    WIN_GET = 7,
    WIN_ACCUMULATE = 8,
    BARRIER=9,
    PAIR_GOSSIP = 10
  };

  static const std::string& RequestType_Name(RequestType value);

  // The request rank is necessary to create a consistent ordering of results,
  // for example in the allgather where the order of outputs should be sorted
  // by rank.
  int32_t request_rank() const;

  void set_request_rank(int32_t value);

  RequestType request_type() const;

  void set_request_type(RequestType value);

  DataType tensor_type() const;

  void set_tensor_type(DataType value);

  const std::string& tensor_name() const;

  void set_tensor_name(const std::string& value);

  int32_t root_rank() const;

  void set_root_rank(int32_t value);

  int32_t device() const;

  void set_device(int32_t value);

  static void ParseFromBytes(Request& request, const uint8_t* input);

  static void SerializeToString(const Request& request, std::string& output);

private:
  int32_t request_rank_ = 0;
  RequestType request_type_ = RequestType::ALLREDUCE;
  DataType tensor_type_ = DataType::BLUEFOG_UINT8;
  int32_t root_rank_ = 0;
  int32_t device_ = 0;
  std::string tensor_name_;
};

class RequestList {
public:
  const std::vector<Request>& requests() const;

  void set_requests(const std::vector<Request>& value);

  void add_request(const Request& value);

  void emplace_request(Request&& value);

  bool shutdown() const;

  void set_shutdown(bool value);

  static void ParseFromBytes(RequestList& request_list,
                             const uint8_t* input);

  static void SerializeToString(const RequestList& request_list,
                                std::string& output);

private:
  std::vector<Request> requests_;
  bool shutdown_ = false;
};

// A Response is a message sent from the coordinator (rank zero) to a rank
// greater than zero, informing the rank of an operation should be performed
// now. If the operation requested would result in an error (for example, due
// to a type or shape mismatch), then the Response can contain an error and
// an error message instead.
class Response {
public:
 enum ResponseType {
   ERROR = 0,
   ALLREDUCE = 1,
   ALLGATHER = 2,
   BROADCAST = 3,
   NEIGHBOR_ALLREDUCE = 4,
   NEIGHBOR_ALLGATHER = 5
 };

 static const std::string& ResponseType_Name(ResponseType value);

 ResponseType response_type() const;

 void set_response_type(ResponseType value);

 // Empty if the type is DONE or SHUTDOWN.
 const std::vector<std::string>& tensor_names() const;

 const std::string tensor_names_string() const;

 void set_tensor_names(const std::vector<std::string>& value);

 void add_tensor_name(const std::string& value);

 // Empty unless response_type is ERROR.
 const std::string& error_message() const;

 void set_error_message(const std::string& value);

 const std::vector<int32_t>& devices() const;

 void set_devices(const std::vector<int32_t>& value);

 void add_device(int32_t value);

 static void ParseFromBytes(Response& response, const uint8_t* input);

 static void SerializeToString(const Response& response, std::string& output);

private:
  ResponseType response_type_ = ResponseType::ALLREDUCE;
  std::vector<std::string> tensor_names_;
  std::string error_message_;
  std::vector<int32_t> devices_;
};

class ResponseList {
public:
  const std::vector<Response>& responses() const;

  void set_responses(const std::vector<Response>& value);

  void add_response(const Response& value);

  void emplace_response(Response&& value);

  bool shutdown() const;

  void set_shutdown(bool value);

  static void ParseFromBytes(ResponseList& response_list,
                             const uint8_t* input);

  static void SerializeToString(const ResponseList& response_list,
                                std::string& output);

private:
  std::vector<Response> responses_;
  bool shutdown_ = false;
};

} // namespace common
} // namespace bluefog

#endif // BLUEFOG_MESSAGE_H
