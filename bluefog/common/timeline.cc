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

#include <cassert>
#include <chrono>
#include <sstream>
#include <thread>

#include "logging.h"
#include "timeline.h"

namespace bluefog {
namespace common {

void TimelineWriter::Initialize(std::string file_name) {
  file_.open(file_name, std::ios::out | std::ios::trunc);
  if (file_.good()) {
    // Initialize the timeline with '[' character.
    file_ << "[\n";
    healthy_ = true;

    // Spawn writer thread.
    std::thread writer_thread(&TimelineWriter::WriterLoop, this);
    writer_thread.detach();
  } else {
    BFLOG(ERROR) << "Error opening the Bluefog Timeline file " << file_name
                 << ", will not write a timeline.";
  }
}

void TimelineWriter::EnqueueWriteEvent(const std::string& tensor_name,
                                       char phase, const std::string& op_name,
                                       const std::thread::id tid, long ts_micros) {
  TimelineRecord r{};
  r.type = TimelineRecordType::EVENT;
  r.tensor_name = tensor_name;
  r.phase = phase;
  r.op_name = op_name;
  r.tid = tid;
  r.ts_micros = ts_micros;

  while (healthy_ && !record_queue_.push(r))
    ;
}

void TimelineWriter::DoWriteEvent(const TimelineRecord& r) {
  assert(r.type == TimelineRecordType::EVENT);

  auto& tensor_idx = tensor_table_[r.tensor_name];
  auto& thread_idx = tid_table_[r.tid];
  if (thread_idx == 0) {
    thread_idx = (int)tid_table_.size();
  }
  if (tensor_idx == 0) {
    tensor_idx = (int)tensor_table_.size();

    // We model tensors as processes. Register metadata for this "pid".
    file_ << "{";
    file_ << "\"name\": \"process_name\"";
    file_ << ", \"ph\": \"M\"";
    file_ << ", \"pid\": " << tensor_idx << "";
    file_ << ", \"tid\": " << thread_idx << "";
    file_ << ", \"args\": {\"name\": \"" << r.tensor_name << "\"}";
    file_ << "}," << std::endl;
    file_ << "{";
    file_ << "\"name\": \"process_sort_index\"";
    file_ << ", \"ph\": \"M\"";
    file_ << ", \"pid\": " << tensor_idx << "";
    file_ << ", \"tid\": " << thread_idx << "";
    file_ << ", \"args\": {\"sort_index\": " << tensor_idx << "}";
    file_ << "}," << std::endl;
  }

  file_ << "{";
  file_ << "\"ph\": \"" << r.phase << "\"";
  if (r.phase != 'E') {
    // Not necessary for ending event.
    file_ << ", \"name\": \"" << r.op_name << "\"";
  }
  file_ << ", \"ts\": " << r.ts_micros << "";
  file_ << ", \"pid\": " << tensor_idx << "";
  file_ << ", \"tid\": " << thread_idx << "";
  file_ << "}," << std::endl;
}

void TimelineWriter::WriterLoop() {
  while (healthy_) {
    while (healthy_ && !record_queue_.empty()) {
      auto& r = record_queue_.front();
      switch (r.type) {
        case TimelineRecordType::EVENT:
          DoWriteEvent(r);
          break;
        default:
          throw std::logic_error("Unknown event type provided.");
      }
      record_queue_.pop();

      if (!file_.good()) {
        BFLOG(ERROR) << "Error writing to the Bluefog Timeline after it was "
                        "successfully opened, will stop writing the timeline.";
        healthy_ = false;
      }
    }

    // Allow scheduler to schedule other work for this core.
    std::this_thread::yield();
  }
}

void Timeline::Initialize(const std::string& file_name, unsigned int bluefog_size) {
  if (initialized_) {
    return;
  }

  // Start the writer.
  writer_.Initialize(file_name);

  // Initialize if we were able to open the file successfully.
  initialized_ = writer_.IsHealthy();

  // Pre-initialize the string representation for each rank.
  rank_strings_ = std::vector<std::string>(bluefog_size);
  for (unsigned int i = 0; i < bluefog_size; i++) {
    rank_strings_[i] = std::to_string(i);
  }
}

long Timeline::TimeSinceStartMicros() const {
  auto now = std::chrono::steady_clock::now();
  auto ts = now - start_time_;
  return std::chrono::duration_cast<std::chrono::microseconds>(ts).count();
}

// Write event to the Bluefog Timeline file.
void Timeline::WriteEvent(const std::string& tensor_name, const char phase,
                          const std::thread::id tid, const std::string& op_name) {
  auto ts_micros = TimeSinceStartMicros();
  writer_.EnqueueWriteEvent(tensor_name, phase, op_name, tid, ts_micros);
}

void Timeline::ActivityStart(const std::string& tensor_name,
                             const std::string& activity,
                             const std::thread::id* tid_ptr) {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(tensor_states_[tensor_name] == TimelineState::TOP_LEVEL);
  std::thread::id tid;
  if (tid_ptr == nullptr) {
    tid = std::this_thread::get_id();
  } else {
    tid = *tid_ptr;
  }
  WriteEvent(tensor_name, 'B', tid, activity);
  tensor_states_[tensor_name] = TimelineState::ACTIVITY;
}

void Timeline::ActivityEnd(const std::string& tensor_name,
                           const std::thread::id* tid_ptr) {
  if (!initialized_) {
    return;
  }

  std::lock_guard<std::recursive_mutex> guard(mutex_);
  assert(tensor_states_[tensor_name] == TimelineState::ACTIVITY);
  std::thread::id tid;
  if (tid_ptr == nullptr) {
    tid = std::this_thread::get_id();
  } else {
    tid = *tid_ptr;
  }
  WriteEvent(tensor_name, 'E', tid);
  tensor_states_[tensor_name] = TimelineState::TOP_LEVEL;
}

}  // namespace common
}  // namespace bluefog