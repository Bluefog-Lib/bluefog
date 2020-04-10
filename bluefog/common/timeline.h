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

#ifndef BLUEFOG_COMMON_TIMELINE_H
#define BLUEFOG_COMMON_TIMELINE_H

#include <atomic>
#include <boost/lockfree/spsc_queue.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <vector>

#include "common.h"

namespace bluefog {
namespace common {

enum TimelineRecordType { EVENT };

struct TimelineRecord {
  TimelineRecordType type;
  std::string tensor_name;
  char phase;
  std::string op_name;
  std::thread::id tid;
  long ts_micros;
};

class TimelineWriter {
 public:
  void Initialize(std::string file_name);
  inline bool IsHealthy() const { return healthy_; }
  void EnqueueWriteEvent(const std::string& tensor_name, char phase,
                         const std::string& op_name, 
                         const std::thread::id tid, long ts_micros);

 private:
  void DoWriteEvent(const TimelineRecord& r);
  void WriterLoop();

  // Are we healthy?
  std::atomic_bool healthy_{false};

  // Timeline file.
  std::ofstream file_;

  // Timeline record queue.
  boost::lockfree::spsc_queue<TimelineRecord,
                              boost::lockfree::capacity<1048576>>
      record_queue_;

  // Mapping of tensor names to indexes. It is used to reduce size of the
  // timeline file.
  std::unordered_map<std::string, int> tensor_table_;

  // Mapping of thread ID to indexes. It is used to transform thread::id
  // to int and reduce size of the timeline file.
  std::unordered_map<std::thread::id, int> tid_table_;
};

enum TimelineState { ACTIVITY, TOP_LEVEL };

// Writes timeline in Chrome Tracing format. Timeline spec is from:
// https://github.com/catapult-project/catapult/tree/master/tracing
class Timeline {
 public:
  void Initialize(const std::string& file_name, unsigned int bluefog_size);
  inline bool Initialized() const { return initialized_; }

  void ActivityStart(const std::string& tensor_name,
                     const std::string& activity,
                     const std::thread::id* tid_ptr = nullptr);
  void ActivityEnd(const std::string& tensor_name,
                   const std::thread::id* tid_ptr = nullptr);

 private:
  long TimeSinceStartMicros() const;
  void WriteEvent(const std::string& tensor_name, char phase,
                  const std::thread::id tid, const std::string& op_name = "");

  // Boolean flag indicating whether Timeline was initialized (and thus should
  // be recorded).
  bool initialized_ = false;

  // Timeline writer.
  TimelineWriter writer_;

  // Time point when Bluefog was started.
  std::chrono::steady_clock::time_point start_time_;

  // A mutex that guards timeline state from concurrent access.
  std::recursive_mutex mutex_;

  // Current state of each tensor in the timeline.
  std::unordered_map<std::string, TimelineState> tensor_states_;

  // Map of ranks to their string representations.
  // std::to_string() is very slow.
  std::vector<std::string> rank_strings_;
};

}  // namespace common
}  // namespace bluefog

#endif  // BLUEFOG_COMMON_TIMELINE_H