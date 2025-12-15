// Copyright 2025 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "litert/runtime/profiler_summarizer.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "tflite/core/api/profiler.h"
#include "tflite/core/interpreter.h"
#include "tflite/profiling/profile_buffer.h"

namespace litert {
namespace profiling {

namespace {

// Helper to update stats
void UpdateStat(OpStat& stat, int64_t time_us) {
  if (stat.count == 0) {
    stat.first_run_time_us = time_us;
    stat.min_time_us = time_us;
    stat.max_time_us = time_us;
  } else {
    stat.min_time_us = std::min(stat.min_time_us, time_us);
    stat.max_time_us = std::max(stat.max_time_us, time_us);
  }
  stat.total_time_us += time_us;
  stat.count++;
}

}  // namespace

LiteRtProfileSummarizer::LiteRtProfileSummarizer() = default;

void LiteRtProfileSummarizer::ProcessProfiles(
    const std::vector<const tflite::profiling::ProfileEvent*>& profile_stats,
    const tflite::Interpreter& interpreter) {
  for (auto event : profile_stats) {
    if (event->event_type ==
        tflite::Profiler::EventType::OPERATOR_INVOKE_EVENT) {
      int node_index = event->event_metadata;
      int subgraph_index = event->extra_event_metadata;

      std::string type_in_stats(event->tag);
      const auto* node_and_reg =
          interpreter.node_and_registration(subgraph_index, node_index);
      if (node_and_reg) {
        const char* profiling_string = interpreter.OpProfilingString(
            node_and_reg->second, &node_and_reg->first);
        if (profiling_string) {
          type_in_stats += "/";
          type_in_stats += profiling_string;
        }
      }

      UpdateStat(stats_[type_in_stats], event->elapsed_time);

    } else if (event->event_type ==
               tflite::Profiler::EventType::DELEGATE_OPERATOR_INVOKE_EVENT) {
      std::string op_name = event->tag;
      UpdateStat(delegate_stats_[op_name], event->elapsed_time);
    }
  }
}

std::string LiteRtProfileSummarizer::GetOutputString() const {
  std::stringstream ss;
  ss << "Profile Summary:\n";
  ss << "================\n";

  auto print_stats = [&](const std::map<std::string, OpStat>& stats,
                         const char* title) {
    if (stats.empty()) return;
    ss << title << ":\n";
    ss << absl::StrFormat("%-30s %10s %10s %10s %10s %10s\n", "Op Name",
                          "Count", "Avg(us)", "Min(us)", "Max(us)",
                          "Total(us)");
    for (const auto& [name, stat] : stats) {
      double avg = static_cast<double>(stat.total_time_us) / stat.count;
      ss << absl::StrFormat("%-30s %10lld %10.2f %10lld %10lld %10lld\n", name,
                            stat.count, avg, stat.min_time_us, stat.max_time_us,
                            stat.total_time_us);
    }
    ss << "\n";
  };

  print_stats(stats_, "Operator-wise Statistics");
  print_stats(delegate_stats_, "Delegate Statistics");

  return ss.str();
}

}  // namespace profiling
}  // namespace litert
