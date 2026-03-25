#ifndef LOG_FORENSICS_ANALYZER_LOG_ANALYZER_H_
#define LOG_FORENSICS_ANALYZER_LOG_ANALYZER_H_

#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "./models/log_entry.h"

namespace logforensics {

std::optional<std::vector<LogEntry>> ParseLogFile(
    std::filesystem::path &filepath);

template <typename Container>
void PrintTopN(const Container &c, int n) {
  using ExpectedType = std::pair<std::string, int>;
  using ActualType = typename Container::value_type;

  static_assert(
      std::is_same_v<ExpectedType, ActualType>,
      "PrintTopN requires a container holding std::pair<std::string, int>.");

  int count = 0;
  for (auto it = c.begin(); it != c.end() && count < n; ++it, ++count) {
    std::cout << "IP: " << it->first << " | Count: " << it->second << "\n";
  }
}

}  // namespace logforensics

#endif  // LOG_FORENSICS_ANALYZER_LOG_ANALYZER_H_