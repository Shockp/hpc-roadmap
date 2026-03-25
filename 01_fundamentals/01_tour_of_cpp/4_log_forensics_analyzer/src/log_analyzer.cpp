#include "../include/log_analyzer.h"

#include <ctime>
#include <fstream>
#include <iomanip>
#include <optional>
#include <regex>
#include <sstream>
#include <string>

namespace logforensics {

std::optional<std::vector<LogEntry>> ParseLogFile(
    std::filesystem::path &filepath) {
  std::vector<LogEntry> entries;
  std::ifstream in_file(filepath);

  if (!std::filesystem::exists(filepath)) {
    std::cerr << "Failed to open " << filepath << " for reading.\n";
    return std::nullopt;
  }

  if (!in_file.is_open()) {
    std::cerr << "Failed to open " << filepath << "\n";
    return std::nullopt;
  }

  std::string line;

  std::regex pattern(
      R"(IP:\s*(\d{1,3}(?:\.\d{1,3}){3})\s*-\s*\[(.*?)\]\s*.*STATUS:\s*(\d+))");
  std::smatch matches;

  while (std::getline(in_file, line)) {
    if (std::regex_search(line, matches, pattern)) {
      LogEntry entry;
      entry.ip_address = matches[1].str();

      std::tm tm = {};
      std::istringstream ss(matches[2].str());

      ss >> std::get_time(&tm, "%Y-%m-%d %H:%M:%S");

      entry.timestamp = std::mktime(&tm);

      entry.status_code = std::stoi(matches[3].str());
      entries.push_back(std::move(entry));
    }
  }

  return entries;
}

}  // namespace logforensics