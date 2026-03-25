#include <algorithm>
#include <filesystem>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../include/log_analyzer.h"
#include "../include/log_generator.h"
#include "../include/models/log_entry.h"

int main() {
  std::filesystem::path kLogFileName = "../data/server.log";
  const int kNumLines = 100000;

  std::cout << "Generating log file with " << kNumLines << " lines...\n";
  logforensics::GenerateMockFile(kLogFileName, kNumLines);

  std::cout << "Parsing log file...\n";
  auto parsed = logforensics::ParseLogFile(kLogFileName);
  if (!parsed) {
    std::cerr << "Error: Failed to parse log file.\n";
    return 1;
  }
  std::vector<logforensics::LogEntry> entries = std::move(parsed.value());

  std::cout << "Total parsed entries: " << entries.size() << "\n\n";

  // --- Algorithm 1: Filter 404 Errors ---
  std::vector<logforensics::LogEntry> errors_404;
  std::copy_if(
      entries.begin(), entries.end(), std::back_inserter(errors_404),
      [](const logforensics::LogEntry &e) { return e.status_code == 404; });

  std::cout << "Found " << errors_404.size() << " HTTP 404 errors.\n\n";

  // --- Algorithm 2: Count Frequencies ---
  std::unordered_map<std::string, int> ip_counts;
  std::for_each(entries.begin(), entries.end(),
                [&ip_counts](const logforensics::LogEntry &e) {
                  ip_counts[e.ip_address]++;
                });

  // --- Algorithm 3: Sort by Frequency ---
  // Transfer map to vector of pairs for sorting
  std::vector<std::pair<std::string, int>> sorted_ips(ip_counts.begin(),
                                                      ip_counts.end());

  std::sort(sorted_ips.begin(), sorted_ips.end(),
            [](const auto &a, const auto &b) {
              return a.second > b.second;  // Descending order
            });

  // --- Output using our constrained template ---
  std::cout << "--- Top 3 Most Active IPs ---\n";
  logforensics::PrintTopN(sorted_ips, 3);

  return 0;
}