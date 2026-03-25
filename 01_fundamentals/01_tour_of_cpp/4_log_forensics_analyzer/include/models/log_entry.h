#ifndef LOG_FORENSICS_ANALYZER_LOG_ENTRY_H_
#define LOG_FORENSICS_ANALYZER_LOG_ENTRY_H_

#include <ctime>
#include <string>

namespace logforensics {

struct LogEntry {
  std::string ip_address;
  std::time_t timestamp;
  int status_code;
};

}  // namespace logforensics

#endif  // LOG_FORENSICS_ANALYZER_LOG_ENTRY_H_