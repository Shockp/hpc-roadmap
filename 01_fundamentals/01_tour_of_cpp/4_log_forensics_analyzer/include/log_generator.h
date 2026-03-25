#ifndef LOG_FORENSICS_ANALYZER_LOG_GENERATOR_H_
#define LOG_FORENSICS_ANALYZER_LOG_GENERATOR_H_

#include <filesystem>
#include <string>

namespace logforensics {

void GenerateMockFile(const std::filesystem::path &filepath, int num_lines);

}  // namespace logforensics

#endif  // LOG_FORENSICS_ANALYZER_LOG_GENERATOR_H_