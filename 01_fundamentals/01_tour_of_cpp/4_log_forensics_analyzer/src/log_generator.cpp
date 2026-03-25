#include "../include/log_generator.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

namespace logforensics {

void GenerateMockFile(const std::filesystem::path &filepath, int num_lines) {
  std::ofstream out_file(filepath);
  if (!out_file.is_open()) {
    std::cerr << "Failed to open " << filepath << " for writing.\n";
    return;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> ip_dist(1, 5);
  std::uniform_real_distribution<> status_dist(1, 10);

  for (int i = 0; i < num_lines; ++i) {
    std::string ip = "192.168.1." + std::to_string(ip_dist(gen));
    int status_code = (status_dist(gen) == 1) ? 404 : 200;

    out_file << "IP: " << ip
             << " - [2026-03-01 10:00:00] STATUS: " << status_code
             << " \"GET /index.html HTTP/1.1\"\n";
  }
  out_file.close();
}

}  // namespace logforensics