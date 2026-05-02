#include "../include/io_utils.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>

namespace openmp {

/**
 * @brief Implementation of ReadBinaryFile.
 *
 * @copydetails ReadBinaryFile
 */
std::optional<Dataset> ReadBinaryFile(const std::filesystem::path &filepath) {
  Dataset dataset;

  if (!std::filesystem::exists(filepath)) {
    std::cerr << "File does not exist at path: " << filepath << "\n";
    return std::nullopt;
  }

  std::ifstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Failed to open file: " << filepath << "\n";
    return std::nullopt;
  }

  file.read(reinterpret_cast<char *>(&dataset.n_rows), sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(&dataset.n_cols), sizeof(uint32_t));

  if (!file) {
    std::cerr << "Failed to read dataset dimensions.\n";
    return std::nullopt;
  }

  const size_t total_elements =
      static_cast<size_t>(dataset.n_rows) * dataset.n_cols;

  dataset.data.resize(total_elements);

  file.read(reinterpret_cast<char *>(dataset.data.data()),
            total_elements * sizeof(float));

  if (!file) {
    std::cerr << "Error reading dataset values or EOF reached unexpectedly.\n";
  }

  return dataset;
}

} // namespace openmp