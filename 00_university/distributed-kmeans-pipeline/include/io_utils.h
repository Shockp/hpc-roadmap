#ifndef UNIVERSITY_OPENMP_IO_UTILS_H_
#define UNIVERSITY_OPENMP_IO_UTILS_H_

#include <filesystem>
#include <optional>

#include "models/dataset.h"

namespace openmp {

/**
 * @brief Reads a dataset from a binary file.
 *
 * Opens a binary file specified by the given filepath, reads the dimensions
 * (number of rows and columns), and then reads the corresponding dataset
 * values.
 *
 * The expected binary format is:
 * - uint32_t n_rows
 * - uint32_t n_cols
 * - float data[n_rows * n_cols]
 *
 * @param filepath The path to the binary file to be read.
 * @return Dataset The dataset populated with the dimensions and data from the
 * file.
 * @throws std::runtime_error If the file cannot be opened, dimensions cannot be
 *         read, or if there is an error reading the dataset values.
 */
std::optional<Dataset> ReadBinaryFile(const std::filesystem::path &filepath);

} // namespace openmp

#endif // UNIVERSITY_OPENMP_IO_UTILS_H_