#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <num_rows> <num_cols> <output_file>\n";
    std::cerr << "Example: " << argv[0] << " 1000 3 ../data/dataset.bin\n";
    return EXIT_FAILURE;
  }

  uint32_t num_rows = static_cast<uint32_t>(std::stoul(argv[1]));
  uint32_t num_cols = static_cast<uint32_t>(std::stoul(argv[2]));
  std::string filepath = argv[3];

  uint64_t total_elements = static_cast<uint64_t>(num_rows) * num_cols;
  std::vector<float> data(total_elements);

  // Generador de números aleatorios de alta calidad (Mersenne Twister)
  std::mt19937 rng(42); // Semilla fija para resultados reproducibles
  std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

  std::cout << "Generating " << num_rows << " x " << num_cols << " dataset ("
            << (total_elements * 4) / (1024 * 1024.0) << " MB)...\n";

  for (uint64_t i = 0; i < total_elements; ++i) {
    data[i] = dist(rng);
  }

  // Escribir en formato binario
  std::ofstream file(filepath, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Could not open file " << filepath << " for writing.\n";
    return EXIT_FAILURE;
  }

  // 1. Escribir metadatos (UINT32)
  file.write(reinterpret_cast<const char *>(&num_rows), sizeof(uint32_t));
  file.write(reinterpret_cast<const char *>(&num_cols), sizeof(uint32_t));

  // 2. Escribir datos reales (REAL32)
  file.write(reinterpret_cast<const char *>(data.data()),
             total_elements * sizeof(float));

  if (!file) {
    std::cerr << "Error writing to file.\n";
    return EXIT_FAILURE;
  }

  std::cout << "Dataset successfully saved to " << filepath << "\n";
  return EXIT_SUCCESS;
}