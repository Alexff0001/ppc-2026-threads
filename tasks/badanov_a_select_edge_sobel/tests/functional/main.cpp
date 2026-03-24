#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "badanov_a_select_edge_sobel/common/include/common.hpp"
#include "badanov_a_select_edge_sobel/omp/include/ops_omp.hpp"
#include "badanov_a_select_edge_sobel/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace badanov_a_select_edge_sobel {

class BadanovASelectEdgeSobelFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    std::string name = std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
    for (char &c : name) {
      if ((std::isalnum(static_cast<unsigned char>(c)) == 0) && c != '_') {
        c = '_';
      }
    }
    return name;
  }

 protected:
  void SetUp() override {
    const auto &[threshold, filename] =
        std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    threshold_ = threshold;
    filename_ = filename;

    const std::string abs_path =
        ppc::util::GetAbsoluteTaskPath(std::string(PPC_ID_badanov_a_select_edge_sobel), filename);

    std::ifstream file(abs_path);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file: " + abs_path);
    }

    int width = 0;
    int height = 0;
    file >> width >> height;

    const size_t total_pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    input_data_.resize(total_pixels);

    for (size_t i = 0; i < total_pixels; ++i) {
      int pixel_value = 0;
      file >> pixel_value;
      input_data_[i] = static_cast<uint8_t>(pixel_value);
    }

    file.close();
  }

  static bool CheckAllPixelsZero(const std::vector<uint8_t> &data) {
    return std::ranges::all_of(data.begin(), data.end(), [](uint8_t pixel) { return pixel == 0; });
  }

  static bool CheckImageBorders(const std::vector<uint8_t> &data, int image_width, int image_height) {
    for (int column = 0; column < image_width; ++column) {
      const auto index = static_cast<size_t>(column);
      if (data[index] != 0) {
        return false;
      }
    }

    for (int column = 0; column < image_width; ++column) {
      const int64_t flat_index =
          (static_cast<int64_t>(image_height - 1) * static_cast<int64_t>(image_width)) + static_cast<int64_t>(column);
      const auto index = static_cast<size_t>(flat_index);
      if (data[index] != 0) {
        return false;
      }
    }

    for (int row = 0; row < image_height; ++row) {
      const auto index = static_cast<size_t>(row) * static_cast<size_t>(image_width);
      if (data[index] != 0) {
        return false;
      }
    }

    for (int row = 0; row < image_height; ++row) {
      const auto index =
          (static_cast<size_t>(row) * static_cast<size_t>(image_width)) + static_cast<size_t>(image_width - 1);
      if (data[index] != 0) {
        return false;
      }
    }

    return true;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.empty()) {
      return false;
    }

    if (output_data.size() != input_data_.size()) {
      return false;
    }

    const int image_width = static_cast<int>(std::sqrt(static_cast<double>(input_data_.size())));
    const int image_height = image_width;

    if (!CheckImageBorders(output_data, image_width, image_height)) {
      return false;
    }

    const bool input_all_zeros = CheckAllPixelsZero(input_data_);

    if (input_all_zeros) {
      return CheckAllPixelsZero(output_data);
    }