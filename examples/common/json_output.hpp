/// @file json_output.hpp
/// @brief Minimal JSON writer for the documentation examples.
///
/// @details Each example writes the numbers it prints as one JSON object. A test compares
/// the C++ and Python versions of an example value by value. The writer prints doubles
/// with 17 significant digits, which round-trip exactly.

#pragma once

#include <cmath>
#include <cstring>

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace geodex_examples {

/// @brief A JSON value built from numbers, booleans, strings, arrays and objects.
class Json {
 public:
  Json() : text_("null") {}
  Json(double v) : text_(number(v)) {}
  Json(int v) : text_(std::to_string(v)) {}
  Json(long v) : text_(std::to_string(v)) {}
  Json(std::size_t v) : text_(std::to_string(v)) {}
  Json(bool v) : text_(v ? "true" : "false") {}
  Json(const char* v) : text_(quote(v)) {}
  Json(const std::string& v) : text_(quote(v)) {}

  /// @brief A vector as a JSON array of numbers.
  template <typename Derived>
  Json(const Eigen::MatrixBase<Derived>& v) {
    std::vector<Json> items;
    for (Eigen::Index i = 0; i < v.size(); ++i) items.emplace_back(static_cast<double>(v(i)));
    text_ = array(items).text_;
  }

  /// @brief A list of vectors (a path) as a JSON array of arrays.
  template <typename Point>
  static Json path(const std::vector<Point>& points) {
    std::vector<Json> items;
    for (const auto& p : points) items.emplace_back(p);
    return array(items);
  }

  static Json array(const std::vector<Json>& items) {
    Json out;
    out.text_ = "[";
    for (std::size_t i = 0; i < items.size(); ++i) out.text_ += (i ? ", " : "") + items[i].text_;
    out.text_ += "]";
    return out;
  }

  static Json object(const std::vector<std::pair<std::string, Json>>& fields) {
    Json out;
    out.text_ = "{";
    for (std::size_t i = 0; i < fields.size(); ++i) {
      out.text_ += (i ? ", " : "") + quote(fields[i].first) + ": " + fields[i].second.text_;
    }
    out.text_ += "}";
    return out;
  }

  const std::string& str() const { return text_; }

 private:
  static std::string number(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream os;
    os << std::setprecision(std::numeric_limits<double>::max_digits10) << v;
    return os.str();
  }

  static std::string quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) {
      if (c == '"' || c == '\\') out += '\\';
      out += c;
    }
    return out + "\"";
  }

  std::string text_;
};

/// @brief Write `value` to the path after `--json` on the command line, when given.
inline void write_json_arg(int argc, char** argv, const Json& value) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (std::strcmp(argv[i], "--json") == 0) {
      std::ofstream out(argv[i + 1]);
      if (!out) throw std::runtime_error(std::string("cannot write ") + argv[i + 1]);
      out << value.str() << "\n";
    }
  }
}

}  // namespace geodex_examples
