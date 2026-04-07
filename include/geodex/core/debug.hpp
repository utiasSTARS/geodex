#pragma once

#include <iostream>
#include <string_view>
#include <typeinfo>

#ifdef GEODEX_DEBUG
#define GEODEX_LOG(msg) std::cerr << "[geodex] " << msg << "\n"
#else
#define GEODEX_LOG(msg) ((void)0)
#endif

namespace geodex::detail {

/// @brief Extract a clean type name from compiler intrinsics.
///
/// @details Used by `GEODEX_LOG` sites that want to print template parameter
/// names in a human-readable form (e.g. "SphereRoundMetric" instead of a
/// mangled name). Only active when GEODEX_DEBUG is enabled at the call site.
template <typename T>
constexpr std::string_view type_name() {
#if defined(__clang__) || defined(__GNUC__)
  // __PRETTY_FUNCTION__ looks like:
  //   "std::string_view geodex::detail::type_name() [T = SphereRoundMetric]"
  std::string_view fn = __PRETTY_FUNCTION__;
  auto start = fn.find("T = ");
  if (start == std::string_view::npos) return fn;
  start += 4;
  auto end = fn.rfind(']');
  if (end == std::string_view::npos) return fn.substr(start);
  return fn.substr(start, end - start);
#else
  return typeid(T).name();
#endif
}

}  // namespace geodex::detail
