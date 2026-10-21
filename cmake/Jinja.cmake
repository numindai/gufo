include(FetchContent)
if (POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()

# Import only headers: upstream Minja's CMake also enables its own tests,
# examples and dependency downloads. Nix supplies local, hash-checked archives.
file(READ "${CMAKE_CURRENT_LIST_DIR}/jinja-dependencies.json" jinja_dependencies)
foreach(dependency IN ITEMS MINJA NLOHMANN_JSON)
  string(JSON url GET "${jinja_dependencies}" "${dependency}" url)
  string(JSON hash GET "${jinja_dependencies}" "${dependency}" sha256)
  set(GUFO_${dependency}_URL "${url}" CACHE STRING "${dependency} source archive")
  FetchContent_Declare(${dependency}
    URL "${GUFO_${dependency}_URL}" URL_HASH "SHA256=${hash}"
    SOURCE_SUBDIR gufo-headers-only)
endforeach()
FetchContent_MakeAvailable(MINJA NLOHMANN_JSON)

# Qwen's official template uses "is undefined". Minja has "is defined" but
# omits its inverse. Add that single test using Minja's existing null/undefined
# semantics, in a build-local copy so dependency sources remain untouched.
file(READ "${minja_SOURCE_DIR}/include/minja/minja.hpp" minja_header)
set(defined_test "if (name == \"defined\") return !l.is_null();")
string(FIND "${minja_header}" "${defined_test}" defined_test_offset)
if (defined_test_offset EQUAL -1)
  message(FATAL_ERROR "Minja changed: review the is-undefined compatibility patch")
endif()
string(REPLACE "${defined_test}"
  "${defined_test}\n              if (name == \"undefined\") return l.is_null();"
  minja_header "${minja_header}")
# Jinja returns undefined for a missing list item. NuExtract reads the last
# part of an empty developer example before deciding to skip that example.
set(array_lookup "return array_->at(index < 0 ? array_->size() + index : index);")
string(FIND "${minja_header}" "${array_lookup}" array_lookup_offset)
if (array_lookup_offset EQUAL -1)
  message(FATAL_ERROR "Minja changed: review the missing-list-item compatibility patch")
endif()
string(REPLACE "${array_lookup}"
  "const auto size = static_cast<int64_t>(array_->size());\n      const auto resolved = index < 0 ? size + index : index;\n      return resolved < 0 || resolved >= size ? Value() : array_->at(resolved);"
  minja_header "${minja_header}")
file(CONFIGURE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/jinja-include/minja/minja.hpp"
  CONTENT "${minja_header}" @ONLY)
target_include_directories(gufo_core SYSTEM PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/jinja-include" "${nlohmann_json_SOURCE_DIR}/include")
