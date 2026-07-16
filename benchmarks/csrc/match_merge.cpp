#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

py::tuple match(py::array_t<int64_t> chunks0, py::array_t<int64_t> chunks1,
                int chunk_size, int mismatch_thres) {
  auto a = chunks0.unchecked<1>();
  auto b = chunks1.unchecked<1>();
  int len0 = a.shape(0), len1 = b.shape(0);

  int match_a = 0, match_b = 0, match_size = 0;
  int i0 = len0 - 1, current = 0;

  for (int i1 = len1 - 1; i1 >= 0; i1--) {
    while (i0 >= 0 && a(i0) > b(i1) + chunk_size) {
      i0--;
      current = 0;
    }
    if (i0 >= 0 && a(i0) == b(i1) + chunk_size) {
      if (current > match_size) {
        match_a = i0;
        match_b = i1;
        match_size = current;
      }
      current++;
      i0--;
    } else {
      current = 0;
    }
  }

  if (match_size <= mismatch_thres) {
    return py::make_tuple(-1, -1);
  }
  return py::make_tuple(len0 - (match_a + match_size - 1),
                        len1 - (match_b + match_size - 1));
}

py::array_t<int64_t> merge(py::list chunks, py::list matches) {
  if (matches.size() != chunks.size() * 2) {
    throw std::runtime_error("merge: matches.size() != chunks.size() * 2");
  }

  std::vector<int64_t> result;
  for (int i = 0, n = chunks.size(); i < n; i++) {
    auto chunk = chunks[i].cast<py::array_t<int64_t>>();
    auto buf = chunk.unchecked<1>();
    int l = matches[i * 2].cast<int>();
    int r = matches[i * 2 + 1].cast<int>();
    int start = buf.shape(0) - l;
    int end = buf.shape(0) - r;
    for (int j = start; j < end; j++) {
      result.push_back(buf(j));
    }
  }

  // Build an owned numpy array by copying into allocated memory
  py::array_t<int64_t> arr(py::ssize_t(result.size()));
  std::copy(result.begin(), result.end(), arr.mutable_data());
  return arr;
}

PYBIND11_MODULE(lopt_cpp, m) {
  m.doc() = "LoPT C++ extension: token match and merge";
  m.def("match", &match,
        "Match overlapping regions between two token sequences",
        py::arg("chunks0"), py::arg("chunks1"), py::arg("chunk_size"),
        py::arg("mismatch_thres"));
  m.def("merge", &merge, "Merge token chunks using match indices",
        py::arg("chunks"), py::arg("matches"));
}
