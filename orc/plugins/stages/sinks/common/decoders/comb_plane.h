/*
 * File:        comb_plane.h
 * Module:      orc-core
 * Purpose:     Whole-frame sample planes and the filters the NTSC 3D comb
 *              builds its measurements from
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 decode-orc contributors
 */

#ifndef COMB_PLANE_H
#define COMB_PLANE_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <utility>
#include <vector>

namespace ntsc_comb {

// The frame's geometry in field-separated ("flat") rows: the first field's
// lines, top to bottom, then the second field's. A row's neighbours in the
// same field are the rows next to it, so a filter running down a field moves
// one row at a time.
struct FieldLayout {
  int32_t rows = 0;        // all lines of the frame
  int32_t cols = 0;        // samples per line
  int32_t field1Rows = 0;  // lines of the first field

  // The flat row holding interlaced frame line |frameLine|.
  int32_t rowOf(int32_t frameLine) const {
    return (frameLine / 2) + ((frameLine % 2 == 0) ? 0 : field1Rows);
  }
  bool sameField(int32_t a, int32_t b) const {
    return (a < field1Rows) == (b < field1Rows);
  }
};

// The active picture, in flat rows of each field and samples of each line.
struct ActivePicture {
  int32_t field1First = 0, field1End = 0;
  int32_t field2First = 0, field2End = 0;
  int32_t sampleStart = 0, sampleEnd = 0;

  bool containsRow(int32_t row) const {
    return (row >= field1First && row < field1End) ||
           (row >= field2First && row < field2End);
  }
  bool contains(int32_t row, int32_t col) const {
    return containsRow(row) && col >= sampleStart && col < sampleEnd;
  }

  // The picture between interlaced frame lines [first, last) and samples
  // [start, end) of a frame laid out as |layout|.
  static ActivePicture fromFrameLines(const FieldLayout& layout, int32_t first,
                                      int32_t last, int32_t start,
                                      int32_t end) {
    ActivePicture p;
    p.field1First = first / 2;
    p.field1End = std::min(layout.field1Rows, (last + 1) / 2);
    p.field2First = layout.field1Rows + (first / 2);
    p.field2End =
        layout.field1Rows + std::min(layout.rows - layout.field1Rows, last / 2);
    p.sampleStart = start;
    p.sampleEnd = end;
    return p;
  }
};

// The comb's planes store single precision: they are streamed through dozens
// of passes per frame, and memory bandwidth, not arithmetic, bounds the
// comb. Sums and filters accumulate in double.
using Sample = float;

namespace plane_storage {
// The comb builds and drops dozens of whole-frame planes for every frame;
// fresh allocations that size are mapped, page-faulted and zero-filled each
// time, which costs more than the arithmetic. Planes therefore hand their
// storage back to a store for the thread on destruction and take it from
// there again. At most kKept buffers are kept per thread.
constexpr size_t kKept = 48;

inline std::vector<std::vector<Sample>>& store() {
  thread_local std::vector<std::vector<Sample>> buffers;
  return buffers;
}

// A buffer of |size| samples, contents unspecified.
inline std::vector<Sample> take(size_t size) {
  auto& buffers = store();
  for (auto it = buffers.rbegin(); it != buffers.rend(); ++it) {
    if (it->size() == size) {
      std::vector<Sample> buffer = std::move(*it);
      buffers.erase(std::next(it).base());
      return buffer;
    }
  }
  return std::vector<Sample>(size);
}

inline void give(std::vector<Sample>&& buffer) {
  if (buffer.empty()) return;
  auto& buffers = store();
  if (buffers.size() < kKept) buffers.push_back(std::move(buffer));
}
}  // namespace plane_storage

// A whole frame of samples in flat rows; a plane holds rows * cols of them.
// Storage is recycled within the thread (see plane_storage), so a plane made
// with Plane(rows, cols) has unspecified contents: every sample must be
// written.
struct Plane {
  int32_t rows = 0;
  int32_t cols = 0;
  std::vector<Sample> v;

  Plane() = default;
  Plane(int32_t r, int32_t c)
      : rows(r), cols(c), v(plane_storage::take(static_cast<size_t>(r) * c)) {}
  Plane(int32_t r, int32_t c, double fill) : Plane(r, c) {
    std::fill(v.begin(), v.end(), static_cast<Sample>(fill));
  }
  Plane(const Plane& other) : Plane(other.rows, other.cols) {
    std::copy(other.v.begin(), other.v.end(), v.begin());
  }
  Plane(Plane&& other) noexcept
      : rows(other.rows), cols(other.cols), v(std::move(other.v)) {
    other.rows = other.cols = 0;
  }
  Plane& operator=(const Plane& other) {
    if (this != &other) {
      if (v.size() != other.v.size()) {
        plane_storage::give(std::move(v));
        v = plane_storage::take(other.v.size());
      }
      rows = other.rows;
      cols = other.cols;
      std::copy(other.v.begin(), other.v.end(), v.begin());
    }
    return *this;
  }
  Plane& operator=(Plane&& other) noexcept {
    if (this != &other) {
      plane_storage::give(std::move(v));
      rows = other.rows;
      cols = other.cols;
      v = std::move(other.v);
      other.rows = other.cols = 0;
    }
    return *this;
  }
  ~Plane() { plane_storage::give(std::move(v)); }

  double at(int32_t r, int32_t c) const {
    return v[static_cast<size_t>(r) * cols + c];
  }
  void set(int32_t r, int32_t c, double value) {
    v[static_cast<size_t>(r) * cols + c] = static_cast<Sample>(value);
  }
  Sample* row(int32_t r) { return v.data() + static_cast<size_t>(r) * cols; }
  const Sample* row(int32_t r) const {
    return v.data() + static_cast<size_t>(r) * cols;
  }
  bool empty() const { return v.empty(); }
};

// out = f(a) sample by sample.
template <typename F>
Plane map(const Plane& a, F f) {
  Plane out(a.rows, a.cols);
  for (size_t i = 0; i < a.v.size(); i++) {
    out.v[i] = static_cast<Sample>(f(static_cast<double>(a.v[i])));
  }
  return out;
}

// out = f(a, b) sample by sample.
template <typename F>
Plane map(const Plane& a, const Plane& b, F f) {
  Plane out(a.rows, a.cols);
  for (size_t i = 0; i < a.v.size(); i++) {
    out.v[i] = static_cast<Sample>(
        f(static_cast<double>(a.v[i]), static_cast<double>(b.v[i])));
  }
  return out;
}

// The 1D chroma bandpass [-1/4, 0, 1/2, 0, -1/4] along every line: unity gain
// at the subcarrier (a quarter of the 4fsc rate), none at DC. The two samples
// at each end of a line are left at zero.
inline Plane bandpass(const Plane& x) {
  Plane out(x.rows, x.cols);
  for (int32_t r = 0; r < x.rows; r++) {
    const Sample* in = x.row(r);
    Sample* o = out.row(r);
    for (int32_t c = 0; c < std::min(2, x.cols); c++) o[c] = 0.0F;
    for (int32_t c = std::max(2, x.cols - 2); c < x.cols; c++) o[c] = 0.0F;
    for (int32_t c = 2; c < x.cols - 2; c++) {
      const double value =
          (static_cast<double>(in[c]) -
           ((static_cast<double>(in[c - 2]) + static_cast<double>(in[c + 2])) /
            2.0)) /
          2.0;
      o[c] = static_cast<Sample>(value);
    }
  }
  return out;
}

// x moved so out[r][c] = x[r + lines][c + samples]: rows wrap around the
// frame, samples repeat the line's end. lines = -1 is the line above in the
// same field (inside the field).
inline Plane shift(const Plane& x, int32_t lines, int32_t samples) {
  Plane out(x.rows, x.cols);
  for (int32_t r = 0; r < x.rows; r++) {
    const int32_t src = (((r + lines) % x.rows) + x.rows) % x.rows;
    const Sample* in = x.row(src);
    Sample* o = out.row(r);
    for (int32_t c = 0; c < x.cols; c++) {
      o[c] = in[std::clamp(c + samples, 0, x.cols - 1)];
    }
  }
  return out;
}

// The mean of each sample's neighbourhood: |width| samples along the line by
// |lines| lines of the same field (fields are never mixed), the edges
// extended. Even widths take one more sample before than after. A width of 4
// has a null at the subcarrier (and twice it), so boxMean(frame, 4, 1) is the
// composite with its chroma removed.
inline Plane boxMean(const Plane& x, int32_t width, int32_t lines,
                     int32_t field1Rows) {
  Plane across(x.rows, x.cols);
  if (width <= 1) {
    across = x;
  } else {
    // A running sum along each line: sample c's window is
    // [c - before, c - before + width), indices clamped to the line.
    const int32_t before = width / 2;
    const int32_t last = x.cols - 1;
    const auto count = static_cast<double>(width);
    for (int32_t r = 0; r < x.rows; r++) {
      const Sample* in = x.row(r);
      Sample* o = across.row(r);
      double total = 0.0;
      for (int32_t k = -before; k < width - before; k++) {
        total += static_cast<double>(in[std::clamp(k, 0, last)]);
      }
      o[0] = static_cast<Sample>(total / count);
      for (int32_t c = 1; c < x.cols; c++) {
        total += static_cast<double>(
                     in[std::clamp(c - before + width - 1, 0, last)]) -
                 static_cast<double>(in[std::clamp(c - 1 - before, 0, last)]);
        o[c] = static_cast<Sample>(total / count);
      }
    }
  }
  if (lines <= 1) return across;

  Plane out(x.rows, x.cols);
  const int32_t before = lines / 2;
  const auto count = static_cast<double>(lines);
  std::vector<double> sum(static_cast<size_t>(x.cols));
  auto field = [&](int32_t first, int32_t end) {
    for (int32_t r = first; r < end; r++) {
      std::fill(sum.begin(), sum.end(), 0.0);
      for (int32_t k = r - before; k < r - before + lines; k++) {
        const Sample* in = across.row(std::clamp(k, first, end - 1));
        for (int32_t c = 0; c < x.cols; c++) {
          sum[c] += static_cast<double>(in[c]);
        }
      }
      Sample* o = out.row(r);
      for (int32_t c = 0; c < x.cols; c++) {
        o[c] = static_cast<Sample>(sum[c] / count);
      }
    }
  };
  field(0, field1Rows);
  field(field1Rows, x.rows);
  return out;
}

// The local mean of |x| for a chroma-band signal, over whole subcarrier
// cycles (4 samples) by 3 lines: |x| alternates between the I and Q samples,
// so a box of odd width would ripple from one sample to the next, and so
// would every weight taken from it.
inline Plane cycleMean(const Plane& x, int32_t field1Rows) {
  return boxMean(map(x, [](double s) { return std::fabs(s); }), 4, 3,
                 field1Rows);
}

// The |q|th percentile of |values| (0..100), interpolating linearly between
// the two nearest ranks; values is reordered. 0 for no values.
inline double percentile(std::vector<double>& values, double q) {
  if (values.empty()) return 0.0;
  const double position = (q / 100.0) * static_cast<double>(values.size() - 1);
  const auto lower = static_cast<size_t>(std::floor(position));
  const double fraction = position - static_cast<double>(lower);
  std::nth_element(values.begin(), values.begin() + lower, values.end());
  const double a = values[lower];
  if (fraction == 0.0 || lower + 1 >= values.size()) return a;
  const double b = *std::min_element(values.begin() + lower + 1, values.end());
  return a + (fraction * (b - a));
}

}  // namespace ntsc_comb

#endif  // COMB_PLANE_H
