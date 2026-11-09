/*
 * File:        fractional_comb.cpp
 * Module:      orc-core
 * Purpose:     NTSC 2D line comb along the picture's own direction, at any
 *              slope
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 decode-orc contributors
 */

#include "fractional_comb.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <map>

namespace ntsc_comb {
namespace {

constexpr double kPi = 3.14159265358979323846;

// Current-line taps, and how many neighbour-line taps each side of the slope:
// floor(d) - 2 .. floor(d) + 3 on the line above, mirrored below.
constexpr int32_t kCurrentTaps[] = {-1, 0, 1};
constexpr int32_t kNeighbourSpan = 3;

// Weight of the taps' squared sum in the fit, keeping noise gain down.
constexpr double kRidge = 0.01;

// Each filter is fitted over slopes d - kRobustSpread .. d + kRobustSpread
// together, so it tolerates the slope estimate's error on a blurred or moving
// edge.
constexpr double kRobustSpread = 0.2;
constexpr int32_t kRobustSteps = 5;

// Half-width of the chroma band fitted, and the sample rate.
// SMPTE 170M-2004 Section 7: chroma bandwidth about 1.3 MHz either side of
// the subcarrier; 4fsc sampling = 4 x 315/88 MHz.
constexpr double kBandHz = 1.3e6;
constexpr double kSampleRateHz = 4.0 * 315e6 / 88.0;
constexpr int32_t kBandSteps = 41;

// Trusting a filter: fully below the first value, not at all above the
// second.
constexpr double kResidualLimits[] = {0.02, 0.06};
constexpr double kNoiseGainLimits[] = {1.2, 2.0};

// Structure tensor window (samples x field lines), and the coherence below
// which the picture has no clear direction (texture, noise) and above which
// it is an edge or a line.
constexpr int32_t kTensorWidth = 9;
constexpr int32_t kTensorLines = 3;
constexpr double kCoherenceLow = 0.7;
constexpr double kCoherenceHigh = 0.9;
// Gradient energy (codes squared) counted as flat.
constexpr double kFlatEnergy = 4.0;

double trust(double value, const double (&limits)[2]) {
  return std::clamp((limits[1] - value) / (limits[1] - limits[0]), 0.0, 1.0);
}

double linspace(double a, double b, int32_t steps, int32_t i) {
  return a + ((b - a) * static_cast<double>(i) / (steps - 1));
}

// Solves (m) x = rhs by Gaussian elimination with partial pivoting; m is n x n
// row-major and is overwritten.
std::vector<double> solve(std::vector<double> m, std::vector<double> rhs) {
  const auto n = static_cast<int32_t>(rhs.size());
  for (int32_t col = 0; col < n; col++) {
    int32_t pivot = col;
    for (int32_t r = col + 1; r < n; r++) {
      if (std::fabs(m[r * n + col]) > std::fabs(m[pivot * n + col])) pivot = r;
    }
    if (pivot != col) {
      for (int32_t k = 0; k < n; k++) {
        std::swap(m[col * n + k], m[pivot * n + k]);
      }
      std::swap(rhs[col], rhs[pivot]);
    }
    for (int32_t r = col + 1; r < n; r++) {
      const double f = m[r * n + col] / m[col * n + col];
      for (int32_t k = col; k < n; k++) m[r * n + k] -= f * m[col * n + k];
      rhs[r] -= f * rhs[col];
    }
  }
  std::vector<double> x(n);
  for (int32_t r = n - 1; r >= 0; r--) {
    double s = rhs[r];
    for (int32_t k = r + 1; k < n; k++) s -= m[r * n + k] * x[k];
    x[r] = s / m[r * n + r];
  }
  return x;
}

// Every slope's filter over the union of their tap positions.
struct DesignTable {
  std::vector<double> slopes;
  std::vector<std::pair<int32_t, int32_t>> positions;
  std::vector<std::vector<double>> taps;  // [design][position]
  std::vector<double> quality;
};

const DesignTable& designTable() {
  static const DesignTable table = [] {
    DesignTable t;
    const auto count = static_cast<int32_t>(std::lround(
        2.0 * FractionalComb::kMaxSlope / FractionalComb::kSlopeStep));
    std::vector<FractionalComb::Design> designs;
    for (int32_t i = 0; i <= count; i++) {
      const double d =
          -FractionalComb::kMaxSlope + (i * FractionalComb::kSlopeStep);
      t.slopes.push_back(d);
      designs.push_back(FractionalComb::design(d));
    }
    std::map<std::pair<int32_t, int32_t>, int32_t> index;
    for (const auto& design : designs) {
      for (const auto& [position, weight] : design.taps) index[position] = 0;
    }
    for (auto& [position, k] : index) {
      k = static_cast<int32_t>(t.positions.size());
      t.positions.push_back(position);
    }
    for (const auto& design : designs) {
      std::vector<double> row(t.positions.size(), 0.0);
      for (const auto& [position, weight] : design.taps) {
        row[index[position]] = weight;
      }
      t.taps.push_back(std::move(row));
      t.quality.push_back(design.quality);
    }
    return t;
  }();
  return table;
}

}  // namespace

FractionalComb::Design FractionalComb::design(double d) {
  const double w0 = kPi / 2.0;
  const double half = 2.0 * kPi * kBandHz / kSampleRateHz;
  const auto base = static_cast<int32_t>(std::floor(d));

  // Tap positions: the current line, then the line above (-1) and below (+1).
  std::vector<std::pair<int32_t, int32_t>> positions;
  for (int32_t i : kCurrentTaps) positions.push_back({0, i});
  for (int32_t j = base - kNeighbourSpan + 1; j <= base + kNeighbourSpan; j++) {
    positions.push_back({-1, j});
  }
  for (int32_t j = base - kNeighbourSpan + 1; j <= base + kNeighbourSpan; j++) {
    positions.push_back({1, -j});
  }
  const auto n = static_cast<int32_t>(positions.size());

  // Least squares with a ridge term: (A'A + ridge I) w = A'b, A's rows being
  // the real and imaginary parts of each constraint, each slope's
  // constraints weighted 1 / kRobustSteps so the ridge keeps the same weight
  // however many slopes are fitted.
  const double scale = 1.0 / std::sqrt(static_cast<double>(kRobustSteps));
  std::vector<std::vector<double>> rows;
  std::vector<double> targets;
  for (int32_t s = 0; s < kRobustSteps; s++) {
    const double slope =
        linspace(d - kRobustSpread, d + kRobustSpread, kRobustSteps, s);
    for (int32_t f = 0; f < kBandSteps; f++) {
      const double omega = linspace(w0 - half, w0 + half, kBandSteps, f);
      std::vector<std::complex<double>> luma, chroma;
      for (const auto& [line, j] : positions) {
        const std::complex<double> tap =
            std::exp(std::complex<double>(0.0, omega * j));
        if (line == 0) {
          luma.push_back(tap);
          chroma.push_back(tap);
        } else {
          // The line above (-1) carries the picture delayed by the slope,
          // the line below advanced by it; chroma is inverted from line to
          // line, its carrier staying on the sample grid.
          const double along = -line * slope;
          luma.push_back(tap *
                         std::exp(std::complex<double>(0.0, -omega * along)));
          chroma.push_back(-tap * std::exp(std::complex<double>(
                                      0.0, -(omega - w0) * along)));
        }
      }
      for (const auto& [row, target] :
           {std::pair{&luma, 0.0}, std::pair{&chroma, 1.0}}) {
        std::vector<double> re(n), im(n);
        for (int32_t k = 0; k < n; k++) {
          re[k] = (*row)[k].real() * scale;
          im[k] = (*row)[k].imag() * scale;
        }
        rows.push_back(std::move(re));
        targets.push_back(target * scale);
        rows.push_back(std::move(im));
        targets.push_back(0.0);
      }
    }
  }

  std::vector<double> normal(static_cast<size_t>(n) * n, 0.0);
  std::vector<double> rhs(n, 0.0);
  for (size_t r = 0; r < rows.size(); r++) {
    for (int32_t i = 0; i < n; i++) {
      rhs[i] += rows[r][i] * targets[r];
      for (int32_t k = 0; k < n; k++) {
        normal[i * n + k] += rows[r][i] * rows[r][k];
      }
    }
  }
  for (int32_t i = 0; i < n; i++) normal[i * n + i] += kRidge;
  const auto weights = solve(normal, rhs);

  double squared = 0.0;
  for (size_t r = 0; r < rows.size(); r++) {
    double fit = 0.0;
    for (int32_t k = 0; k < n; k++) fit += rows[r][k] * weights[k];
    squared += (fit - targets[r]) * (fit - targets[r]);
  }
  double norm = 0.0;
  for (double w : weights) norm += w * w;

  Design result;
  result.slope = d;
  for (int32_t k = 0; k < n; k++) {
    result.taps.push_back({positions[k], weights[k]});
  }
  result.residual =
      std::sqrt(squared / static_cast<double>(rows.size())) / scale;
  result.noiseGain = std::sqrt(norm);
  result.quality = trust(result.residual, kResidualLimits) *
                   trust(result.noiseGain, kNoiseGainLimits);
  return result;
}

Plane FractionalComb::separate(const Plane& frame, const FieldLayout& layout,
                               const ActivePicture& picture) {
  const DesignTable& table = designTable();
  const Plane bp = bandpass(frame);

  // The structure tensor of the chroma-free luma. A 5-tap derivative,
  // (-1, 8, 0, -8, 1) / 12, stays accurate to much higher spatial frequencies
  // than a central difference, whose fall-off would read steep edges as
  // shallower than they are.
  const Plane luma = boxMean(frame, 4, 1, layout.field1Rows);
  auto derivative = [&](int32_t lines, int32_t samples) {
    const Plane p1 = shift(luma, lines, samples);
    const Plane m1 = shift(luma, -lines, -samples);
    const Plane p2 = shift(luma, 2 * lines, 2 * samples);
    const Plane m2 = shift(luma, -2 * lines, -2 * samples);
    Plane out(luma.rows, luma.cols);
    for (size_t i = 0; i < out.v.size(); i++) {
      out.v[i] = static_cast<Sample>(
          ((8.0 * (p1.v[i] - m1.v[i])) - (p2.v[i] - m2.v[i])) / 12.0);
    }
    return out;
  };
  const Plane gx = derivative(0, 1);
  const Plane gy = derivative(1, 0);
  const Plane jxx = boxMean(map(gx, [](double g) { return g * g; }),
                            kTensorWidth, kTensorLines, layout.field1Rows);
  const Plane jxy =
      boxMean(map(gx, gy, [](double a, double b) { return a * b; }),
              kTensorWidth, kTensorLines, layout.field1Rows);
  const Plane jyy = boxMean(map(gy, [](double g) { return g * g; }),
                            kTensorWidth, kTensorLines, layout.field1Rows);

  const auto designs = static_cast<int32_t>(table.slopes.size());
  Plane chroma(frame.rows, frame.cols);
  for (int32_t r = 0; r < frame.rows; r++) {
    const int32_t above = (r - 1 + frame.rows) % frame.rows;
    const int32_t below = (r + 1) % frame.rows;
    const bool usable =
        r >= 1 && r < frame.rows - 1 && picture.containsRow(r) &&
        picture.containsRow(r - 1) && picture.containsRow(r + 1) &&
        layout.sameField(r, r - 1) && layout.sameField(r, r + 1);
    for (int32_t c = 0; c < frame.cols; c++) {
      if (!usable) {
        chroma.set(r, c, bp.at(r, c));
        continue;
      }
      const double xx = jxx.at(r, c), xy = jxy.at(r, c), yy = jyy.at(r, c);
      const double trace = xx + yy + kFlatEnergy;
      const double a = xx - yy;
      const double b = 2.0 * xy;
      const double spread = std::sqrt((a * a) + (b * b));
      const double coherence = spread / trace;
      // The slope is tan(atan2(b, a) / 2), by the half-angle identity
      // b / (spread + a), or (spread - a) / b, the same value without the
      // cancellation, where a < 0. With no gradient (a = b = 0) it is NaN:
      // no direction, so none taken.
      double slope = (a >= 0.0) ? b / (spread + a) : (spread - a) / b;
      if (std::isnan(slope)) slope = 0.0;

      // The two nearest filters and the weight between them.
      const double position =
          (std::clamp(slope, -kMaxSlope, kMaxSlope) + kMaxSlope) / kSlopeStep;
      const int32_t lower = std::clamp(
          static_cast<int32_t>(std::floor(position)), 0, designs - 2);
      const double frac = std::clamp(position - lower, 0.0, 1.0);
      const double quality = ((1.0 - frac) * table.quality[lower]) +
                             (frac * table.quality[lower + 1]);
      double gate = std::clamp((coherence - kCoherenceLow) /
                                   (kCoherenceHigh - kCoherenceLow),
                               0.0, 1.0) *
                    quality;
      if (std::fabs(slope) > kMaxSlope) gate = 0.0;

      const double base =
          (0.5 * bp.at(r, c)) - (0.25 * (bp.at(above, c) + bp.at(below, c)));
      double directional = 0.0;
      if (gate > 0.0) {
        const auto& lo = table.taps[lower];
        const auto& hi = table.taps[lower + 1];
        for (size_t k = 0; k < table.positions.size(); k++) {
          const double weight = ((1.0 - frac) * lo[k]) + (frac * hi[k]);
          if (weight == 0.0) continue;
          const auto [line, sample] = table.positions[k];
          const int32_t row = (r + line + frame.rows) % frame.rows;
          const int32_t col = std::clamp(c + sample, 0, frame.cols - 1);
          directional += weight * bp.at(row, col);
        }
      }
      chroma.set(r, c, (gate * directional) + ((1.0 - gate) * base));
    }
  }
  return chroma;
}

}  // namespace ntsc_comb
