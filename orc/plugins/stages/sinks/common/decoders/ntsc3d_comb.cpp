/*
 * File:        ntsc3d_comb.cpp
 * Module:      orc-core
 * Purpose:     NTSC motion-adaptive 3D comb: frame comb where the picture is
 *              still and the 2D comb would get it wrong, 2D comb elsewhere
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2026 decode-orc contributors
 */

#include "ntsc3d_comb.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <vector>

#include "fractional_comb.h"

namespace ntsc_comb {
namespace {

// Where both neighbours are still, allow this much of the chroma expected to
// leak through the 4-sample mean at colour edges.
constexpr double kLeakK = 1.0;
// Steady motion: the frame differs from the mean of its neighbours by up to
// this many times the noise floor, about equally towards each.
constexpr double kMiddleK = 3.0;
// The two one-sided frame combs agree if within this many times their noise
// floor, plus this fraction of the local chroma-band amplitude (under steady
// motion they err in opposite directions, so their average stays good while
// they disagree; 0.6 allows about half a sample a frame at the subcarrier).
constexpr double kNoiseK = 2.0;
constexpr double kDetailK = 0.6;
// A one-sided frame comb needs the frame beyond to match in the chroma band
// within this many times its noise; the averaged comb, the previous and next
// frames within kPairK.
constexpr double kSideK = 2.0;
constexpr double kPairK = 3.0;
// A flash: the frames two away match each other, and the current frame is
// over this many times as far from the nearer of them.
constexpr double kFlashK = 3.0;
// Where needed, the frame comb is kept only where the current frame matches a
// frame two away within this many times as well as those two match each
// other.
constexpr double kConfirmK = 1.5;
// The weights are averaged over this many samples along the line.
constexpr int32_t kMaskWidth = 9;
// Blend between a threshold T (still) and kSoft x T (moving).
constexpr double kSoft = 1.5;

// The noise floor: the quietest frame pair's kNoisePercentile over the
// picture, held between kPictureNoiseMin and kPictureNoiseMax times the
// disc's, measured the same way on the sync tip (which carries neither
// picture nor chroma, so motion, cuts and flashes cannot touch it). Where
// every pair moves the picture's percentile measures motion, not noise; a
// dark, clipped or noise-reduced picture can read quieter than the disc.
constexpr double kNoisePercentile = 25.0;
constexpr double kPictureNoiseMin = 0.5;
constexpr double kPictureNoiseMax = 2.0;
// SMPTE 170M-2004 Section 8: horizontal sync is 4.7 us (about 67 samples at
// 4fsc) from the line's start; these samples sit inside its tip.
constexpr int32_t kSyncTipStart = 10;
constexpr int32_t kSyncTipEnd = 55;
// SMPTE 170M-2004 Section 6.4: the colour burst; its middle, clear of the
// envelope's rise and fall, carries constant chroma that motion cannot
// touch, against which the chroma band's noise slope is measured.
constexpr int32_t kBurstStart = 80;
constexpr int32_t kBurstEnd = 104;
// Chroma-band amplitude (codes) the picture's noise slope is measured over.
constexpr double kChromaStrong = 40.0;
// The picture's slope needs at least this many strong samples.
constexpr size_t kMinStrongSamples = 1000;

// 1 below the threshold, 0 above kSoft times it, linear between.
double stillWeight(double level, double threshold) {
  threshold = std::max(threshold, 1e-6);
  return std::clamp((kSoft * threshold - level) / ((kSoft - 1.0) * threshold),
                    0.0, 1.0);
}

// Measurements over regions of the frame.
class Regions {
 public:
  Regions(const FieldLayout& layout, const ActivePicture& picture)
      : layout_(layout), picture_(picture) {
    for (int32_t r = 0; r < layout.rows; r++) {
      if (picture.containsRow(r)) pictureRows_.push_back(r);
    }
    hasSyncTip_ = layout.cols >= kSyncTipEnd;
    hasBurst_ = layout.cols >= kBurstEnd;
  }

  bool hasSyncTip() const { return hasSyncTip_ && !pictureRows_.empty(); }
  bool hasBurst() const { return hasBurst_ && !pictureRows_.empty(); }

  // Percentile of |p| over the picture's lines, samples [start, end).
  double columns(const Plane& p, int32_t start, int32_t end, double q) const {
    std::vector<double> values;
    values.reserve(pictureRows_.size() * static_cast<size_t>(end - start));
    for (int32_t r : pictureRows_) {
      const Sample* row = p.row(r);
      values.insert(values.end(), row + start, row + end);
    }
    return percentile(values, q);
  }
  double syncTip(const Plane& p, double q) const {
    return columns(p, kSyncTipStart, kSyncTipEnd, q);
  }
  double burst(const Plane& p, double q) const {
    return columns(p, kBurstStart, kBurstEnd, q);
  }
  double burstMean(const Plane& p) const {
    double total = 0.0;
    for (int32_t r : pictureRows_) {
      for (int32_t c = kBurstStart; c < kBurstEnd; c++) total += p.at(r, c);
    }
    return total /
           static_cast<double>(pictureRows_.size() * (kBurstEnd - kBurstStart));
  }

  // Percentile of |p| over the picture, or over the picture samples where
  // |mask| holds.
  double picture(const Plane& p, double q,
                 const std::vector<bool>* mask = nullptr) const {
    std::vector<double> values;
    for (int32_t r : pictureRows_) {
      for (int32_t c = picture_.sampleStart; c < picture_.sampleEnd; c++) {
        if (mask != nullptr && !(*mask)[index(r, c)]) continue;
        values.push_back(p.at(r, c));
      }
    }
    return percentile(values, q);
  }

  size_t index(int32_t r, int32_t c) const {
    return static_cast<size_t>(r) * layout_.cols + c;
  }
  const std::vector<int32_t>& pictureRows() const { return pictureRows_; }
  const ActivePicture& active() const { return picture_; }
  size_t pictureSamples() const {
    return pictureRows_.size() *
           static_cast<size_t>(picture_.sampleEnd - picture_.sampleStart);
  }

 private:
  const FieldLayout& layout_;
  const ActivePicture& picture_;
  std::vector<int32_t> pictureRows_;
  bool hasSyncTip_ = false;
  bool hasBurst_ = false;
};

// The floor for a change measure, from its planes for each frame pair: the
// quietest pair's percentile over the picture, held to kPictureNoiseMin..
// kPictureNoiseMax times the disc's (the same measure and percentile on the
// sync tip, averaged over the pairs).
double noiseFloor(const std::vector<const Plane*>& changes,
                  const Regions& regions) {
  double pictureNoise = 0.0;
  bool first = true;
  for (const Plane* c : changes) {
    const double value = regions.picture(*c, kNoisePercentile);
    pictureNoise = first ? value : std::min(pictureNoise, value);
    first = false;
  }
  if (!regions.hasSyncTip()) return pictureNoise;
  double disc = 0.0;
  for (const Plane* c : changes) disc += regions.syncTip(*c, kNoisePercentile);
  disc /= static_cast<double>(changes.size());
  return std::clamp(pictureNoise, kPictureNoiseMin * disc,
                    kPictureNoiseMax * disc);
}

// How much a change measure's noise rises per code of chroma-band amplitude,
// from its planes for frame pairs of the same chroma phase: on the picture,
// the quietest pair's percentile of (change - floor) / amplitude where the
// chroma band is strong; on the disc, the burst's change over the sync tip's,
// per code of burst amplitude.
double chromaNoiseSlope(const std::vector<const Plane*>& changes, double floor,
                        const Plane& amplitude, const Regions& regions) {
  if (changes.empty()) return 0.0;
  std::optional<double> disc;
  if (regions.hasBurst() && regions.hasSyncTip()) {
    double sum = 0.0;
    for (const Plane* c : changes) {
      sum += regions.burst(*c, kNoisePercentile) -
             regions.syncTip(*c, kNoisePercentile);
    }
    disc = std::max(sum / static_cast<double>(changes.size()), 0.0) /
           std::max(regions.burstMean(amplitude), 1.0);
  }

  std::vector<size_t> strong;
  const ActivePicture& picture = regions.active();
  for (int32_t r : regions.pictureRows()) {
    for (int32_t c = picture.sampleStart; c < picture.sampleEnd; c++) {
      if (amplitude.at(r, c) > kChromaStrong) {
        strong.push_back(regions.index(r, c));
      }
    }
  }
  if (strong.size() < kMinStrongSamples) return disc.value_or(0.0);

  double pictureSlope = 0.0;
  bool first = true;
  std::vector<double> values(strong.size());
  for (const Plane* c : changes) {
    for (size_t i = 0; i < strong.size(); i++) {
      values[i] =
          std::max(c->v[strong[i]] - floor, 0.0) / amplitude.v[strong[i]];
    }
    const double value = percentile(values, kNoisePercentile);
    pictureSlope = first ? value : std::min(pictureSlope, value);
    first = false;
  }
  if (!disc) return pictureSlope;
  return std::clamp(pictureSlope, kPictureNoiseMin * *disc,
                    kPictureNoiseMax * *disc);
}

}  // namespace

Ntsc3dComb::Result Ntsc3dComb::separate(const Frames& frames,
                                        const FieldLayout& layout,
                                        const ActivePicture& picture,
                                        const Settings& settings) {
  const Plane& current = *frames.current;
  const Plane* before = frames.before;
  const Plane* after = frames.after;
  const Plane* before2 = frames.before2;
  const Plane* after2 = frames.after2;
  const int32_t f1 = layout.field1Rows;
  const Regions regions(layout, picture);
  const double lumaK = settings.motionThreshold;

  // Change measures between two frames.
  // Low-frequency change: a 4-sample mean has a null at the subcarrier, so a
  // still picture's chroma mostly drops out.
  // (Each measure takes its difference and filter in one pass, then the
  // box mean.)
  auto lumaChange = [&](const Plane& a, const Plane& b) {
    Plane low(a.rows, a.cols);
    std::vector<double> d(static_cast<size_t>(a.cols));
    const int32_t last = a.cols - 1;
    for (int32_t r = 0; r < a.rows; r++) {
      const Sample* pa = a.row(r);
      const Sample* pb = b.row(r);
      for (int32_t c = 0; c < a.cols; c++) d[c] = pa[c] - pb[c];
      // |4-sample mean|, window [c - 2, c + 2), as boxMean(d, 4, 1) has it
      Sample* o = low.row(r);
      for (int32_t c = 0; c < a.cols; c++) {
        const double total = d[std::clamp(c - 2, 0, last)] +
                             d[std::clamp(c - 1, 0, last)] + d[c] +
                             d[std::clamp(c + 1, 0, last)];
        o[c] = static_cast<Sample>(std::fabs(total / 4.0));
      }
    }
    return boxMean(low, 5, 3, f1);
  };
  // Change in the chroma band: chroma-free only between frames of the same
  // chroma phase (two apart). Sees the fine, thin detail the 4-sample mean
  // blurs away.
  auto detailChange = [&](const Plane& a, const Plane& b) {
    Plane band(a.rows, a.cols);
    for (int32_t r = 0; r < a.rows; r++) {
      const Sample* pa = a.row(r);
      const Sample* pb = b.row(r);
      Sample* o = band.row(r);
      for (int32_t c = 0; c < a.cols; c++) {
        if (c < 2 || c >= a.cols - 2) {
          o[c] = static_cast<Sample>(0.0);
          continue;
        }
        const double d = pa[c] - pb[c];
        const double side =
            ((pa[c - 2] - pb[c - 2]) + (pa[c + 2] - pb[c + 2])) / 2.0;
        o[c] = static_cast<Sample>(std::fabs((d - side) / 2.0));
      }
    }
    return boxMean(band, 4, 3, f1);
  };
  // Change over the whole signal: chroma-free only between frames of the
  // same chroma phase.
  auto fullChange = [&](const Plane& a, const Plane& b) {
    return boxMean(
        map(a, b, [](double x, double y) { return std::fabs(x - y); }), 4, 3,
        f1);
  };
  auto mean = [](const Plane& a, const Plane& b) {
    return map(a, b, [](double x, double y) { return (x + y) / 2.0; });
  };

  Result result;
  const Plane line = FractionalComb::separate(current, layout, picture);
  const Plane now = bandpass(current);
  const size_t n = now.v.size();

  // Each neighbour's frame comb, and how much the low-frequency luma changed
  // from it.
  // Planes computed only where their frames exist are held here; the pointers
  // below are null where they do not.
  Plane fromBeforePlane, fromAfterPlane, lumaBeforePlane, lumaAfterPlane;
  const Plane* fromBefore = nullptr;
  const Plane* fromAfter = nullptr;
  const Plane* lumaBefore = nullptr;
  const Plane* lumaAfter = nullptr;
  auto frameComb = [&](const Plane& other) {
    return map(now, bandpass(other),
               [](double x, double y) { return (x - y) / 2.0; });
  };
  if (before != nullptr) {
    fromBeforePlane = frameComb(*before);
    lumaBeforePlane = lumaChange(current, *before);
    fromBefore = &fromBeforePlane;
    lumaBefore = &lumaBeforePlane;
  }
  if (after != nullptr) {
    fromAfterPlane = frameComb(*after);
    lumaAfterPlane = lumaChange(current, *after);
    fromAfter = &fromAfterPlane;
    lumaAfter = &lumaAfterPlane;
  }
  if (fromBefore == nullptr && fromAfter == nullptr) {
    result.chroma = line;
    result.frameCombWeight = Plane(now.rows, now.cols, 0.0);
    return result;
  }

  // Low-frequency changes between the frame pairs the floor is taken from.
  // A change measure between two frames, or null where either is missing.
  auto changeIf = [](const Plane* a, const Plane* b, Plane& storage,
                     const auto& measure) -> const Plane* {
    if (a == nullptr || b == nullptr) return nullptr;
    storage = measure(*a, *b);
    return &storage;
  };
  auto present = [](std::initializer_list<const Plane*> planes) {
    std::vector<const Plane*> out;
    for (const Plane* p : planes) {
      if (p != nullptr) out.push_back(p);
    }
    return out;
  };
  Plane lumaBAPlane, lumaB2Plane, lumaA2Plane;
  const Plane* lumaBA = changeIf(before, after, lumaBAPlane, lumaChange);
  const Plane* lumaB2 = changeIf(&current, before2, lumaB2Plane, lumaChange);
  const Plane* lumaA2 = changeIf(&current, after2, lumaA2Plane, lumaChange);
  const double baseFloor = noiseFloor(
      present({lumaBefore, lumaAfter, lumaBA, lumaB2, lumaA2}), regions);

  // Noise in the chroma band rises with its amplitude, so each floor does
  // too, from frame pairs of the same chroma phase.
  const Plane amplitude = cycleMean(now, f1);
  const auto samePhaseLuma = present({lumaB2, lumaA2, lumaBA});
  const double lumaSlope =
      chromaNoiseSlope(samePhaseLuma, baseFloor, amplitude, regions);
  auto floorAt = [&](size_t i) {
    return baseFloor + (amplitude.v[i] * lumaSlope);
  };

  Plane detailB2Plane, detailA2Plane, detailBAPlane, detailTwosPlane;
  const Plane* detailB2 =
      changeIf(&current, before2, detailB2Plane, detailChange);
  const Plane* detailA2 =
      changeIf(&current, after2, detailA2Plane, detailChange);
  const Plane* detailBA = changeIf(before, after, detailBAPlane, detailChange);
  // The frames two away against each other: how far apart they are in the
  // chroma band (noise, on a still picture).
  const Plane* detailTwos =
      changeIf(before2, after2, detailTwosPlane, detailChange);
  const auto samePhaseDetail = present({detailB2, detailA2, detailBA});
  double detailFloor = 0.0, detailSlope = 0.0;
  if (!samePhaseDetail.empty()) {
    detailFloor = noiseFloor(samePhaseDetail, regions);
    detailSlope =
        chromaNoiseSlope(samePhaseDetail, detailFloor, amplitude, regions);
  }
  auto detailFloorAt = [&](size_t i) {
    return detailFloor + (amplitude.v[i] * detailSlope);
  };

  // Colour edges: a still neighbour's chroma is the current frame's,
  // inverted, so comparing the two leaves twice the chroma, and the 4-sample
  // mean only nulls it at the subcarrier itself; a sharp colour edge's
  // sidebands get through and read as luma change. Where the previous and
  // next frames match, and the current frame matches one two away (so it is
  // not a flash), the picture is still and the thresholds allow for the leak
  // the current frame's own chroma (3-line comb estimate) would cause.
  Plane allowance(now.rows, now.cols, 0.0);
  if (fromBefore != nullptr && fromAfter != nullptr && kLeakK > 0) {
    Plane lineComb = now;
    for (int32_t r = 1; r < now.rows - 1; r++) {
      if (!(picture.containsRow(r) && picture.containsRow(r - 1) &&
            picture.containsRow(r + 1) && layout.sameField(r, r - 1) &&
            layout.sameField(r, r + 1))) {
        continue;
      }
      for (int32_t c = 0; c < now.cols; c++) {
        lineComb.set(
            r, c,
            ((2.0 * now.at(r, c)) - now.at(r - 1, c) - now.at(r + 1, c)) / 4.0);
      }
    }
    const Plane leak = boxMean(
        map(boxMean(map(lineComb, [](double x) { return 2.0 * x; }), 4, 1, f1),
            [](double x) { return std::fabs(x); }),
        5, 3, f1);
    for (size_t i = 0; i < n; i++) {
      double same = stillWeight(lumaBA->v[i], lumaK * floorAt(i));
      double twos = 0.0;
      if (lumaB2 != nullptr) {
        twos = std::max(twos, stillWeight(lumaB2->v[i], lumaK * floorAt(i)));
      }
      if (lumaA2 != nullptr) {
        twos = std::max(twos, stillWeight(lumaA2->v[i], lumaK * floorAt(i)));
      }
      allowance.v[i] = static_cast<Sample>(kLeakK * same * twos * leak.v[i]);
    }
  }
  auto still = [&](const Plane& change, size_t i) {
    return stillWeight(change.v[i], (lumaK * floorAt(i)) + allowance.v[i]);
  };

  if (fromBefore == nullptr || fromAfter == nullptr) {
    // The first or last frame of a sequence
    const Plane& estimate = (fromBefore != nullptr) ? *fromBefore : *fromAfter;
    const Plane& change = (fromBefore != nullptr) ? *lumaBefore : *lumaAfter;
    result.chroma = Plane(now.rows, now.cols);
    result.frameCombWeight = Plane(now.rows, now.cols);
    for (size_t i = 0; i < n; i++) {
      const double s = still(change, i);
      result.chroma.v[i] =
          static_cast<Sample>((s * estimate.v[i]) + ((1.0 - s) * line.v[i]));
      result.frameCombWeight.v[i] = static_cast<Sample>(s);
    }
    return result;
  }

  // Steady motion: the frame is the mean of its neighbours, having moved
  // about as far from each. The two one-sided combs then err in opposite
  // directions and their average cancels (the agreement test below caps the
  // speed). A scene cut fails the symmetry: one side changes, the other does
  // not. (Against the plain floor: raised where the chroma band is strong,
  // it would let luma near the subcarrier, which a 4-sample mean hides, pass
  // as steady while it moves.) Steady also puts the frame at the mean of the
  // frames two away (twice the distance, so twice the tolerance): a field
  // that differs from all four neighbours, a flash passing through, is not
  // steady even where its neighbours' mean happens to come close.
  Plane steady(now.rows, now.cols);
  {
    const Plane middle = lumaChange(current, mean(*before, *after));
    Plane middle2Plane;
    const Plane* middle2 = nullptr;
    if (before2 != nullptr && after2 != nullptr) {
      middle2Plane = lumaChange(current, mean(*before2, *after2));
      middle2 = &middle2Plane;
    }
    for (size_t i = 0; i < n; i++) {
      double s =
          stillWeight(middle.v[i], (kMiddleK * baseFloor) + allowance.v[i]) *
          stillWeight(std::fabs(lumaBefore->v[i] - lumaAfter->v[i]),
                      kMiddleK * baseFloor);
      if (middle2 != nullptr) {
        s *= stillWeight(middle2->v[i],
                         (2.0 * kMiddleK * baseFloor) + allowance.v[i]);
      }
      steady.v[i] = static_cast<Sample>(s);
    }
  }

  // Where both are still, the two frame combs should agree to within noise
  // and an allowance for slow steady motion, which their average cancels.
  // Both still also means the previous and next frames match in the chroma
  // band (they share a chroma phase): luma near the subcarrier, which the
  // luma test cannot see (a zone plate, fine stripes), moving between them
  // fails here.
  const Plane disagreement = cycleMean(
      map(*fromBefore, *fromAfter, [](double x, double y) { return x - y; }),
      f1);
  Plane bothStill(now.rows, now.cols);
  std::vector<bool> both(n, false);
  size_t bothCount = 0;
  for (size_t i = 0; i < n; i++) {
    bothStill.v[i] = static_cast<Sample>(
        still(*lumaBefore, i) * still(*lumaAfter, i) *
        stillWeight(detailBA->v[i], kPairK * detailFloorAt(i)));
    both[i] = std::max(bothStill.v[i], steady.v[i]) > 0.5;
  }
  for (int32_t r : regions.pictureRows()) {
    for (int32_t c = picture.sampleStart; c < picture.sampleEnd; c++) {
      if (both[regions.index(r, c)]) bothCount++;
    }
  }
  const double disagreementNoise =
      regions.picture(disagreement, kNoisePercentile,
                      (static_cast<double>(bothCount) >
                       0.05 * static_cast<double>(regions.pictureSamples()))
                          ? &both
                          : nullptr);

  Plane wBoth(now.rows, now.cols), wBefore(now.rows, now.cols),
      wAfter(now.rows, now.cols);
  for (size_t i = 0; i < n; i++) {
    const double agree =
        stillWeight(disagreement.v[i], (kNoiseK * disagreementNoise) +
                                           (kDetailK * amplitude.v[i]));
    // Both neighbours, averaged: both still, or steady motion.
    wBoth.v[i] =
        static_cast<Sample>(std::max(bothStill.v[i], steady.v[i]) * agree);
    const double rest = 1.0 - wBoth.v[i];
    // One neighbour alone, each against the plain floor: a bright edge
    // moving past, strong in the chroma band, would otherwise pass.
    const double plainBefore =
        stillWeight(lumaBefore->v[i], (lumaK * baseFloor) + allowance.v[i]);
    const double plainAfter =
        stillWeight(lumaAfter->v[i], (lumaK * baseFloor) + allowance.v[i]);
    wBefore.v[i] =
        static_cast<Sample>(rest * plainBefore * (1.0 - still(*lumaAfter, i)));
    wAfter.v[i] =
        static_cast<Sample>(rest * plainAfter * (1.0 - still(*lumaBefore, i)));
    // The luma test cannot see changes in fine detail (a thin line moving, a
    // flash in one field): the frame beyond the neighbour has the current
    // frame's chroma phase, so the chroma band compares directly, and the
    // one-sided comb needs it to match.
    if (detailB2 != nullptr) {
      wBefore.v[i] = static_cast<Sample>(
          wBefore.v[i] *
          (stillWeight(detailB2->v[i], kSideK * detailFloorAt(i))));
    }
    if (detailA2 != nullptr) {
      wAfter.v[i] = static_cast<Sample>(
          wAfter.v[i] *
          (stillWeight(detailA2->v[i], kSideK * detailFloorAt(i))));
    }
  }

  // A flash or a one-frame insert: the previous and next frames match each
  // other, but the current frame matches neither frame two away (the same
  // chroma phase, so the whole signal compares, colour included). No frame
  // comb then.
  if (before2 != nullptr || after2 != nullptr) {
    const Plane fullBA = fullChange(*before, *after);
    Plane fullB2Plane, fullA2Plane;
    const Plane* fullB2 = changeIf(&current, before2, fullB2Plane, fullChange);
    const Plane* fullA2 = changeIf(&current, after2, fullA2Plane, fullChange);
    const auto flashChanges = present({&fullBA, fullB2, fullA2});
    const double fullFloor = noiseFloor(flashChanges, regions);
    const double fullSlope =
        chromaNoiseSlope(flashChanges, fullFloor, amplitude, regions);

    // A fainter flash, near the noise floor, passes the thresholds above.
    // Relative to each other it does not: the frames two away (the current
    // frame's chroma phase) match each other far better than the current
    // frame matches either, in the luma or the chroma band. (Still, all are
    // noise apart; in steady motion the current frame is half as far from
    // each as they are from each other.)
    Plane lumaTwosPlane;
    const Plane* lumaTwos =
        (kFlashK > 0) ? changeIf(before2, after2, lumaTwosPlane, lumaChange)
                      : nullptr;
    for (size_t i = 0; i < n; i++) {
      const double fullFloorAt = fullFloor + (amplitude.v[i] * fullSlope);
      auto same = [&](const Plane& full, const Plane& luma) {
        return std::min(stillWeight(full.v[i], lumaK * fullFloorAt),
                        stillWeight(luma.v[i], lumaK * floorAt(i)));
      };
      const double matched = same(fullBA, *lumaBA);
      double consistent = 0.0;
      if (fullB2 != nullptr) {
        consistent = std::max(consistent, same(*fullB2, *lumaB2));
      }
      if (fullA2 != nullptr) {
        consistent = std::max(consistent, same(*fullA2, *lumaA2));
      }
      double keep = 1.0 - (matched * (1.0 - consistent));
      if (lumaTwos != nullptr) {
        const double lumaApart =
            stillWeight(std::min(lumaB2->v[i], lumaA2->v[i]),
                        kFlashK * std::max(static_cast<double>(lumaTwos->v[i]),
                                           floorAt(i)));
        const double detailApart = stillWeight(
            std::min(detailB2->v[i], detailA2->v[i]),
            kFlashK * std::max(static_cast<double>(detailTwos->v[i]),
                               detailFloorAt(i)));
        keep *= std::min(lumaApart, detailApart);
      }
      wBoth.v[i] = static_cast<Sample>(wBoth.v[i] * (keep));
      wBefore.v[i] = static_cast<Sample>(wBefore.v[i] * (keep));
      wAfter.v[i] = static_cast<Sample>(wAfter.v[i] * (keep));
    }
  }

  // Only where needed: where the 2D comb's chroma inverts cleanly from one
  // line of the field to the next, as chroma does, the 2D comb has nothing to
  // get wrong and is as right as a frame comb, and smoother: a frame comb
  // keeps the luma's noise near the subcarrier, which differs from line to
  // line, as a fine dot pattern, where the 2D comb averages it across lines.
  // Where the 2D chroma does not invert cleanly, it took luma detail for
  // chroma, and the frame comb is needed. This is judged on the 2D comb's own
  // output, not against the frame comb, so it holds where the frame comb is
  // the wrong one: a colour that changes from frame to frame and back (A, B,
  // A, B) passes every test between frames of the same chroma phase. Its
  // noise is held to the sync tip's, like the floors above.
  const Plane above = shift(line, -1, 0);
  const Plane below = shift(line, 1, 0);
  Plane inversion(now.rows, now.cols);
  for (size_t i = 0; i < n; i++) {
    inversion.v[i] =
        static_cast<Sample>(line.v[i] + ((above.v[i] + below.v[i]) / 2.0));
  }
  const Plane unsure = cycleMean(inversion, f1);
  const double unsureNoise = noiseFloor({&unsure}, regions);

  // Where it is needed, the frame comb is kept only where the frames two
  // away (the current frame's chroma phase) confirm the detail is still
  // there, relative to how well those two match each other (all noise apart
  // on a still picture; the picture's floor, which a strong chroma band
  // raises, is the most allowed). A one-sided comb needs the frame beyond
  // it; the averaged comb either, or steady motion (which passed its own
  // five-frame test).
  for (size_t i = 0; i < n; i++) {
    const double needed =
        1.0 - stillWeight(unsure.v[i], settings.detailThreshold * unsureNoise);
    const double apart =
        (detailTwos != nullptr)
            ? std::min(static_cast<double>(detailTwos->v[i]), detailFloorAt(i))
            : detailFloorAt(i);
    const double confirmBefore =
        (detailB2 != nullptr) ? stillWeight(detailB2->v[i], kConfirmK * apart)
                              : 0.0;
    const double confirmAfter =
        (detailA2 != nullptr) ? stillWeight(detailA2->v[i], kConfirmK * apart)
                              : 0.0;
    wBoth.v[i] =
        static_cast<Sample>(wBoth.v[i] * needed *
                            std::max(static_cast<double>(steady.v[i]),
                                     std::max(confirmBefore, confirmAfter)));
    wBefore.v[i] = static_cast<Sample>(wBefore.v[i] * (needed * confirmBefore));
    wAfter.v[i] = static_cast<Sample>(wAfter.v[i] * (needed * confirmAfter));
  }

  // The weights then change smoothly along the line: switching between the
  // combs from one sample to the next would modulate their difference.
  wBoth = boxMean(wBoth, kMaskWidth, 1, f1);
  wBefore = boxMean(wBefore, kMaskWidth, 1, f1);
  wAfter = boxMean(wAfter, kMaskWidth, 1, f1);

  result.chroma = Plane(now.rows, now.cols);
  result.frameCombWeight = Plane(now.rows, now.cols);
  for (size_t i = 0; i < n; i++) {
    const double weightBoth = wBoth.v[i];
    const double weightBefore = wBefore.v[i];
    const double weightAfter = wAfter.v[i];
    const double frameComb = weightBoth + weightBefore + weightAfter;
    const double fromB = fromBefore->v[i];
    const double fromA = fromAfter->v[i];
    result.chroma.v[i] = static_cast<Sample>(
        (weightBoth * (fromB + fromA) / 2.0) + (weightBefore * fromB) +
        (weightAfter * fromA) + ((1.0 - frameComb) * line.v[i]));
    result.frameCombWeight.v[i] = static_cast<Sample>(frameComb);
  }
  return result;
}

}  // namespace ntsc_comb
