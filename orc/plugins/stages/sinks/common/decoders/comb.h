/*
 * File:        comb.h
 * Module:      orc-core
 * Purpose:     NTSC comb filter decoder
 *
 * SPDX-License-Identifier: GPL-3.0-or-later
 * SPDX-FileCopyrightText: 2018-2021 Chad Page
 * SPDX-FileCopyrightText: 2018-2026 Simon Inns
 * SPDX-FileCopyrightText: 2019 Adam Sampson
 */

#ifndef COMB_H
#define COMB_H

#include <orc/stage/orc_source_parameters.h>

#include <cmath>
#include <fstream>
#include <iostream>

#include "comb_plane.h"
#include "componentframe.h"
#include "decoder.h"
#include "ntsc3d_comb.h"
#include "sourcefield.h"

class Comb {
 public:
  Comb();

  // Information about burst phase on a line (used by both composite and YC
  // paths)
  struct BurstInfo {
    double bsin, bcos;
  };

  // Comb filter configuration parameters
  struct Configuration {
    double chromaGain = 1.0;
    double chromaPhase = 0.0;
    int32_t dimensions = 2;
    // 3D only: motion adaptive (ntsc3d), or always against the previous
    // frame (ntsc3dnoadapt).
    bool adaptive = true;
    bool phaseCompensation = false;

    // Adaptive 3D only: low-frequency luma change, in multiples of the
    // measured noise floor, still counted as still (higher frame-combs slower
    // motion too).
    double motionThreshold = 4.0;

    // Adaptive 3D only: how far the 2D comb's chroma must fail to invert from
    // line to line, in multiples of its noise, before the frame comb is used
    // (lower frame-combs more widely; 0 wherever the picture is still).
    double detailThreshold = 2.0;

    int32_t getLookBehind() const;
    int32_t getLookAhead() const;
  };

  const Configuration& getConfiguration() const;
  void updateConfiguration(const ::orc::SourceParameters& videoParameters,
                           const Configuration& configuration);

  // Decode a sequence of fields into a sequence of interlaced frames
  void decodeFrames(const std::vector<SourceField>& inputFields,
                    int32_t startIndex, int32_t endIndex,
                    std::vector<ComponentFrame>& componentFrames);

  // Maximum frame size
  static constexpr int32_t MAX_WIDTH = 910;
  static constexpr int32_t MAX_HEIGHT = 625;  // PAL frame height

 protected:
  // YC decode path - for sources with separate Y and C channels
  void decodeFramesYC(const std::vector<SourceField>& inputFields,
                      int32_t startIndex, int32_t endIndex,
                      std::vector<ComponentFrame>& componentFrames);

  // Composite decode path - full comb filter for Y/C separation
  void decodeFramesComposite(const std::vector<SourceField>& inputFields,
                             int32_t startIndex, int32_t endIndex,
                             std::vector<ComponentFrame>& componentFrames);

 private:
  // Comb-filter configuration parameters
  bool configurationSet;
  Configuration configuration;
  ::orc::SourceParameters videoParameters;

  // An input frame in the process of being decoded
  class FrameBuffer {
   public:
    FrameBuffer(const ::orc::SourceParameters& videoParameters_,
                const Configuration& configuration_);

    void loadFields(const SourceField& firstField,
                    const SourceField& secondField);

    // YC-specific loading - for sources with separate Y and C channels
    void loadFieldsYC(const SourceField& firstField,
                      const SourceField& secondField);

    void split1D();
    void split2D();

    // 3D against the previous frame alone (ntsc3dnoadapt).
    void splitPreviousFrame(const FrameBuffer& previousFrame);

    // Motion-adaptive 3D (ntsc3d) from the frames up to two either side;
    // nullptr where a frame cannot be used.
    void splitAdaptive3D(const FrameBuffer* before2, const FrameBuffer* before,
                         const FrameBuffer* after, const FrameBuffer* after2);

    // Whether this frame can stand |offset| frames from |current| in a 3D
    // comb: it was loaded, is not blank padding, and (where the field phases
    // are known) has the chroma phase that offset implies.
    bool usableAround(const FrameBuffer& current, int32_t offset) const;

    // No frame loaded into this buffer for the current position.
    void markAbsent() { present = false; }

    void setComponentFrame(ComponentFrame& _componentFrame) {
      componentFrame = &_componentFrame;
    }

    void splitIQ();
    void splitIQlocked();

    // YC-specific chroma demodulation - skip Y/C separation, only demodulate C
    void splitIQ_YC();
    void splitIQlocked_YC();

    void filterIQ();
    void filterIQFull();
    void adjustY();
    void transformIQ(double chromaGain, double chromaPhase);

   private:
    const ::orc::SourceParameters& videoParameters;
    const Configuration& configuration;

    // Calculated frame height
    int32_t frameHeight;

    // IRE scaling
    double irescale;

    // Baseband samples (interlaced to form a complete frame).
    // int16_t in CVBS_U10_4FSC 10-bit domain.
    std::vector<int16_t> rawbuffer;

    // YC-specific: separate Y and C buffers for YC sources.
    // int16_t in CVBS_U10_4FSC 10-bit domain.
    std::vector<int16_t> luma_buffer;
    std::vector<int16_t> chroma_buffer;
    bool is_yc = false;  // True if loaded from YC source

    // Whether a frame is loaded, and whether it is uniform (the blank fields
    // a sink pads a sequence's ends with).
    bool present = false;
    bool blank = false;

    // Adaptive 3D only: the composite as doubles in field-separated rows.
    ntsc_comb::FieldLayout layout;
    ntsc_comb::Plane flat;

    // Per-field phase id (1-4) of the frame's two fields, measured from the
    // burst by the colour_frame_phase observer; -1 unknown.  getLinePhase()
    // needs the per-field value: the two fields of a frame have opposite
    // burst phase on the same field line.
    int32_t firstFieldPhaseID;
    int32_t secondFieldPhaseID;

    // 1D, 2D and 3D-filtered chroma samples
    struct Sample {
      double pixel[MAX_HEIGHT][MAX_WIDTH];
    } clpbuffer[3];

    // The component frame for output (if there is one)
    ComponentFrame* componentFrame;

    inline int32_t getFieldID(int32_t lineNumber) const;
    inline bool getLinePhase(int32_t lineNumber) const;

    // Helper to demodulate chroma with phase compensation (shared between
    // composite and YC)
    void demodulateChromaLocked(const double* chromaLine, int32_t lineNumber,
                                const Comb::BurstInfo& burstInfo, double* I,
                                double* Q, int32_t xOffset);

    // Helper to demodulate chroma without phase compensation (shared between
    // composite and YC)
    void demodulateChroma(const double* chromaLine, int32_t lineNumber,
                          bool linePhase, double* I, double* Q,
                          int32_t xOffset);
  };
};

#endif  // COMB_H
