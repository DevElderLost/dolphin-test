// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Planar silhouette shadows.
//
// The algorithm is a port of GPU/Common/PlanarShadows.{h,cpp} from the PPSSPP project
// (GPL-2.0-or-later), which in turn was written for the Yakumo (MHP3rd) host renderer. It is a
// heuristic: GX draw calls carry no entity IDs, so
//  * skinned, opaque, depth-writing 3D draws are shadow casters, and
//  * unskinned, opaque, depth-writing 3D draws are receiver candidates (terrain).
// Casters are flattened along a fixed direction onto the ground found below them, rasterized into
// a small coverage mask (union of all triangles, four samples per cell, 3x3 filtered), and the mask
// is clipped against the real receiver triangles. The result is a list of triangles that lie on the
// visible terrain, which the GPU then draws with alpha blending.
//
// Differences from the PPSSPP version:
//  * GX has no separate world matrix. All geometry is collected in eye (view) space and converted
//    to a Y-up "shadow space" when a scene is built. The up axis is either the camera's Y axis or
//    estimated from the dominant plane of the receiver triangles.
//  * Thresholds are relative to the size of the caster so that any game unit scale works.
//  * Draw calls of one actor are merged by overlapping bounding boxes instead of a world matrix.
//
// This file has no dependency on the rest of VideoCommon so that it can be unit tested on its own.

#include <array>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

struct PlanarShadowOptions
{
  bool enabled = false;
  float opacity = 0.25f;     // 0 to 0.6
  float directionX = 0.45f;  // Offset along the camera's right axis per unit of height.
  float directionZ = 0.30f;  // Offset along the camera's forward (towards the viewer) axis.
  float floorOffset = 0.0f;
  unsigned resolution = 96;  // Mask size in cells: 32 to 192.
  // Distance LOD: 0 keeps every caster, 1 to 3 drop casters that cover less than 6%, 3% or 1.5%
  // of the screen height.
  unsigned lod = 2;
  // Estimate the ground plane from the receivers instead of assuming a level camera.
  bool auto_up = true;
};

// A vertex of the generated shadow geometry, in eye space.
struct PlanarShadowVertex
{
  float x, y, z;
  uint32_t alpha;  // 0 - 255, the shadow strength.
};

class PlanarShadows
{
public:
  using Point = std::array<float, 3>;
  using Triangle = std::array<Point, 3>;

  // Everything that was drawn with the same projection and viewport belongs to the same scene.
  struct Scene
  {
    uint32_t id = 0;
    float cot = 2.4f;  // Projection Y scale, cot(fov / 2) for a perspective projection.
  };

  PlanarShadows() = default;

  void SetOptions(const PlanarShadowOptions& o) { m_options = o; }
  const PlanarShadowOptions& Options() const { return m_options; }

  // Forget everything that was collected.
  void Reset();
  // Forget what was collected for one scene.
  void DropScene(uint32_t id);

  // Collect eye space triangles of one draw.
  void AddCaster(const Scene& scene, const std::vector<Triangle>& triangles);
  void AddReceiver(const Scene& scene, const std::vector<Triangle>& triangles);

  bool HasCasters(uint32_t id) const;
  // True while a draw of that kind can still be collected. Lets the caller skip decoding when a
  // per frame limit was reached.
  bool WantsCasters(uint32_t id) const
  {
    return m_options.enabled && m_options.opacity > 0.0f && !m_finished.count(id) &&
           m_triangles < kTriangleLimit;
  }
  bool WantsReceivers(uint32_t id) const
  {
    return m_options.enabled && !m_finished.count(id) && m_receiver_triangles < kReceiverLimit;
  }
  // Scenes that have casters waiting to be drawn.
  std::vector<uint32_t> PendingScenes() const;

  // Turns everything collected for the scene into shadow triangles (eye space), and clears it.
  // Returns false (and leaves out empty) when there is nothing to draw.
  bool Build(uint32_t id, std::vector<PlanarShadowVertex>& out);

  // Statistics of the previous Build, for logging.
  struct Stats
  {
    size_t clusters = 0;           // Actors left after merging draws.
    size_t receiver_triangles = 0; // Up facing receiver triangles used.
    Point up{0.0f, 1.0f, 0.0f};    // Ground normal in eye space.
    Point largest_extent{0, 0, 0}; // Width, height, depth of the biggest actor (shadow space).
    size_t casters = 0;
    size_t lod_culled = 0;
    size_t emitted = 0;
    size_t conformed = 0;
    size_t skipped = 0;
    size_t work = 0;
  };
  const Stats& GetStats() const { return m_stats; }

  // Orthonormal basis of the shadow space, expressed in eye space.
  struct Basis
  {
    Point right{1.0f, 0.0f, 0.0f};
    Point up{0.0f, 1.0f, 0.0f};
    Point forward{0.0f, 0.0f, 1.0f};
  };
  // Builds the basis for a given up vector (need not be normalized).
  static Basis MakeBasis(const Point& up);
  // Dominant plane normal of a set of triangles, weighted by area. Returns false when the
  // triangles do not agree on one plane.
  static bool EstimateUp(const std::vector<Triangle>& triangles, Point& up);

private:
  struct Grid
  {
    bool built = false;
    size_t count = 0;
    float x0 = 0, z0 = 0, inverse = 0;
    unsigned width = 0, height = 0;
    std::vector<std::vector<uint32_t>> cells;
    std::vector<uint32_t> large;
    std::vector<uint32_t> seen;
    uint32_t stamp = 0;
  };
  struct Caster
  {
    uint32_t scene = 0;
    float cot = 2.4f;
    std::vector<Triangle> triangles;
    Grid grid;  // receivers only
    // Receivers: covariance of the area weighted triangle normals, used to find the ground plane.
    // Order: xx, xy, xz, yy, yz, zz.
    std::array<double, 6> normal_covariance{};
  };
  struct Pick
  {
    const Caster* caster = nullptr;
    float distance = 0.0f;
    unsigned resolution = 0;
    float fade = 1.0f;
  };
  struct ClippedPolygon;

  static ClippedPolygon ClipRectangle(const Triangle& t, float x0, float z0, float x1, float z1);
  static void BuildGrid(Caster& receiver);
  template <class Visit>
  void NearReceivers(Caster* receiver, float x0, float z0, float x1, float z1, Visit visit);
  static float LodCullFraction(unsigned lod);
  bool LodSelect(const Caster& caster, Pick& pick) const;
  void Rasterize(Caster* receiver, const Caster& caster, unsigned n, float fade,
                 std::vector<PlanarShadowVertex>& out);
  Caster* FindReceiver(uint32_t id);
  Basis ChooseBasis(const Caster* receiver);
  void PrepareReceiver(Caster& receiver, const Basis& basis);
  std::vector<Caster> ClusterCasters(uint32_t id, const Basis& basis);

  PlanarShadowOptions m_options;
  std::vector<Caster> m_casters;
  std::vector<Caster> m_receivers;
  std::set<uint32_t> m_finished;
  std::vector<uint32_t> m_candidates;
  size_t m_receiver_triangles = 0;
  size_t m_triangles = 0;
  Stats m_stats;

  // The ground plane estimate of the previous scene. Smoothed so that it does not jitter.
  Point m_last_up{0.0f, 1.0f, 0.0f};
  bool m_have_last_up = false;

  static constexpr size_t kTriangleLimit = 40000;
  static constexpr size_t kReceiverLimit = 80000;
  static constexpr size_t kMaxCasters = 96;
  static constexpr size_t kMaxReceiverSets = 16;
  static constexpr size_t kMaxShadowVertices = 120000;
  static constexpr size_t kPixelWorkLimit = 4000000;
  static constexpr unsigned kGridCells = 48;
  static constexpr size_t kGridLargeCells = 64;
};
