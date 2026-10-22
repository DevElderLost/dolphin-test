// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// Ported from PPSSPP's GPU/Common/PlanarShadows.cpp, see PlanarShadows.h.

#include "VideoCommon/PlanarShadows.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

using Point = PlanarShadows::Point;
using Triangle = PlanarShadows::Triangle;

// Sutherland-Hodgman clipping in shadow space X/Z. Y interpolates along the receiver triangle, so
// the emitted polygon lies on the actual surface.
struct PlanarShadows::ClippedPolygon
{
  std::array<Point, 8> points{};  // A triangle clipped by four half-planes has at most seven vertices.
  size_t count = 0;
  bool empty() const { return count == 0; }
  size_t size() const { return count; }
  const Point& back() const { return points[count - 1]; }
  const Point* begin() const { return points.data(); }
  const Point* end() const { return points.data() + count; }
  const Point& operator[](size_t i) const { return points[i]; }
  void push_back(const Point& p) { points[count++] = p; }
};

static inline unsigned Popcount4(unsigned v)
{
  static constexpr unsigned char table[16] = {0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4};
  return table[v & 15];
}

static inline float Edge(const Point& a, const Point& b, float x, float z)
{
  return (b[0] - a[0]) * (z - a[2]) - (b[2] - a[2]) * (x - a[0]);
}

static inline float Dot(const Point& a, const Point& b)
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static inline Point Cross(const Point& a, const Point& b)
{
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

static inline float Length(const Point& a)
{
  return std::sqrt(Dot(a, a));
}

static bool Normalize(Point& a)
{
  const float length = Length(a);
  if (!std::isfinite(length) || length < 1e-8f)
    return false;
  for (float& c : a)
    c /= length;
  return true;
}

static bool HeightAt(const Triangle& t, float x, float z, float& height)
{
  const float area = Edge(t[0], t[1], t[2][0], t[2][2]);
  if (std::abs(area) < 1e-12f)
    return false;
  const float a = Edge(t[1], t[2], x, z) / area;
  const float b = Edge(t[2], t[0], x, z) / area;
  const float c = 1 - a - b;
  if (a < -1e-5f || b < -1e-5f || c < -1e-5f)
    return false;
  height = a * t[0][1] + b * t[1][1] + c * t[2][1];
  return std::isfinite(height);
}

static unsigned CellOf(float offset, float inverse, unsigned count)
{
  const float cell = std::floor(offset * inverse);
  if (!(cell > 0.0f))
    return 0;
  if (cell >= static_cast<float>(count - 1))
    return count - 1;
  return static_cast<unsigned>(cell);
}

PlanarShadows::ClippedPolygon PlanarShadows::ClipRectangle(const Triangle& t, float x0, float z0,
                                                           float x1, float z1)
{
  ClippedPolygon polygon;
  for (const auto& p : t)
    polygon.push_back(p);
  const auto cut = [&](unsigned axis, float bound, bool greater) {
    ClippedPolygon next;
    if (polygon.empty())
      return;
    Point a = polygon.back();
    float da = greater ? a[axis] - bound : bound - a[axis];
    for (const Point& b : polygon)
    {
      const float db = greater ? b[axis] - bound : bound - b[axis];
      if ((da >= 0) != (db >= 0))
      {
        const float ratio = da / (da - db);
        Point hit{};
        for (unsigned k = 0; k < 3; ++k)
          hit[k] = a[k] + ratio * (b[k] - a[k]);
        hit[axis] = bound;
        next.push_back(hit);
      }
      if (db >= 0)
        next.push_back(b);
      a = b;
      da = db;
    }
    polygon = next;
  };
  cut(0, x0, true);
  cut(0, x1, false);
  cut(2, z0, true);
  cut(2, z1, false);
  return polygon;
}

void PlanarShadows::Reset()
{
  m_casters.clear();
  m_receivers.clear();
  m_finished.clear();
  m_receiver_triangles = 0;
  m_triangles = 0;
  m_stats = Stats();
}

void PlanarShadows::DropScene(uint32_t id)
{
  const auto on_scene = [&](const Caster& c) { return c.scene == id; };
  m_casters.erase(std::remove_if(m_casters.begin(), m_casters.end(), on_scene), m_casters.end());
  m_receivers.erase(std::remove_if(m_receivers.begin(), m_receivers.end(), on_scene),
                    m_receivers.end());
  m_finished.erase(id);
  // The counters are per frame limits, recount what is left.
  m_triangles = 0;
  for (const Caster& c : m_casters)
    m_triangles += c.triangles.size();
  m_receiver_triangles = 0;
  for (const Caster& c : m_receivers)
    m_receiver_triangles += c.triangles.size();
}

std::vector<uint32_t> PlanarShadows::PendingScenes() const
{
  std::set<uint32_t> scenes;
  for (const Caster& c : m_casters)
  {
    if (!c.triangles.empty())
      scenes.insert(c.scene);
  }
  return std::vector<uint32_t>(scenes.begin(), scenes.end());
}

bool PlanarShadows::HasCasters(uint32_t id) const
{
  for (const Caster& c : m_casters)
  {
    if (c.scene == id && !c.triangles.empty())
      return true;
  }
  return false;
}

PlanarShadows::Caster* PlanarShadows::FindReceiver(uint32_t id)
{
  const auto it = std::find_if(m_receivers.begin(), m_receivers.end(),
                               [&](const Caster& r) { return r.scene == id; });
  return it == m_receivers.end() ? nullptr : &*it;
}

void PlanarShadows::AddReceiver(const Scene& scene, const std::vector<Triangle>& triangles)
{
  if (!m_options.enabled || m_finished.count(scene.id))
    return;
  Caster* receiver = FindReceiver(scene.id);
  if (!receiver)
  {
    if (m_receivers.size() >= kMaxReceiverSets)
      return;
    Caster r;
    r.scene = scene.id;
    r.cot = scene.cot;
    m_receivers.push_back(std::move(r));
    receiver = &m_receivers.back();
  }
  for (const Triangle& t : triangles)
  {
    if (m_receiver_triangles >= kReceiverLimit)
      return;
    Point a{}, b{};
    for (unsigned k = 0; k < 3; ++k)
    {
      a[k] = t[1][k] - t[0][k];
      b[k] = t[2][k] - t[0][k];
    }
    const Point normal = Cross(a, b);
    const float length = Length(normal);
    if (!std::isfinite(length) || !(length > 1e-12f))
      continue;
    receiver->triangles.push_back(t);
    ++m_receiver_triangles;
    // Area weighted normal covariance, only from surfaces that could be a floor for a camera that
    // is not looking straight up or down.
    if (std::abs(normal[1]) > 0.3f * length)
    {
      const double w = 0.5 / length;
      receiver->normal_covariance[0] += w * normal[0] * normal[0];
      receiver->normal_covariance[1] += w * normal[0] * normal[1];
      receiver->normal_covariance[2] += w * normal[0] * normal[2];
      receiver->normal_covariance[3] += w * normal[1] * normal[1];
      receiver->normal_covariance[4] += w * normal[1] * normal[2];
      receiver->normal_covariance[5] += w * normal[2] * normal[2];
    }
  }
}

void PlanarShadows::AddCaster(const Scene& scene, const std::vector<Triangle>& triangles)
{
  if (!m_options.enabled || m_options.opacity <= 0.0f || m_finished.count(scene.id))
    return;
  if (m_casters.size() >= kMaxCasters)
  {
    ++m_stats.skipped;
    return;
  }
  Caster c;
  c.scene = scene.id;
  c.cot = scene.cot;
  for (const Triangle& t : triangles)
  {
    if (m_triangles >= kTriangleLimit)
    {
      ++m_stats.skipped;
      break;
    }
    c.triangles.push_back(t);
    ++m_triangles;
  }
  if (!c.triangles.empty())
    m_casters.push_back(std::move(c));
}

PlanarShadows::Basis PlanarShadows::MakeBasis(const Point& up_in)
{
  Basis basis;
  Point up = up_in;
  if (!Normalize(up))
    up = {0.0f, 1.0f, 0.0f};
  // The camera's right axis, made perpendicular to the up axis.
  Point right{1.0f, 0.0f, 0.0f};
  const float d = Dot(right, up);
  for (unsigned k = 0; k < 3; ++k)
    right[k] -= up[k] * d;
  if (!Normalize(right))
  {
    right = {0.0f, 0.0f, 1.0f};
    const float d2 = Dot(right, up);
    for (unsigned k = 0; k < 3; ++k)
      right[k] -= up[k] * d2;
    Normalize(right);
  }
  basis.right = right;
  basis.up = up;
  basis.forward = Cross(right, up);
  return basis;
}

bool PlanarShadows::EstimateUp(const std::vector<Triangle>& triangles, Point& up)
{
  std::array<double, 6> c{};
  for (const Triangle& t : triangles)
  {
    Point a{}, b{};
    for (unsigned k = 0; k < 3; ++k)
    {
      a[k] = t[1][k] - t[0][k];
      b[k] = t[2][k] - t[0][k];
    }
    const Point n = Cross(a, b);
    const float length = Length(n);
    if (!std::isfinite(length) || !(length > 1e-12f) || std::abs(n[1]) <= 0.3f * length)
      continue;
    const double w = 0.5 / length;
    c[0] += w * n[0] * n[0];
    c[1] += w * n[0] * n[1];
    c[2] += w * n[0] * n[2];
    c[3] += w * n[1] * n[1];
    c[4] += w * n[1] * n[2];
    c[5] += w * n[2] * n[2];
  }
  const double trace = c[0] + c[3] + c[5];
  if (!(trace > 0.0))
    return false;
  // Power iteration for the dominant eigenvector of the symmetric matrix.
  double v[3] = {0.0, 1.0, 0.0};
  for (int i = 0; i < 16; ++i)
  {
    const double x = c[0] * v[0] + c[1] * v[1] + c[2] * v[2];
    const double y = c[1] * v[0] + c[3] * v[1] + c[4] * v[2];
    const double z = c[2] * v[0] + c[4] * v[1] + c[5] * v[2];
    const double length = std::sqrt(x * x + y * y + z * z);
    if (!(length > 0.0))
      return false;
    v[0] = x / length;
    v[1] = y / length;
    v[2] = z / length;
  }
  const double lambda = v[0] * (c[0] * v[0] + c[1] * v[1] + c[2] * v[2]) +
                        v[1] * (c[1] * v[0] + c[3] * v[1] + c[4] * v[2]) +
                        v[2] * (c[2] * v[0] + c[4] * v[1] + c[5] * v[2]);
  // The surfaces have to mostly agree, otherwise there is no ground to speak of.
  if (lambda < 0.6 * trace)
    return false;
  if (v[1] < 0.0)
  {
    for (double& x : v)
      x = -x;
  }
  up = {static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])};
  return true;
}

PlanarShadows::Basis PlanarShadows::ChooseBasis(const Caster* receiver)
{
  Point up{0.0f, 1.0f, 0.0f};
  if (m_options.auto_up && receiver && !receiver->triangles.empty())
  {
    const std::array<double, 6>& c = receiver->normal_covariance;
    const double trace = c[0] + c[3] + c[5];
    if (trace > 0.0)
    {
      double v[3] = {0.0, 1.0, 0.0};
      for (int i = 0; i < 16; ++i)
      {
        const double x = c[0] * v[0] + c[1] * v[1] + c[2] * v[2];
        const double y = c[1] * v[0] + c[3] * v[1] + c[4] * v[2];
        const double z = c[2] * v[0] + c[4] * v[1] + c[5] * v[2];
        const double length = std::sqrt(x * x + y * y + z * z);
        if (!(length > 0.0))
          break;
        v[0] = x / length;
        v[1] = y / length;
        v[2] = z / length;
      }
      const double lambda = v[0] * (c[0] * v[0] + c[1] * v[1] + c[2] * v[2]) +
                            v[1] * (c[1] * v[0] + c[3] * v[1] + c[4] * v[2]) +
                            v[2] * (c[2] * v[0] + c[4] * v[1] + c[5] * v[2]);
      if (lambda >= 0.6 * trace)
      {
        if (v[1] < 0.0)
        {
          for (double& x : v)
            x = -x;
        }
        up = {static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])};
      }
    }
  }
  // Smooth small changes of the estimate, snap on large ones (a new camera angle).
  if (m_have_last_up && Dot(m_last_up, up) > 0.97f)
  {
    Point blended{};
    for (unsigned k = 0; k < 3; ++k)
      blended[k] = m_last_up[k] * 0.85f + up[k] * 0.15f;
    if (Normalize(blended))
      up = blended;
  }
  m_last_up = up;
  m_have_last_up = true;
  return MakeBasis(up);
}

// Moves the receiver triangles into shadow space and drops those that do not face up.
void PlanarShadows::PrepareReceiver(Caster& receiver, const Basis& basis)
{
  std::vector<Triangle> kept;
  kept.reserve(receiver.triangles.size());
  for (const Triangle& t : receiver.triangles)
  {
    Triangle s;
    for (unsigned i = 0; i < 3; ++i)
    {
      s[i] = {Dot(t[i], basis.right), Dot(t[i], basis.up), Dot(t[i], basis.forward)};
    }
    Point a{}, b{};
    for (unsigned k = 0; k < 3; ++k)
    {
      a[k] = s[1][k] - s[0][k];
      b[k] = s[2][k] - s[0][k];
    }
    const Point normal = Cross(a, b);
    const float length = Length(normal);
    // Only surfaces that face mostly up can receive a shadow.
    if (std::isfinite(length) && length > 1e-12f && std::abs(normal[1]) > 0.35f * length)
      kept.push_back(s);
  }
  receiver.triangles = std::move(kept);
  receiver.grid = Grid{};
}

// Moves the caster triangles into shadow space and merges draws that belong to the same actor.
std::vector<PlanarShadows::Caster> PlanarShadows::ClusterCasters(uint32_t id, const Basis& basis)
{
  struct Box
  {
    Point low{1e30f, 1e30f, 1e30f};
    Point high{-1e30f, -1e30f, -1e30f};
  };
  std::vector<Caster> parts;
  std::vector<Box> boxes;
  for (const Caster& c : m_casters)
  {
    if (c.scene != id || c.triangles.empty())
      continue;
    Caster part;
    part.scene = id;
    part.cot = c.cot;
    Box box;
    part.triangles.reserve(c.triangles.size());
    for (const Triangle& t : c.triangles)
    {
      Triangle s;
      for (unsigned i = 0; i < 3; ++i)
      {
        s[i] = {Dot(t[i], basis.right), Dot(t[i], basis.up), Dot(t[i], basis.forward)};
        for (unsigned k = 0; k < 3; ++k)
        {
          box.low[k] = std::min(box.low[k], s[i][k]);
          box.high[k] = std::max(box.high[k], s[i][k]);
        }
      }
      part.triangles.push_back(s);
    }
    parts.push_back(std::move(part));
    boxes.push_back(box);
  }

  // Union-find over overlapping bounding boxes.
  std::vector<size_t> parent(parts.size());
  std::iota(parent.begin(), parent.end(), size_t(0));
  const auto find = [&](size_t i) {
    while (parent[i] != i)
    {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  const auto extent = [](const Box& b) {
    return std::max({b.high[0] - b.low[0], b.high[1] - b.low[1], b.high[2] - b.low[2]});
  };
  for (size_t i = 0; i < parts.size(); ++i)
  {
    for (size_t j = i + 1; j < parts.size(); ++j)
    {
      const float margin = 0.05f * std::max(extent(boxes[i]), extent(boxes[j]));
      bool overlap = true;
      for (unsigned k = 0; k < 3; ++k)
      {
        if (boxes[i].low[k] > boxes[j].high[k] + margin ||
            boxes[j].low[k] > boxes[i].high[k] + margin)
        {
          overlap = false;
          break;
        }
      }
      if (overlap)
        parent[find(i)] = find(j);
    }
  }
  std::vector<Caster> clusters;
  std::vector<int> cluster_of(parts.size(), -1);
  for (size_t i = 0; i < parts.size(); ++i)
  {
    const size_t root = find(i);
    if (cluster_of[root] < 0)
    {
      cluster_of[root] = static_cast<int>(clusters.size());
      Caster merged;
      merged.scene = id;
      merged.cot = parts[i].cot;
      clusters.push_back(std::move(merged));
    }
    Caster& merged = clusters[cluster_of[root]];
    merged.triangles.insert(merged.triangles.end(), parts[i].triangles.begin(),
                            parts[i].triangles.end());
  }
  return clusters;
}

void PlanarShadows::BuildGrid(Caster& receiver)
{
  Grid& g = receiver.grid;
  g = Grid{};
  g.built = true;
  g.count = receiver.triangles.size();
  if (receiver.triangles.empty())
    return;
  float min_x = std::numeric_limits<float>::max(), min_z = min_x;
  float max_x = -min_x, max_z = -min_x;
  for (const Triangle& t : receiver.triangles)
  {
    for (const Point& p : t)
    {
      min_x = std::min(min_x, p[0]);
      max_x = std::max(max_x, p[0]);
      min_z = std::min(min_z, p[2]);
      max_z = std::max(max_z, p[2]);
    }
  }
  const float span = std::max({max_x - min_x, max_z - min_z, 1e-6f});
  g.x0 = min_x;
  g.z0 = min_z;
  g.inverse = static_cast<float>(kGridCells) / span;
  g.width = std::min(kGridCells, static_cast<unsigned>((max_x - min_x) * g.inverse) + 1u);
  g.height = std::min(kGridCells, static_cast<unsigned>((max_z - min_z) * g.inverse) + 1u);
  g.cells.assign(static_cast<size_t>(g.width) * g.height, {});
  g.seen.assign(receiver.triangles.size(), 0u);
  for (size_t i = 0; i < receiver.triangles.size(); ++i)
  {
    const Triangle& t = receiver.triangles[i];
    const unsigned cx0 =
        CellOf(std::min({t[0][0], t[1][0], t[2][0]}) - g.x0, g.inverse, g.width);
    const unsigned cx1 =
        CellOf(std::max({t[0][0], t[1][0], t[2][0]}) - g.x0, g.inverse, g.width);
    const unsigned cz0 =
        CellOf(std::min({t[0][2], t[1][2], t[2][2]}) - g.z0, g.inverse, g.height);
    const unsigned cz1 =
        CellOf(std::max({t[0][2], t[1][2], t[2][2]}) - g.z0, g.inverse, g.height);
    if (static_cast<size_t>(cx1 - cx0 + 1) * (cz1 - cz0 + 1) > kGridLargeCells)
    {
      g.large.push_back(static_cast<uint32_t>(i));
      continue;
    }
    for (unsigned cz = cz0; cz <= cz1; ++cz)
    {
      for (unsigned cx = cx0; cx <= cx1; ++cx)
        g.cells[static_cast<size_t>(cz) * g.width + cx].push_back(static_cast<uint32_t>(i));
    }
  }
}

// Visits the receiver triangles whose X/Z cells overlap the rectangle, each once and in their
// original order. The visitor returns false to stop.
template <class Visit>
void PlanarShadows::NearReceivers(Caster* receiver, float x0, float z0, float x1, float z1,
                                  Visit visit)
{
  if (!receiver)
    return;
  if (!receiver->grid.built || receiver->grid.count != receiver->triangles.size())
    BuildGrid(*receiver);
  Grid& g = receiver->grid;
  if (g.cells.empty())
    return;
  if (++g.stamp == 0u)
  {
    std::fill(g.seen.begin(), g.seen.end(), 0u);
    g.stamp = 1u;
  }
  m_candidates.clear();
  const auto add = [&](uint32_t i) {
    if (g.seen[i] == g.stamp)
      return;
    g.seen[i] = g.stamp;
    m_candidates.push_back(i);
  };
  for (uint32_t i : g.large)
    add(i);
  const unsigned cx0 = CellOf(x0 - g.x0, g.inverse, g.width),
                 cx1 = CellOf(x1 - g.x0, g.inverse, g.width);
  const unsigned cz0 = CellOf(z0 - g.z0, g.inverse, g.height),
                 cz1 = CellOf(z1 - g.z0, g.inverse, g.height);
  for (unsigned cz = cz0; cz <= cz1; ++cz)
  {
    for (unsigned cx = cx0; cx <= cx1; ++cx)
    {
      for (uint32_t i : g.cells[static_cast<size_t>(cz) * g.width + cx])
        add(i);
    }
  }
  std::sort(m_candidates.begin(), m_candidates.end());
  for (uint32_t i : m_candidates)
  {
    if (!visit(receiver->triangles[i]))
      return;
  }
}

// Share of the screen height a caster covers below which its shadow is dropped.
float PlanarShadows::LodCullFraction(unsigned lod)
{
  switch (lod)
  {
  case 1:
    return 0.06f;
  case 2:
    return 0.03f;
  case 3:
    return 0.015f;
  default:
    return 0.0f;
  }
}

// Decides from the caster's size on screen whether it gets a silhouette, how fine its mask is and
// how strongly it fades out near the limit. Shadow space is a rotated eye space, so the distance to
// the camera is the length of the position.
bool PlanarShadows::LodSelect(const Caster& caster, Pick& pick) const
{
  pick.resolution = m_options.resolution;
  if (m_options.lod == 0)
    return true;
  Point low{1e30f, 1e30f, 1e30f}, high{-1e30f, -1e30f, -1e30f};
  for (const Triangle& t : caster.triangles)
  {
    for (const Point& p : t)
    {
      for (unsigned k = 0; k < 3; ++k)
      {
        low[k] = std::min(low[k], p[k]);
        high[k] = std::max(high[k], p[k]);
      }
    }
  }
  const float height = high[1] - low[1];
  if (!(height > 1e-4f))
    return true;
  const Point centre{(low[0] + high[0]) * 0.5f, (low[1] + high[1]) * 0.5f,
                     (low[2] + high[2]) * 0.5f};
  const float distance = Length(centre);
  if (!std::isfinite(distance) || distance < 1e-4f)
    return true;
  // The height covers height / distance * cot / 2 of the screen.
  const float cot = std::abs(caster.cot);
  const float scale = std::isfinite(cot) && cot > 0.1f && cot < 50.0f ? cot : 2.4f;
  const float screen = height / distance * scale * 0.5f;
  const float cull = LodCullFraction(m_options.lod);
  if (screen <= cull)
    return false;
  pick.distance = distance;
  pick.fade = std::clamp((screen - cull) / cull, 0.0f, 1.0f);
  unsigned resolution = m_options.resolution;
  if (screen < 0.07f)
    resolution /= 4u;
  else if (screen < 0.15f)
    resolution /= 2u;
  pick.resolution = std::max(32u, (resolution / 8u) * 8u);
  return true;
}

void PlanarShadows::Rasterize(Caster* receiver, const Caster& caster, unsigned n, float fade,
                              std::vector<PlanarShadowVertex>& out)
{
  const float opacity = m_options.opacity * fade;
  if (std::lround(opacity * 255.0f) <= 0)
    return;
  float floor_y = std::numeric_limits<float>::max();
  float top = -floor_y;
  for (const auto& t : caster.triangles)
  {
    for (const auto& p : t)
    {
      floor_y = std::min(floor_y, p[1]);
      top = std::max(top, p[1]);
    }
  }
  if (!(top > floor_y + 1e-4f))
    return;
  const float height = top - floor_y;
  // Ground under the actor stabilizes the projection reference instead of following a moving toe
  // every frame. No collision data is read.
  float centre_x = 0, centre_z = 0;
  size_t points = 0;
  for (const auto& t : caster.triangles)
  {
    for (const auto& p : t)
    {
      centre_x += p[0];
      centre_z += p[2];
      ++points;
    }
  }
  centre_x /= static_cast<float>(points);
  centre_z /= static_cast<float>(points);
  float ground = -std::numeric_limits<float>::infinity();
  NearReceivers(receiver, centre_x, centre_z, centre_x, centre_z, [&](const Triangle& t) {
    float y{};
    if (HeightAt(t, centre_x, centre_z, y) && y <= floor_y + height * 0.35f &&
        y >= floor_y - height * 2 && y > ground)
    {
      ground = y;
    }
    return true;
  });
  if (std::isfinite(ground))
    floor_y = ground;
  floor_y += m_options.floorOffset;
  const float bias = std::max(height * 0.003f, 1e-5f);
  std::vector<Triangle> projected = caster.triangles;
  float x0 = std::numeric_limits<float>::max(), z0 = x0;
  float x1 = -x0, z1 = -x0;
  for (auto& t : projected)
  {
    for (auto& p : t)
    {
      const float h = p[1] - floor_y;
      p = {p[0] + m_options.directionX * h, floor_y + bias, p[2] + m_options.directionZ * h};
      x0 = std::min(x0, p[0]);
      x1 = std::max(x1, p[0]);
      z0 = std::min(z0, p[2]);
      z1 = std::max(z1, p[2]);
    }
  }
  if (!(x1 > x0 + 1e-4f && z1 > z0 + 1e-4f))
    return;
  // Square cells with a power-of-two size: small pose changes no longer resize and stretch every
  // texel as an animated bounding box changes. Quantized scale can still step when an actor
  // crosses a power-of-two extent.
  const float span = std::max(x1 - x0, z1 - z0) / static_cast<float>(n - 8);
  const float dx = std::exp2(std::ceil(std::log2(std::max(span, height * 1e-3f))));
  const float dz = dx;
  x0 = (std::floor(x0 / dx) - 2) * dx;
  z0 = (std::floor(z0 / dz) - 2) * dz;

  // Bin nearby receivers by mask row to bound clipping work.
  std::vector<std::vector<const Triangle*>> rows(n);
  NearReceivers(receiver, x0, z0, x0 + n * dx, z0 + n * dz, [&](const Triangle& t) {
    const float low_y = std::min({t[0][1], t[1][1], t[2][1]});
    const float high_y = std::max({t[0][1], t[1][1], t[2][1]});
    const float left = std::min({t[0][0], t[1][0], t[2][0]});
    const float right = std::max({t[0][0], t[1][0], t[2][0]});
    const float low = std::min({t[0][2], t[1][2], t[2][2]});
    const float high = std::max({t[0][2], t[1][2], t[2][2]});
    if (low_y > floor_y + height * 0.5f || high_y < floor_y - height * 2 || right < x0 ||
        left > x0 + n * dx || high < z0 || low > z0 + n * dz)
    {
      return true;
    }
    const int first = int(std::clamp(std::floor((low - z0) / dz), 0.0f, float(n - 1)));
    const int last = int(std::clamp(std::floor((high - z0) / dz), 0.0f, float(n - 1)));
    for (int row = first; row <= last; ++row)
      rows[row].push_back(&t);
    return true;
  });

  std::vector<unsigned char> mask(n * n, 0);
  for (auto& t : projected)
  {
    for (auto& p : t)
    {
      p[0] = (p[0] - x0) / dx;
      p[2] = (p[2] - z0) / dz;
    }
    if (std::abs(Edge(t[0], t[1], t[2][0], t[2][2])) < 1e-7f)
      continue;
    const int left = std::clamp(
        static_cast<int>(std::floor(std::min({t[0][0], t[1][0], t[2][0]}))), 0, int(n) - 1);
    const int right = std::clamp(
        static_cast<int>(std::ceil(std::max({t[0][0], t[1][0], t[2][0]}))), 0, int(n) - 1);
    const int low = std::clamp(
        static_cast<int>(std::floor(std::min({t[0][2], t[1][2], t[2][2]}))), 0, int(n) - 1);
    const int high = std::clamp(
        static_cast<int>(std::ceil(std::max({t[0][2], t[1][2], t[2][2]}))), 0, int(n) - 1);
    m_stats.work += static_cast<size_t>(right - left + 1) * static_cast<size_t>(high - low + 1);
    if (m_stats.work > kPixelWorkLimit)
    {
      ++m_stats.skipped;
      return;
    }
    for (int z = low; z <= high; ++z)
    {
      for (int x = left; x <= right; ++x)
      {
        // Union is monotonic: samples covered by an earlier triangle cannot change.
        unsigned char& cell = mask[z * n + x];
        if (cell == 15u)
          continue;
        // Four coverage bits union across triangles before filtering. This gives partial
        // coverage without a hard one-sample toggle.
        for (unsigned sample = 0; sample < 4; ++sample)
        {
          if (cell & (1u << sample))
            continue;
          const float sx = x + ((sample & 1) ? 0.75f : 0.25f);
          const float sz = z + ((sample & 2) ? 0.75f : 0.25f);
          const float a = Edge(t[0], t[1], sx, sz), b = Edge(t[1], t[2], sx, sz),
                      c = Edge(t[2], t[0], sx, sz);
          if ((a >= 0 && b >= 0 && c >= 0) || (a <= 0 && b <= 0 && c <= 0))
            cell |= 1u << sample;
        }
      }
    }
  }
  std::vector<unsigned char> coverage(n * n, 0);
  for (unsigned z = 1; z + 1 < n; ++z)
  {
    for (unsigned x = 1; x + 1 < n; ++x)
    {
      unsigned sum = 0;
      for (int j = -1; j <= 1; ++j)
      {
        for (int i = -1; i <= 1; ++i)
          sum += Popcount4(mask[(int(z) + j) * n + int(x) + i]);
      }
      coverage[z * n + x] = static_cast<unsigned char>(sum);
    }
  }
  // Clip each constant-alpha mask run against actual receiver triangles. Do not bridge holes or
  // use a floating fallback when terrain is absent.
  for (unsigned z = 0; z < n; ++z)
  {
    for (unsigned x = 0; x < n;)
    {
      const unsigned strength = coverage[z * n + x];
      if (!strength)
      {
        ++x;
        continue;
      }
      const unsigned start = x++;
      while (x < n && coverage[z * n + x] == strength)
        ++x;
      const uint32_t alpha =
          static_cast<uint32_t>(std::lround(opacity * 255.0f * strength / 36.0f));
      if (!alpha)
        continue;
      const float left = x0 + start * dx, right = x0 + x * dx, low = z0 + z * dz, high = low + dz;
      for (const Triangle* t : rows[z])
      {
        if (++m_stats.work > kPixelWorkLimit)
        {
          ++m_stats.skipped;
          return;
        }
        if (std::max({(*t)[0][0], (*t)[1][0], (*t)[2][0]}) < left ||
            std::min({(*t)[0][0], (*t)[1][0], (*t)[2][0]}) > right)
        {
          continue;
        }
        const ClippedPolygon polygon = ClipRectangle(*t, left, low, right, high);
        if (polygon.size() < 3)
          continue;
        const auto vertex = [&](const Point& p) {
          PlanarShadowVertex v;
          v.alpha = alpha;
          v.x = p[0];
          v.y = p[1] + bias + m_options.floorOffset;
          v.z = p[2];
          return v;
        };
        for (size_t k = 1; k + 1 < polygon.size(); ++k)
        {
          if (std::abs(Edge(polygon[0], polygon[k], polygon[k + 1][0], polygon[k + 1][2])) < 1e-12f)
            continue;
          // Bound extra vertex memory independently from input size.
          if (out.size() + 3 > kMaxShadowVertices)
          {
            ++m_stats.skipped;
            return;
          }
          out.push_back(vertex(polygon[0]));
          out.push_back(vertex(polygon[k]));
          out.push_back(vertex(polygon[k + 1]));
          ++m_stats.conformed;
        }
      }
    }
  }
}

bool PlanarShadows::Build(uint32_t id, std::vector<PlanarShadowVertex>& out)
{
  out.clear();
  if (!HasCasters(id))
    return false;
  m_stats = Stats();

  Caster* receiver = FindReceiver(id);
  const Basis basis = ChooseBasis(receiver);
  if (receiver)
    PrepareReceiver(*receiver, basis);
  const std::vector<Caster> clusters = ClusterCasters(id, basis);
  m_stats.clusters = clusters.size();
  m_stats.receiver_triangles = receiver ? receiver->triangles.size() : 0;
  m_stats.up = basis.up;
  {
    float best = -1.0f;
    for (const Caster& c : clusters)
    {
      Point low{1e30f, 1e30f, 1e30f}, high{-1e30f, -1e30f, -1e30f};
      for (const Triangle& t : c.triangles)
      {
        for (const Point& p : t)
        {
          for (unsigned k = 0; k < 3; ++k)
          {
            low[k] = std::min(low[k], p[k]);
            high[k] = std::max(high[k], p[k]);
          }
        }
      }
      const float volume = (high[0] - low[0]) * (high[1] - low[1]) * (high[2] - low[2]);
      if (volume > best)
      {
        best = volume;
        m_stats.largest_extent = {high[0] - low[0], high[1] - low[1], high[2] - low[2]};
      }
    }
  }

  std::vector<Pick> picks;
  for (const Caster& caster : clusters)
  {
    ++m_stats.casters;
    Pick pick;
    pick.caster = &caster;
    if (!LodSelect(caster, pick))
    {
      ++m_stats.lod_culled;
      continue;
    }
    picks.push_back(pick);
  }
  // With the limit on, the nearest casters get the shadow slots first.
  if (m_options.lod != 0)
  {
    std::stable_sort(picks.begin(), picks.end(),
                     [](const Pick& a, const Pick& b) { return a.distance < b.distance; });
  }
  for (const Pick& pick : picks)
  {
    const size_t before = out.size();
    Rasterize(receiver, *pick.caster, pick.resolution, pick.fade, out);
    if (out.size() != before)
      ++m_stats.emitted;
  }
  DropScene(id);
  m_finished.insert(id);
  if (out.empty())
    return false;

  // Back to eye space.
  for (PlanarShadowVertex& v : out)
  {
    const float x = v.x, y = v.y, z = v.z;
    v.x = x * basis.right[0] + y * basis.up[0] + z * basis.forward[0];
    v.y = x * basis.right[1] + y * basis.up[1] + z * basis.forward[1];
    v.z = x * basis.right[2] + y * basis.up[2] + z * basis.forward[2];
  }
  return true;
}
