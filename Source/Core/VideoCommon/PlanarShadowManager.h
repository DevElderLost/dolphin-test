// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <memory>
#include <vector>

#include "Common/CommonTypes.h"
#include "VideoCommon/AbstractGfx.h"
#include "VideoCommon/PlanarShadows.h"
#include "VideoCommon/RenderState.h"

class AbstractPipeline;
class AbstractShader;
class NativeVertexFormat;
struct PortableVertexDeclaration;
struct VertexShaderConstants;

// Glue between the GX draw path and PlanarShadows (see PlanarShadows.h).
//
// Draw calls are observed when the vertex manager flushes them, where the native vertices, the
// position matrices and the projection constants of the draw are all current. The shadows are
// drawn right before the first orthographic (2D) draw that follows a 3D scene, or before an EFB
// copy, as an ordinary alpha blended triangle list through the utility pipeline.
class PlanarShadowManager
{
public:
  PlanarShadowManager();
  ~PlanarShadowManager();

  // Reads the options from the active config.
  void RefreshOptions();

  bool IsEnabled() const { return m_shadows.Options().enabled; }
  bool HasPending() const { return m_has_pending; }

  // Called by the vertex manager before a batch is committed, with the batch's CPU side data.
  void ObserveBatch(const u8* vertices, u32 num_vertices, const PortableVertexDeclaration& decl,
                    const u16* indices, u32 num_indices, PrimitiveType primitive,
                    const VertexShaderConstants& constants);

  // Draws the shadows of everything that was collected. The vertex manager must be flushed.
  void Inject();

  // Drops what is left of the frame and picks up option changes.
  void EndFrame();

private:
  // Everything needed to reproduce a draw's vertex transform and rasterizer state.
  struct SceneState
  {
    std::array<float, 16> projection;
    std::array<float, 4> pixel_center_correction;
    AbstractGfx::ViewportAndScissor viewport;
    u32 depth_func;
    bool operator==(const SceneState& other) const;
  };

  struct DrawVertex
  {
    float position[4];
    u8 color[4];
  };

  static constexpr size_t kMaxScenes = 8;
  static constexpr u32 kInvalidScene = 0xFFFFFFFFu;

  u32 FindOrAddScene(const SceneState& state);
  bool EnsurePipeline(u32 depth_func);
  void DestroyPipelines();
  void ProjectShadow(const SceneState& scene, const std::vector<PlanarShadowVertex>& shadow);
  void DrawVertices(const SceneState& scene, u32 depth_func);

  PlanarShadows m_shadows;
  bool m_has_pending = false;
  bool m_injecting = false;

  std::vector<SceneState> m_scenes;

  // Scratch buffers.
  std::vector<PlanarShadows::Point> m_points;
  std::vector<PlanarShadows::Triangle> m_triangles;
  std::vector<PlanarShadowVertex> m_shadow_vertices;
  std::vector<DrawVertex> m_draw_vertices;

  // Pipelines, one per depth compare function.
  std::unique_ptr<NativeVertexFormat> m_vertex_format;
  std::unique_ptr<AbstractShader> m_vertex_shader;
  std::array<std::unique_ptr<AbstractPipeline>, 8> m_pipelines;
  const AbstractShader* m_pipeline_pixel_shader = nullptr;
  const AbstractShader* m_pipeline_geometry_shader = nullptr;
  u32 m_pipeline_framebuffer_state = 0;
  bool m_pipeline_failed = false;
};
