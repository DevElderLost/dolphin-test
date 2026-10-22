// Copyright 2026 Dolphin Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "VideoCommon/PlanarShadowManager.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstring>

#include "Common/CommonTypes.h"
#include "Common/Logging/Log.h"

#include "VideoCommon/AbstractFramebuffer.h"
#include "VideoCommon/AbstractGfx.h"
#include "VideoCommon/AbstractPipeline.h"
#include "VideoCommon/AbstractShader.h"
#include "VideoCommon/BPMemory.h"
#include "VideoCommon/ConstantManager.h"
#include "VideoCommon/FramebufferManager.h"
#include "VideoCommon/FramebufferShaderGen.h"
#include "VideoCommon/NativeVertexFormat.h"
#include "VideoCommon/ShaderCache.h"
#include "VideoCommon/VertexManagerBase.h"
#include "VideoCommon/VideoConfig.h"
#include "VideoCommon/XFMemory.h"

namespace
{
// Draws with more vertices than this are not looked at, the decode runs on the GPU thread.
constexpr u32 kMaxObservedVertices = 30000;
// Triangles taken from a single draw.
constexpr u32 kMaxTrianglesPerDraw = 30000;
// Vertices per utility upload. A multiple of three.
constexpr u32 kUploadChunk = 30000;
// Shadow geometry closer to the camera plane than this is clipped away.
constexpr float kMinW = 1e-4f;
constexpr u16 kPrimitiveRestartIndex = 0xFFFF;

float SignOf(float v)
{
  return static_cast<float>((v > 0.0f) - (v < 0.0f));
}

struct Clip
{
  float v[4];
};
}  // namespace

bool PlanarShadowManager::SceneState::operator==(const SceneState& other) const
{
  const auto& a = viewport;
  const auto& b = other.viewport;
  return projection == other.projection &&
         pixel_center_correction == other.pixel_center_correction &&
         a.scissor_rect.left == b.scissor_rect.left && a.scissor_rect.top == b.scissor_rect.top &&
         a.scissor_rect.right == b.scissor_rect.right &&
         a.scissor_rect.bottom == b.scissor_rect.bottom && a.viewport_x == b.viewport_x &&
         a.viewport_y == b.viewport_y && a.viewport_width == b.viewport_width &&
         a.viewport_height == b.viewport_height &&
         a.viewport_near_depth == b.viewport_near_depth &&
         a.viewport_far_depth == b.viewport_far_depth;
}

PlanarShadowManager::PlanarShadowManager() = default;
PlanarShadowManager::~PlanarShadowManager() = default;

void PlanarShadowManager::RefreshOptions()
{
  static constexpr unsigned kResolutions[3] = {64, 96, 192};
  PlanarShadowOptions o;
  o.enabled = g_ActiveConfig.bPlanarShadows;
  o.resolution = kResolutions[std::clamp(g_ActiveConfig.iPlanarShadowsQuality, 0, 2)];
  o.lod = static_cast<unsigned>(std::clamp(g_ActiveConfig.iPlanarShadowsDistance, 0, 3));
  o.opacity = std::clamp(g_ActiveConfig.iPlanarShadowsOpacity, 0, 60) * 0.01f;
  o.directionX = std::clamp(g_ActiveConfig.iPlanarShadowsDirectionX, -200, 200) * 0.01f;
  o.directionZ = std::clamp(g_ActiveConfig.iPlanarShadowsDirectionZ, -200, 200) * 0.01f;
  o.floorOffset = 0.0f;
  o.auto_up = g_ActiveConfig.iPlanarShadowsUpMode == 0;
  m_shadows.SetOptions(o);
  if (!o.enabled)
  {
    m_shadows.Reset();
    m_scenes.clear();
    m_has_pending = false;
  }
}

void PlanarShadowManager::EndFrame()
{
  // Whatever is left was not followed by a 2D draw or an EFB copy, it is not part of a visible scene.
  m_shadows.Reset();
  m_scenes.clear();
  m_has_pending = false;
  RefreshOptions();
}

u32 PlanarShadowManager::FindOrAddScene(const SceneState& state)
{
  for (size_t i = 0; i < m_scenes.size(); ++i)
  {
    if (m_scenes[i] == state)
      return static_cast<u32>(i);
  }
  if (m_scenes.size() >= kMaxScenes)
    return kInvalidScene;
  m_scenes.push_back(state);
  return static_cast<u32>(m_scenes.size() - 1);
}

void PlanarShadowManager::ObserveBatch(const u8* vertices, u32 num_vertices,
                                       const PortableVertexDeclaration& decl, const u16* indices,
                                       u32 num_indices, PrimitiveType primitive,
                                       const VertexShaderConstants& constants)
{
  if (m_injecting || !m_shadows.Options().enabled)
    return;
  if (primitive != PrimitiveType::Triangles && primitive != PrimitiveType::TriangleStrip)
    return;
  if (num_vertices < 3 || num_vertices > kMaxObservedVertices || num_indices < 3 ||
      vertices == nullptr || indices == nullptr)
  {
    return;
  }
  // Only opaque, depth writing draws of a perspective 3D scene are of interest.
  if (xfmem.projection.type != ProjectionType::Perspective)
    return;
  if (!bpmem.zmode.test_enable.Value() || !bpmem.zmode.update_enable.Value() ||
      bpmem.blendmode.blend_enable.Value() || bpmem.ztex2.op != ZTexOp::Disabled)
  {
    return;
  }
  // The native vertex position is always float, 2 or 3 components.
  if (!decl.position.enable || decl.position.type != ComponentFormat::Float ||
      decl.position.components < 2 || decl.stride <= 0)
  {
    return;
  }
  const size_t stride = static_cast<size_t>(decl.stride);
  const bool per_vertex_matrix = decl.posmtx.enable;

  SceneState scene_state;
  static_assert(sizeof(scene_state.projection) == sizeof(constants.projection));
  std::memcpy(scene_state.projection.data(), constants.projection.data(),
              sizeof(scene_state.projection));
  scene_state.pixel_center_correction = constants.pixelcentercorrection;
  scene_state.viewport = g_gfx->GetViewportAndScissor();
  scene_state.depth_func = static_cast<u32>(bpmem.zmode.func.Value());
  const u32 scene_id = FindOrAddScene(scene_state);
  if (scene_id == kInvalidScene)
    return;
  if (per_vertex_matrix ? !m_shadows.WantsCasters(scene_id) : !m_shadows.WantsReceivers(scene_id))
    return;

  // Vertices to eye space, the same way the vertex shader does it.
  m_points.resize(num_vertices);
  u64 matrix_mask = 0;
  const int components = std::min(decl.position.components, 3);
  for (u32 i = 0; i < num_vertices; ++i)
  {
    const u8* vertex = vertices + i * stride;
    float pos[3] = {0.0f, 0.0f, 0.0f};
    std::memcpy(pos, vertex + decl.position.offset, sizeof(float) * components);

    float eye[3];
    if (per_vertex_matrix)
    {
      // Matrix indices count rows of four floats, a matrix takes three rows.
      const u32 index = vertex[decl.posmtx.offset] & 0x3f;
      matrix_mask |= u64(1) << index;
      for (u32 r = 0; r < 3; ++r)
      {
        const u32 row = ((index + r) * 4) & 0xff;
        eye[r] = pos[0] * xfmem.posMatrices[row] + pos[1] * xfmem.posMatrices[row + 1] +
                 pos[2] * xfmem.posMatrices[row + 2] + xfmem.posMatrices[row + 3];
      }
    }
    else
    {
      for (u32 r = 0; r < 3; ++r)
      {
        const float4& row = constants.posnormalmatrix[r];
        eye[r] = pos[0] * row[0] + pos[1] * row[1] + pos[2] * row[2] + row[3];
      }
    }
    for (u32 k = 0; k < 3; ++k)
    {
      if (!std::isfinite(eye[k]) || std::abs(eye[k]) > 1e7f)
        return;
      m_points[i][k] = eye[k];
    }
  }

  // Skinned meshes use more than one position matrix inside a single draw. Draws with a per vertex
  // matrix that only use one are body parts of unknown kind, they are neither caster nor receiver.
  const bool skinned = per_vertex_matrix && std::popcount(matrix_mask) >= 2;
  if (per_vertex_matrix && !skinned)
    return;

  // Assemble triangles.
  m_triangles.clear();
  const auto add = [&](u16 a, u16 b, u16 c) {
    if (a >= num_vertices || b >= num_vertices || c >= num_vertices)
      return;
    m_triangles.push_back(PlanarShadows::Triangle{m_points[a], m_points[b], m_points[c]});
  };
  if (primitive == PrimitiveType::Triangles)
  {
    for (u32 i = 0; i + 2 < num_indices && m_triangles.size() < kMaxTrianglesPerDraw; i += 3)
      add(indices[i], indices[i + 1], indices[i + 2]);
  }
  else
  {
    u32 run = 0;
    for (u32 i = 0; i < num_indices && m_triangles.size() < kMaxTrianglesPerDraw; ++i)
    {
      if (indices[i] == kPrimitiveRestartIndex)
      {
        run = 0;
        continue;
      }
      if (++run >= 3)
        add(indices[i - 2], indices[i - 1], indices[i]);
    }
  }
  if (m_triangles.empty())
    return;

  PlanarShadows::Scene scene;
  scene.id = scene_id;
  scene.cot = std::abs(constants.projection[1][1]);
  if (skinned)
  {
    m_shadows.AddCaster(scene, m_triangles);
    m_has_pending = true;
  }
  else
  {
    m_shadows.AddReceiver(scene, m_triangles);
  }
}

void PlanarShadowManager::Inject()
{
  if (m_injecting || !m_has_pending)
    return;
  // Everything queued so far belongs to the scene and has to be drawn first.
  g_vertex_manager->Flush();

  m_has_pending = false;
  m_injecting = true;
  for (const u32 id : m_shadows.PendingScenes())
  {
    if (id >= m_scenes.size())
    {
      m_shadows.DropScene(id);
      continue;
    }
    const SceneState scene = m_scenes[id];
    if (!m_shadows.Build(id, m_shadow_vertices))
      continue;
    ProjectShadow(scene, m_shadow_vertices);
    DrawVertices(scene, scene.depth_func);
  }
  m_injecting = false;
}

// Reproduces the position output of the GX vertex shader (VertexShaderGen.cpp) for eye space
// shadow vertices, then divides by w. The result is handed to the EFB poke vertex shader, which
// passes it through, so the depth values match the scene's own and the depth test works.
void PlanarShadowManager::ProjectShadow(const SceneState& scene,
                                        const std::vector<PlanarShadowVertex>& shadow)
{
  m_draw_vertices.clear();
  const auto& pcc = scene.pixel_center_correction;
  const auto transform = [&](const PlanarShadowVertex& v) {
    Clip c;
    for (u32 i = 0; i < 4; ++i)
    {
      c.v[i] = scene.projection[i * 4 + 0] * v.x + scene.projection[i * 4 + 1] * v.y +
               scene.projection[i * 4 + 2] * v.z + scene.projection[i * 4 + 3];
    }
    c.v[2] *= 1.0f - 1e-7f;
    c.v[2] = c.v[3] * pcc[3] - c.v[2] * pcc[2];
    if (!g_backend_info.bSupportsClipControl)
      c.v[2] = c.v[2] * 2.0f - c.v[3];
    c.v[0] *= SignOf(pcc[0]);
    c.v[1] *= SignOf(-pcc[1]);
    c.v[0] -= c.v[3] * pcc[0];
    c.v[1] -= c.v[3] * pcc[1];
    return c;
  };
  const auto emit = [&](const Clip& c, u32 alpha) {
    DrawVertex d;
    d.position[0] = c.v[0] / c.v[3];
    d.position[1] = c.v[1] / c.v[3];
    d.position[2] = c.v[2] / c.v[3];
    d.position[3] = 1.0f;
    d.color[0] = 0;
    d.color[1] = 0;
    d.color[2] = 0;
    d.color[3] = static_cast<u8>(std::min<u32>(alpha, 255));
    m_draw_vertices.push_back(d);
  };

  for (size_t i = 0; i + 2 < shadow.size(); i += 3)
  {
    // Clip the triangle against the w > kMinW plane, which leaves at most four vertices.
    Clip in[3] = {transform(shadow[i]), transform(shadow[i + 1]), transform(shadow[i + 2])};
    Clip polygon[4];
    size_t count = 0;
    for (size_t k = 0; k < 3; ++k)
    {
      const Clip& a = in[k];
      const Clip& b = in[(k + 1) % 3];
      const bool a_inside = a.v[3] > kMinW;
      const bool b_inside = b.v[3] > kMinW;
      if (a_inside)
        polygon[count++] = a;
      if (a_inside != b_inside)
      {
        const float t = (kMinW - a.v[3]) / (b.v[3] - a.v[3]);
        Clip hit;
        for (u32 j = 0; j < 4; ++j)
          hit.v[j] = a.v[j] + t * (b.v[j] - a.v[j]);
        hit.v[3] = kMinW;
        polygon[count++] = hit;
      }
    }
    if (count < 3)
      continue;
    const u32 alpha = shadow[i].alpha;
    for (size_t k = 1; k + 1 < count; ++k)
    {
      emit(polygon[0], alpha);
      emit(polygon[k], alpha);
      emit(polygon[k + 1], alpha);
    }
  }
}

bool PlanarShadowManager::EnsurePipeline(u32 depth_func)
{
  if (m_pipeline_failed || !g_gfx || !g_shader_cache || !g_framebuffer_manager)
    return false;

  const AbstractShader* pixel_shader = g_shader_cache->GetColorPixelShader();
  const AbstractShader* geometry_shader =
      g_framebuffer_manager->IsEFBStereo() ? g_shader_cache->GetColorGeometryShader() : nullptr;
  if (!pixel_shader)
    return false;
  const FramebufferState framebuffer_state = g_framebuffer_manager->GetEFBFramebufferState();

  // The EFB format, MSAA mode or the shaders it depends on changed: start over.
  if (pixel_shader != m_pipeline_pixel_shader || geometry_shader != m_pipeline_geometry_shader ||
      framebuffer_state.hex != m_pipeline_framebuffer_state)
  {
    DestroyPipelines();
    m_pipeline_pixel_shader = pixel_shader;
    m_pipeline_geometry_shader = geometry_shader;
    m_pipeline_framebuffer_state = framebuffer_state.hex;
  }

  if (!m_vertex_format)
  {
    // The same layout as the EFB poke vertices: a float4 position and an RGBA8 color.
    PortableVertexDeclaration vtx_decl = {};
    vtx_decl.position.enable = true;
    vtx_decl.position.type = ComponentFormat::Float;
    vtx_decl.position.components = 4;
    vtx_decl.position.integer = false;
    vtx_decl.position.offset = offsetof(DrawVertex, position);
    vtx_decl.colors[0].enable = true;
    vtx_decl.colors[0].type = ComponentFormat::UByte;
    vtx_decl.colors[0].components = 4;
    vtx_decl.colors[0].integer = false;
    vtx_decl.colors[0].offset = offsetof(DrawVertex, color);
    vtx_decl.stride = sizeof(DrawVertex);
    m_vertex_format = g_gfx->CreateNativeVertexFormat(vtx_decl);
    if (!m_vertex_format)
    {
      m_pipeline_failed = true;
      return false;
    }
  }
  if (!m_vertex_shader)
  {
    // Passes the position through as normalized device coordinates.
    m_vertex_shader = g_gfx->CreateShaderFromSource(
        ShaderStage::Vertex, FramebufferShaderGen::GenerateEFBPokeVertexShader(), nullptr,
        "Planar shadow vertex shader");
    if (!m_vertex_shader)
    {
      m_pipeline_failed = true;
      return false;
    }
  }

  std::unique_ptr<AbstractPipeline>& pipeline = m_pipelines[depth_func & 7];
  if (!pipeline)
  {
    AbstractPipelineConfig config = {};
    config.vertex_format = m_vertex_format.get();
    config.vertex_shader = m_vertex_shader.get();
    config.geometry_shader = geometry_shader;
    config.pixel_shader = pixel_shader;
    config.rasterization_state = RenderState::GetNoCullRasterizationState(PrimitiveType::Triangles);

    // Tested against the scene's depth with the scene's compare function, never written.
    config.depth_state = RenderState::GetNoDepthTestingDepthState();
    config.depth_state.test_enable = true;
    config.depth_state.update_enable = false;
    config.depth_state.func = static_cast<CompareMode>(depth_func & 7);

    // Black with the vertex alpha as strength, the framebuffer alpha stays as it is.
    config.blending_state = RenderState::GetNoBlendingBlendState();
    config.blending_state.blend_enable = true;
    config.blending_state.src_factor = SrcBlendFactor::SrcAlpha;
    config.blending_state.dst_factor = DstBlendFactor::InvSrcAlpha;
    config.blending_state.src_factor_alpha = SrcBlendFactor::Zero;
    config.blending_state.dst_factor_alpha = DstBlendFactor::One;
    config.blending_state.color_update = true;
    config.blending_state.alpha_update = false;
    config.blending_state.use_dual_src = false;

    config.framebuffer_state = framebuffer_state;
    config.usage = AbstractPipelineUsage::Utility;
    pipeline = g_gfx->CreatePipeline(config);
    if (!pipeline)
    {
      m_pipeline_failed = true;
      return false;
    }
  }
  return true;
}

void PlanarShadowManager::DestroyPipelines()
{
  for (auto& pipeline : m_pipelines)
    pipeline.reset();
}

void PlanarShadowManager::DrawVertices(const SceneState& scene, u32 depth_func)
{
  if (m_draw_vertices.empty() || !EnsurePipeline(depth_func))
    return;

  const AbstractGfx::ViewportAndScissor& vp = scene.viewport;
  g_gfx->BeginUtilityDrawing();
  for (size_t first = 0; first < m_draw_vertices.size(); first += kUploadChunk)
  {
    const u32 count =
        static_cast<u32>(std::min<size_t>(kUploadChunk, m_draw_vertices.size() - first));
    u32 base_vertex = 0;
    u32 base_index = 0;
    g_vertex_manager->UploadUtilityVertices(m_draw_vertices.data() + first, sizeof(DrawVertex),
                                            count, nullptr, 0, &base_vertex, &base_index);

    // The viewport, depth range and scissor of the scene the casters were drawn in.
    g_gfx->SetViewport(vp.viewport_x, vp.viewport_y, vp.viewport_width, vp.viewport_height,
                       vp.viewport_near_depth, vp.viewport_far_depth);
    g_gfx->SetScissorRect(vp.scissor_rect);
    g_gfx->SetPipeline(m_pipelines[depth_func & 7].get());
    g_gfx->Draw(base_vertex, count);
  }
  // Restores the EFB framebuffer, viewport and scissor of the GX state.
  g_gfx->EndUtilityDrawing();
}
