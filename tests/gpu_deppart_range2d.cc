/*
 * Copyright 2025 Stanford University, NVIDIA
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <vector>
#include <set>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "realm.h"
#include "realm/id.h"
#include "realm/machine.h"
#include "realm/cmdline.h"
#include "philox.h"

using namespace Realm;

#ifdef REALM_USE_CUDA
#include "realm/cuda/cuda_memcpy.h"
#include "realm/cuda/cuda_module.h"
#endif
#ifdef REALM_USE_HIP
#include "hip_cuda_compat/hip_cuda.h"
#include "realm/hip/hip_module.h"
#endif

#ifdef REALM_USE_CUDA
using namespace Realm::Cuda;
#endif
#ifdef REALM_USE_HIP
using namespace Realm::Hip;
#endif

Logger log_app("app");

// ---------------- Config ----------------
namespace TestConfig {
  int    num_nodes     = 1000;   // nodes are a num_nodes x num_nodes 2D grid
  int    num_rects     = 1000;   // number of 2D rectangles (1D index space)
  int    max_rect_size = 10;
  int    num_pieces    = 4;
  int    random        = 1;      // 1=random (matches Range2DTest node behavior)
  unsigned long long seed = 123456789ULL;
  int    show          = 0;
  int    verify        = 1;
};
static const FieldID FID_NODE_SUBGRAPH = 0;                      // nodes: int
static const FieldID FID_RECT_COLOR    = 0;                      // rects: int
static const FieldID FID_RECT_VAL      = sizeof(int);            // rects: Rect<2,int> at offset after color

// ---------------- Helpers ----------------
template <int N, typename T, typename DT, typename Fn>
static void fill_index_space(RegionInstance inst,
                             FieldID fid,
                             const IndexSpace<N,T>& is,
                             Fn gen)
{
  AffineAccessor<DT, N, T> acc(inst, fid);
  for (IndexSpaceIterator<N,T> it(is); it.valid; it.step()) {
    for (PointInRectIterator<N,T> p(it.rect); p.valid; p.step())
      acc[p.p] = gen(p.p);
  }
}

template <int N, typename T, typename DT>
static void copy_field(const IndexSpace<N,T>& is,
                       RegionInstance src, RegionInstance dst, FieldID fid)
{
  std::vector<CopySrcDstField> srcs(1), dsts(1);
  srcs[0].set_field(src, fid, sizeof(DT));
  dsts[0].set_field(dst, fid, sizeof(DT));
  is.copy(srcs, dsts, ProfilingRequestSet()).wait();
}

static void choose_cpu_and_gpu_mems(Memory& cpu_mem, Memory& gpu_mem, bool& have_gpu)
{
  have_gpu = false;
  for (auto mem : Machine::MemoryQuery(Machine::get_machine())) {
    if (!cpu_mem.exists() && (mem.kind() == Memory::SYSTEM_MEM))
      cpu_mem = mem;
    if (!gpu_mem.exists() && (mem.kind() == Memory::GPU_FB_MEM)) {
      gpu_mem = mem;
      have_gpu = true;
    }
  }
}

static Event make_instance(RegionInstance& ri,
                           Memory mem,
                           const IndexSpace<1,int>& is,
                           const std::vector<size_t>& fields)
{
  return RegionInstance::create_instance(ri, mem, is, fields, /*soa=*/0, ProfilingRequestSet());
}

static Event make_instance(RegionInstance& ri,
                           Memory mem,
                           const IndexSpace<2,int>& is,
                           const std::vector<size_t>& fields)
{
  return RegionInstance::create_instance(ri, mem, is, fields, /*soa=*/0, ProfilingRequestSet());
}

template <int N>
static int compare_partitions(const std::vector<IndexSpace<N,int>>& A,
                              const std::vector<IndexSpace<N,int>>& B)
{
  int errors = 0;
  if (A.size() != B.size()) return 1;
  for (size_t i = 0; i < A.size(); i++) {
    for (IndexSpaceIterator<N,int> it(A[i]); it.valid; it.step())
      for (PointInRectIterator<N,int> p(it.rect); p.valid; p.step())
        if (!B[i].contains(p.p)) { errors++; }
    for (IndexSpaceIterator<N,int> it(B[i]); it.valid; it.step())
      for (PointInRectIterator<N,int> p(it.rect); p.valid; p.step())
        if (!A[i].contains(p.p)) { errors++; }
  }
  return errors;
}

template <int N>
static int compare_spaces(const IndexSpace<N,int>& A, const IndexSpace<N,int>& B)
{
  int errors = 0;
  for (IndexSpaceIterator<N,int> it(A); it.valid; it.step())
    for (PointInRectIterator<N,int> p(it.rect); p.valid; p.step())
      if (!B.contains(p.p)) { errors++; }
  for (IndexSpaceIterator<N,int> it(B); it.valid; it.step())
    for (PointInRectIterator<N,int> p(it.rect); p.valid; p.step())
      if (!A.contains(p.p)) { errors++; }
  return errors;
}

// ---------------- Top-level task ----------------
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 302,
};

static void top_level_task(const void*, size_t, const void*, size_t, Processor)
{
  log_app.print() << "deppart_ranges2d_itest starting";

  // Parent spaces
  IndexSpace<2,int> is_nodes(Rect<2,int>(Point<2,int>(0,0),
                                         Point<2,int>(TestConfig::num_nodes - 1,
                                                      TestConfig::num_nodes - 1)));
  IndexSpace<1,int> is_rects(Rect<1,int>(0, TestConfig::num_rects - 1));

  // Memories
  Memory cpu_mem, gpu_mem;
  bool have_gpu = false;
  choose_cpu_and_gpu_mems(cpu_mem, gpu_mem, have_gpu);
  if (!cpu_mem.exists()) {
    log_app.fatal() << "No SYSTEM_MEM found";
    assert(0);
    return;
  }
  if (!have_gpu) {
    log_app.warning() << "No GPU_FB_MEM found; running CPU-only check.";
  }

  // Equal subspaces (num_pieces)
  std::vector<IndexSpace<2,int>> ss_nodes_eq;
  std::vector<IndexSpace<1,int>> ss_rects_eq;
  is_nodes.create_equal_subspaces(TestConfig::num_pieces, 1, ss_nodes_eq, ProfilingRequestSet()).wait();
  is_rects.create_equal_subspaces(TestConfig::num_pieces, 1, ss_rects_eq, ProfilingRequestSet()).wait();

  // Per-piece CPU instances
  std::vector<RegionInstance> cpu_nodes_inst(TestConfig::num_pieces);
  std::vector<RegionInstance> cpu_rects_inst(TestConfig::num_pieces);

  // Field descriptors (CPU), one per piece
  std::vector<FieldDataDescriptor<IndexSpace<2,int>, int>>            cpu_node_ids(TestConfig::num_pieces);
  std::vector<FieldDataDescriptor<IndexSpace<1,int>, int>>            cpu_rect_ids(TestConfig::num_pieces);
  std::vector<FieldDataDescriptor<IndexSpace<1,int>, Rect<2,int>>>    cpu_rect_vals(TestConfig::num_pieces);

  for (int i = 0; i < TestConfig::num_pieces; i++) {
    make_instance(cpu_nodes_inst[i], cpu_mem, ss_nodes_eq[i], {sizeof(int)}).wait();
    make_instance(cpu_rects_inst[i], cpu_mem, ss_rects_eq[i],
                  {sizeof(int), sizeof(Rect<2,int>)}).wait();

    cpu_node_ids[i].index_space  = ss_nodes_eq[i];
    cpu_node_ids[i].inst         = cpu_nodes_inst[i];
    cpu_node_ids[i].field_offset = FID_NODE_SUBGRAPH;

    cpu_rect_ids[i].index_space  = ss_rects_eq[i];
    cpu_rect_ids[i].inst         = cpu_rects_inst[i];
    cpu_rect_ids[i].field_offset = FID_RECT_COLOR;

    cpu_rect_vals[i].index_space  = ss_rects_eq[i];
    cpu_rect_vals[i].inst         = cpu_rects_inst[i];
    cpu_rect_vals[i].field_offset = FID_RECT_VAL;
  }

  // Data init (piecewise)
  auto rand_int = [](unsigned long long seed, int ctr, int stream, int bound) -> int {
    return Philox_2x32<>::rand_int(seed, ctr, stream, bound);
  };

  auto gen_node_id = [&](Point<2,int> p)->int {
    // Flatten idx like Range2DTest
    int width = TestConfig::num_nodes;
    int idx = p[0] * width + p[1];
    if (TestConfig::random)
      return rand_int(TestConfig::seed, idx, /*stream=*/0, TestConfig::num_pieces);
    else
      return int((long long)idx * TestConfig::num_pieces / (TestConfig::num_nodes * TestConfig::num_nodes));
  };

  auto gen_rect_color = [&](Point<1,int> p)->int {
    if (TestConfig::random)
      return rand_int(TestConfig::seed, p[0], /*stream=*/0, TestConfig::num_pieces);
    else
      return int((long long)p[0] * TestConfig::num_pieces / TestConfig::num_rects);
  };

  auto gen_rect_val = [&](Point<1,int> p)->Rect<2,int> {
    int x      = rand_int(TestConfig::seed, p[0],     /*stream=*/0, TestConfig::num_nodes);
    int y      = rand_int(TestConfig::seed, p[0] + 1, /*stream=*/0, TestConfig::num_nodes);
    int length = rand_int(TestConfig::seed, p[0] + 2, /*stream=*/0, TestConfig::max_rect_size);
    int height = rand_int(TestConfig::seed, p[0] + 3, /*stream=*/0, TestConfig::max_rect_size);
    return Rect<2,int>(Point<2,int>(x, y), Point<2,int>(x + length, y + height)); // matches Range2DTest (no clamping)
  };

  for (int i = 0; i < TestConfig::num_pieces; i++) {
    fill_index_space<2,int,int>(cpu_nodes_inst[i], FID_NODE_SUBGRAPH, ss_nodes_eq[i], gen_node_id);
    fill_index_space<1,int,int>(cpu_rects_inst[i], FID_RECT_COLOR, ss_rects_eq[i], gen_rect_color);
    fill_index_space<1,int,Rect<2,int>>(cpu_rects_inst[i], FID_RECT_VAL, ss_rects_eq[i], gen_rect_val);
  }

  if (TestConfig::show) {
    for (int i = 0; i < TestConfig::num_pieces; i++) {
      AffineAccessor<int,2,int> acc_node(cpu_nodes_inst[i], FID_NODE_SUBGRAPH);
      for (IndexSpaceIterator<2,int> it(ss_nodes_eq[i]); it.valid; it.step())
        for (PointInRectIterator<2,int> p(it.rect); p.valid; p.step())
          log_app.print() << "node_id[" << p.p << "]=" << acc_node[p.p];

      AffineAccessor<int,1,int> acc_rc(cpu_rects_inst[i], FID_RECT_COLOR);
      AffineAccessor<Rect<2,int>,1,int> acc_rv(cpu_rects_inst[i], FID_RECT_VAL);
      for (IndexSpaceIterator<1,int> it(ss_rects_eq[i]); it.valid; it.step())
        for (PointInRectIterator<1,int> p(it.rect); p.valid; p.step())
          log_app.print() << "rect_id[" << p.p << "]=" << acc_rc[p.p]
                          << " rect_val=" << acc_rv[p.p];
    }
  }

  // Colors
  std::vector<int> colors(TestConfig::num_pieces);
  for (int i = 0; i < TestConfig::num_pieces; i++) colors[i] = i;

  // -------- CPU partitioning --------
  std::vector<IndexSpace<1,int>> p_cpu_colored_rects;
  std::vector<IndexSpace<2,int>> p_cpu_rects, p_cpu_intersect, p_cpu_diff;
  IndexSpace<2,int> cpu_union;

  Event e_cpu_byfield = is_rects.create_subspaces_by_field(cpu_rect_ids, colors,
                                                           p_cpu_colored_rects, ProfilingRequestSet());
  Event e_cpu_image   = is_nodes.create_subspaces_by_image(cpu_rect_vals,
                                                           p_cpu_colored_rects,
                                                           p_cpu_rects,
                                                           ProfilingRequestSet(), e_cpu_byfield);
  Event e_cpu_union   = IndexSpace<2,int>::compute_union(p_cpu_rects, cpu_union,
                                                         ProfilingRequestSet(), e_cpu_image);

  // Build rotated RHS and compute intersections/differences
  std::vector<IndexSpace<2,int>> cpu_rhs;
  Event e_cpu_prev = e_cpu_union;
  if (!p_cpu_rects.empty()) {
    cpu_rhs = p_cpu_rects;
    auto last = cpu_rhs.back();
    cpu_rhs.insert(cpu_rhs.begin(), last);
    cpu_rhs.pop_back();
    Event e_cpu_isect = IndexSpace<2,int>::compute_intersections(p_cpu_rects, cpu_rhs,
                                                                 p_cpu_intersect,
                                                                 ProfilingRequestSet(), e_cpu_prev);
    Event e_cpu_diff  = IndexSpace<2,int>::compute_differences  (p_cpu_rects, cpu_rhs,
                                                                 p_cpu_diff,
                                                                 ProfilingRequestSet(), e_cpu_isect);
    e_cpu_prev = e_cpu_diff;
  }

  // -------- GPU partitioning --------
  std::vector<IndexSpace<1,int>> p_gpu_colored_rects;
  std::vector<IndexSpace<2,int>> p_gpu_rects, p_gpu_intersect, p_gpu_diff;
  IndexSpace<2,int> gpu_union;

  if (have_gpu) {
    // Per-piece GPU rect instances (layout matches CPU: [int, Rect<2>])
    std::vector<RegionInstance> gpu_rects_inst(TestConfig::num_pieces);
    std::vector<FieldDataDescriptor<IndexSpace<1,int>, int>>         gpu_rect_ids(TestConfig::num_pieces);
    std::vector<FieldDataDescriptor<IndexSpace<1,int>, Rect<2,int>>> gpu_rect_vals(TestConfig::num_pieces);

    for (int i = 0; i < TestConfig::num_pieces; i++) {
      make_instance(gpu_rects_inst[i], gpu_mem, ss_rects_eq[i],
                    {sizeof(int), sizeof(Rect<2,int>)}).wait();
      // Copy CPU -> GPU per subspace
      copy_field<1,int,int>(ss_rects_eq[i], cpu_rects_inst[i], gpu_rects_inst[i], FID_RECT_COLOR);
      copy_field<1,int,Rect<2,int>>(ss_rects_eq[i], cpu_rects_inst[i], gpu_rects_inst[i], FID_RECT_VAL);

      // Describe fields
      gpu_rect_ids[i].index_space  = ss_rects_eq[i];
      gpu_rect_ids[i].inst         = gpu_rects_inst[i];
      gpu_rect_ids[i].field_offset = FID_RECT_COLOR;

      gpu_rect_vals[i].index_space  = ss_rects_eq[i];
      gpu_rect_vals[i].inst         = gpu_rects_inst[i];
      gpu_rect_vals[i].field_offset = FID_RECT_VAL;
    }

    // Optional warm-up (mirrors class)
    setenv("GPU_UNION", "1", 1);
    setenv("GPU_DIFFERENCE", "1", 1);
    setenv("GPU_INTERSECTION", "1", 1);

    std::vector<IndexSpace<1,int>> p_garbage_colors;
    std::vector<IndexSpace<2,int>> p_garbage_rects;
    Event e_w1 = is_rects.create_subspaces_by_field(gpu_rect_ids, colors,
                                                    p_garbage_colors, ProfilingRequestSet());
    Event e_w2 = is_nodes.create_subspaces_by_image(gpu_rect_vals, p_garbage_colors,
                                                    p_garbage_rects, ProfilingRequestSet(), e_w1);
    IndexSpace<2,int> garbage_union;
    Event e_w3 = IndexSpace<2,int>::compute_union(p_garbage_rects, garbage_union,
                                                  ProfilingRequestSet(), e_w2);
    std::vector<IndexSpace<2,int>> p_garbage_rhs = p_garbage_rects;
    if (!p_garbage_rhs.empty()) {
      auto last = p_garbage_rhs.back();
      p_garbage_rhs.insert(p_garbage_rhs.begin(), last);
      p_garbage_rhs.pop_back();
      std::vector<IndexSpace<2,int>> p_garbage_intersect, p_garbage_diff;
      Event e_w4 = IndexSpace<2,int>::compute_intersections(p_garbage_rects, p_garbage_rhs,
                                                            p_garbage_intersect, ProfilingRequestSet(), e_w3);
      Event e_w5 = IndexSpace<2,int>::compute_differences(p_garbage_rects, p_garbage_rhs,
                                                          p_garbage_diff, ProfilingRequestSet(), e_w4);
      e_w5.wait();
    } else {
      e_w3.wait();
    }

    // Actual GPU pipeline
    Event e_gpu_byfield = is_rects.create_subspaces_by_field(gpu_rect_ids, colors,
                                                             p_gpu_colored_rects, ProfilingRequestSet());
    Event e_gpu_image   = is_nodes.create_subspaces_by_image(gpu_rect_vals, p_gpu_colored_rects,
                                                             p_gpu_rects, ProfilingRequestSet(), e_gpu_byfield);
    Event e_gpu_union   = IndexSpace<2,int>::compute_union(p_gpu_rects, gpu_union,
                                                           ProfilingRequestSet(), e_gpu_image);

    std::vector<IndexSpace<2,int>> gpu_rhs;
    Event e_gpu_prev = e_gpu_union;
    if (!p_gpu_rects.empty()) {
      gpu_rhs = p_gpu_rects;
      auto last = gpu_rhs.back();
      gpu_rhs.insert(gpu_rhs.begin(), last);
      gpu_rhs.pop_back();
      Event e_gpu_isect = IndexSpace<2,int>::compute_intersections(p_gpu_rects, gpu_rhs,
                                                                   p_gpu_intersect, ProfilingRequestSet(), e_gpu_prev);
      Event e_gpu_diff  = IndexSpace<2,int>::compute_differences  (p_gpu_rects, gpu_rhs,
                                                                   p_gpu_diff, ProfilingRequestSet(), e_gpu_isect);
      e_gpu_prev = e_gpu_diff;
    }

    // Wait on CPU and GPU before comparison
    e_cpu_prev.wait();
    e_gpu_prev.wait();

    // Compare CPU vs GPU
    if (TestConfig::verify) {
      int errs = 0;
      errs += compare_partitions<1>(p_cpu_colored_rects, p_gpu_colored_rects);
      errs += compare_partitions<2>(p_cpu_rects,        p_gpu_rects);
      errs += compare_partitions<2>(p_cpu_intersect,    p_gpu_intersect);
      errs += compare_partitions<2>(p_cpu_diff,         p_gpu_diff);
      errs += compare_spaces<2>(cpu_union, gpu_union);
      if (errs) {
        log_app.fatal() << "Mismatch between CPU and GPU partitions, errors=" << errs;
        assert(0);
      }
    }

    // Cleanup GPU
    for (int i = 0; i < TestConfig::num_pieces; i++)
      gpu_rects_inst[i].destroy();

    unsetenv("GPU_UNION");
    unsetenv("GPU_DIFFERENCE");
    unsetenv("GPU_INTERSECTION");
  } else {
    e_cpu_prev.wait();
  }

  // Cleanup CPU
  for (int i = 0; i < TestConfig::num_pieces; i++) {
    cpu_nodes_inst[i].destroy();
    cpu_rects_inst[i].destroy();
  }
  is_nodes.destroy();
  is_rects.destroy();

  log_app.print() << "deppart_ranges2d_itest: PASS";
}

// ---------------- Main ----------------
int main(int argc, char** argv)
{
  Runtime rt;
  rt.init(&argc, &argv);

  CommandLineParser cp;
  cp.add_option_int("-n",      TestConfig::num_nodes)
    .add_option_int("-r",      TestConfig::num_rects)
    .add_option_int("-m",      TestConfig::max_rect_size)
    .add_option_int("-p",      TestConfig::num_pieces)
    .add_option_int("-random", TestConfig::random)
    .add_option_int("-show",   TestConfig::show)
    .add_option_int("-verify", TestConfig::verify);
  bool ok = cp.parse_command_line(argc, const_cast<const char**>(argv));
  assert(ok);

  rt.register_task(TOP_LEVEL_TASK, top_level_task);

  Processor p = Machine::ProcessorQuery(Machine::get_machine())
                  .only_kind(Processor::LOC_PROC)
                  .first();
  assert(p.exists());

  Event e = rt.collective_spawn(p, TOP_LEVEL_TASK, nullptr, 0);
  rt.shutdown(e);
  rt.wait_for_shutdown();
  return 0;
}
