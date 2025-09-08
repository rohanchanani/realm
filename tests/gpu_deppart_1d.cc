/*
 * Copyright 2025 Stanford University, NVIDIA
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cassert>
#include <vector>
#include <set>
#include <cstdio>
#include <cstring>
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

// ---------------- Config (matches transpose_test style) ----------------
namespace TestConfig {
  int    num_nodes   = 1000;
  int    num_edges   = 5000;
  int    num_pieces  = 4;
  int    random      = 0;           // 0 deterministic, 1 random
  unsigned long long seed = 123456789ULL;
  int    show        = 0;           // print assigned ids
  int    verify      = 1;           // do correctness check
};
static const FieldID FID_SUBGRAPH = 0;
static const FieldID FID_SRC = 0;
static const FieldID FID_DST = sizeof(Point<1, int>);

// ---------------- Small helpers (same idioms as transpose_test) --------
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

// For brevity, we use the simple vector<size_t> layout helper (as in many Realm tests)
static Event make_instance(RegionInstance& ri,
                           Memory mem,
                           const IndexSpace<1,int>& is,
                           std::vector<size_t> fields)
{
  return RegionInstance::create_instance(ri, mem, is, fields,
                                         /*soa=*/0, ProfilingRequestSet());
}

// Compare two partitions index-space-by-index-space
static int compare_partitions(const std::vector<IndexSpace<1,int>>& A,
                              const std::vector<IndexSpace<1,int>>& B)
{
  int errors = 0;
  if (A.size() != B.size()) return 1;
  for (size_t i = 0; i < A.size(); i++) {
    // Check A minus B
    for (IndexSpaceIterator<1,int> it(A[i]); it.valid; it.step())
      for (PointInRectIterator<1,int> p(it.rect); p.valid; p.step())
        if (!B[i].contains(p.p)) { errors++; }
    // Check B minus A
    for (IndexSpaceIterator<1,int> it(B[i]); it.valid; it.step())
      for (PointInRectIterator<1,int> p(it.rect); p.valid; p.step())
        if (!A[i].contains(p.p)) { errors++; }
  }
  return errors;
}

// ---------------- Top-level task (like transpose_test_gpu) --------------
enum {
  TOP_LEVEL_TASK = Processor::TASK_ID_FIRST_AVAILABLE + 300,
};

static void top_level_task(const void*, size_t, const void*, size_t, Processor)
{
  log_app.print() << "deppart_byfield_itest starting";

  // Parent spaces
  IndexSpace<1,int> is_nodes(Rect<1,int>(0, TestConfig::num_nodes - 1));
  IndexSpace<1,int> is_edges(Rect<1,int>(0, TestConfig::num_edges - 1));

  // Choose memories
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

  // Create equal subspaces - one per piece
  std::vector<IndexSpace<1,int>> ss_nodes_eq, ss_edges_eq;
  is_nodes.create_equal_subspaces(TestConfig::num_pieces, 1, ss_nodes_eq,
                                  ProfilingRequestSet()).wait();
  is_edges.create_equal_subspaces(TestConfig::num_pieces, 1, ss_edges_eq,
                                  ProfilingRequestSet()).wait();

  // Per-piece CPU instances
  std::vector<RegionInstance> cpu_nodes_inst(TestConfig::num_pieces);
  std::vector<RegionInstance> cpu_edges_inst(TestConfig::num_pieces);

  // Per-piece field descriptors
  std::vector<FieldDataDescriptor<IndexSpace<1,int>, int>>           cpu_nodes(TestConfig::num_pieces);
  std::vector<FieldDataDescriptor<IndexSpace<1,int>, Point<1,int>>>  cpu_src  (TestConfig::num_pieces);
  std::vector<FieldDataDescriptor<IndexSpace<1,int>, Point<1,int>>>  cpu_dst  (TestConfig::num_pieces);

  // Allocate + describe per-piece CPU instances
  for (int i = 0; i < TestConfig::num_pieces; i++) {
    make_instance(cpu_nodes_inst[i], cpu_mem, ss_nodes_eq[i], {sizeof(int)}).wait();
    make_instance(cpu_edges_inst[i], cpu_mem, ss_edges_eq[i],
                  {sizeof(Point<1,int>), sizeof(Point<1,int>)}).wait();

    cpu_nodes[i].index_space  = ss_nodes_eq[i];
    cpu_nodes[i].inst         = cpu_nodes_inst[i];
    cpu_nodes[i].field_offset = FID_SUBGRAPH;

    cpu_src[i].index_space    = ss_edges_eq[i];
    cpu_src[i].inst           = cpu_edges_inst[i];
    cpu_src[i].field_offset   = FID_SRC;

    cpu_dst[i].index_space    = ss_edges_eq[i];
    cpu_dst[i].inst           = cpu_edges_inst[i];
    cpu_dst[i].field_offset   = FID_DST;
  }

  // Fill ids (deterministic or random) — piecewise
  auto gen_id = [&](Point<1,int> p)->int {
    if (TestConfig::random) {
      return Philox_2x32<>::rand_int(TestConfig::seed,
                                     /*counter=*/p[0],
                                     /*stream=*/0,
                                     /*bound=*/TestConfig::num_pieces);
    } else {
      // even split
      return int((long long)p[0] * TestConfig::num_pieces / TestConfig::num_nodes);
    }
  };
  auto gen_src = [&](Point<1,int> p)->Point<1,int> {
    if (TestConfig::random) {
      return Point<1,int>(Philox_2x32<>::rand_int(TestConfig::seed,
                                                  /*counter=*/p[0],
                                                  /*stream=*/0,
                                                  /*bound=*/TestConfig::num_nodes));
    } else {
      return Point<1,int>(p[0] % TestConfig::num_nodes);
    }
  };
  auto gen_dst = [&](Point<1,int> p)->Point<1,int> {
    if (TestConfig::random) {
      return Point<1,int>(Philox_2x32<>::rand_int(TestConfig::seed,
                                                  /*counter=*/p[0] + TestConfig::num_edges,
                                                  /*stream=*/0,
                                                  /*bound=*/TestConfig::num_nodes));
    } else {
      return Point<1,int>((p[0] + 1) % TestConfig::num_nodes);
    }
  };

  for (int i = 0; i < TestConfig::num_pieces; i++) {
    fill_index_space<1,int,int>(cpu_nodes_inst[i], FID_SUBGRAPH, ss_nodes_eq[i], gen_id);
    fill_index_space<1,int,Point<1,int>>(cpu_edges_inst[i], FID_SRC, ss_edges_eq[i], gen_src);
    fill_index_space<1,int,Point<1,int>>(cpu_edges_inst[i], FID_DST, ss_edges_eq[i], gen_dst);
  }

  if (TestConfig::show) {
    for (int i = 0; i < TestConfig::num_pieces; i++) {
      AffineAccessor<int,1,int> acc(cpu_nodes_inst[i], FID_SUBGRAPH);
      for (IndexSpaceIterator<1,int> it(ss_nodes_eq[i]); it.valid; it.step())
        for (PointInRectIterator<1,int> p(it.rect); p.valid; p.step())
          log_app.print() << "id[" << p.p << "]=" << acc[p.p];

      AffineAccessor<Point<1,int>,1,int> acc_src(cpu_edges_inst[i], FID_SRC);
      AffineAccessor<Point<1,int>,1,int> acc_dst(cpu_edges_inst[i], FID_DST);
      for (IndexSpaceIterator<1,int> it(ss_edges_eq[i]); it.valid; it.step())
        for (PointInRectIterator<1,int> p(it.rect); p.valid; p.step())
          log_app.print() << "edge[" << p.p << "]=" << acc_src[p.p] << "->" << acc_dst[p.p];
    }
  }

  // Colors 0..num_pieces-1
  std::vector<int> colors(TestConfig::num_pieces);
  for (int i = 0; i < TestConfig::num_pieces; i++) colors[i] = i;

  // CPU partitioning (use per-piece descriptors)
  std::vector<IndexSpace<1,int>> p_cpu_nodes, p_cpu_edges, p_cpu_rd, p_cpu_preimage2;
  Event e_cpu_byfield   = is_nodes.create_subspaces_by_field(cpu_nodes, colors, p_cpu_nodes,
                                                             ProfilingRequestSet());
  Event e_cpu_bypreimg  = is_edges.create_subspaces_by_preimage(cpu_dst, p_cpu_nodes, p_cpu_edges,
                                                                ProfilingRequestSet(), e_cpu_byfield);
  Event e_cpu_image     = is_nodes.create_subspaces_by_image(cpu_src, p_cpu_edges, p_cpu_rd,
                                                             ProfilingRequestSet(), e_cpu_bypreimg);
  // NEW: second preimage (edges by dst over p_cpu_rd)
  Event e_cpu_bypreimg2 = is_edges.create_subspaces_by_preimage(cpu_dst, p_cpu_rd, p_cpu_preimage2,
                                                                ProfilingRequestSet(), e_cpu_image);

  // GPU path (optional if GPU exists)
  std::vector<IndexSpace<1,int>> p_gpu_nodes, p_gpu_edges, p_gpu_rd, p_gpu_preimage2;
  if (have_gpu) {
    // Per-piece GPU instances & descriptors
    std::vector<RegionInstance> gpu_nodes_inst(TestConfig::num_pieces);
    std::vector<RegionInstance> gpu_edges_inst(TestConfig::num_pieces);

    std::vector<FieldDataDescriptor<IndexSpace<1,int>, int>>          gpu_nodes(TestConfig::num_pieces);
    std::vector<FieldDataDescriptor<IndexSpace<1,int>, Point<1,int>>> gpu_src  (TestConfig::num_pieces);
    std::vector<FieldDataDescriptor<IndexSpace<1,int>, Point<1,int>>> gpu_dst  (TestConfig::num_pieces);

    for (int i = 0; i < TestConfig::num_pieces; i++) {
      make_instance(gpu_nodes_inst[i], gpu_mem, ss_nodes_eq[i], {sizeof(int)}).wait();
      make_instance(gpu_edges_inst[i], gpu_mem, ss_edges_eq[i],
                    {sizeof(Point<1,int>), sizeof(Point<1,int>)}).wait();

      // Copy CPU -> GPU per subspace
      copy_field<1,int,int>(ss_nodes_eq[i], cpu_nodes_inst[i], gpu_nodes_inst[i], FID_SUBGRAPH);
      copy_field<1,int,Point<1,int>>(ss_edges_eq[i], cpu_edges_inst[i], gpu_edges_inst[i], FID_SRC);
      copy_field<1,int,Point<1,int>>(ss_edges_eq[i], cpu_edges_inst[i], gpu_edges_inst[i], FID_DST);

      // GPU descriptors
      gpu_nodes[i].index_space  = ss_nodes_eq[i];
      gpu_nodes[i].inst         = gpu_nodes_inst[i];
      gpu_nodes[i].field_offset = FID_SUBGRAPH;

      gpu_src[i].index_space    = ss_edges_eq[i];
      gpu_src[i].inst           = gpu_edges_inst[i];
      gpu_src[i].field_offset   = FID_SRC;

      gpu_dst[i].index_space    = ss_edges_eq[i];
      gpu_dst[i].inst           = gpu_edges_inst[i];
      gpu_dst[i].field_offset   = FID_DST;
    }

    Event e_gpu_byfield  = is_nodes.create_subspaces_by_field(gpu_nodes, colors, p_gpu_nodes,
                                                              ProfilingRequestSet());
    Event e_gpu_bypreimg = is_edges.create_subspaces_by_preimage(gpu_dst, p_gpu_nodes, p_gpu_edges,
                                                                 ProfilingRequestSet(), e_gpu_byfield);
    Event e_gpu_image    = is_nodes.create_subspaces_by_image(gpu_src, p_gpu_edges, p_gpu_rd,
                                                              ProfilingRequestSet(), e_gpu_bypreimg);
    // NEW: second preimage (edges by dst over p_gpu_rd)
    Event e_gpu_bypreimg2 = is_edges.create_subspaces_by_preimage(gpu_dst, p_gpu_rd, p_gpu_preimage2,
                                                                  ProfilingRequestSet(), e_gpu_image);

    e_cpu_bypreimg2.wait();
    e_gpu_bypreimg2.wait();

    // Compare CPU vs GPU partitions (including second preimage)
    if (TestConfig::verify) {
      int errs = 0;
      errs += compare_partitions(p_cpu_nodes,      p_gpu_nodes);
      errs += compare_partitions(p_cpu_edges,      p_gpu_edges);
      errs += compare_partitions(p_cpu_rd,         p_gpu_rd);
      errs += compare_partitions(p_cpu_preimage2,  p_gpu_preimage2); // NEW check
      if (errs) {
        log_app.fatal() << "Mismatch between CPU and GPU partitions, errors=" << errs;
        assert(0);
      }
    }

    for (int i = 0; i < TestConfig::num_pieces; i++) {
      gpu_nodes_inst[i].destroy();
      gpu_edges_inst[i].destroy();
    }
  } else {
    e_cpu_bypreimg2.wait();
  }

  // Cleanup CPU
  for (int i = 0; i < TestConfig::num_pieces; i++) {
    cpu_nodes_inst[i].destroy();
    cpu_edges_inst[i].destroy();
  }
  is_nodes.destroy();
  is_edges.destroy();

  log_app.print() << "deppart_1d_itest: PASS";
}

// ---------------- Main (same as transpose_test pattern) -----------------
int main(int argc, char** argv)
{
  Runtime rt;
  rt.init(&argc, &argv);

  // Parse simple flags similar to the example
  CommandLineParser cp;
  cp.add_option_int("-n",      TestConfig::num_nodes)
    .add_option_int("-e",      TestConfig::num_edges)
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

