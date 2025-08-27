#pragma once
#include "partitions.h"
#include "../sparsity.h"
#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
namespace cg = cooperative_groups;

namespace Realm {

  // Given a combined buffer of sparsity map entries and the offsets
  // of input within that buffer, converts the entries to rectangles
  // and marks them with which rectangle they came from.
  template <int N, typename T>
  __global__
  void union_map_rects(const SparsityMapEntry<N,T>* d_in,
                                  const size_t*      offsets,
                                  size_t             totalRects,
                                  size_t             numSources,
                                  RectDesc<N,T>*        d_out)
  {
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= totalRects) return;
    size_t low = 0, high = numSources;
    while (low < high) {
      size_t mid = (low + high) >> 1;
      if (offsets[mid+1] <= idx) low = mid + 1;
      else                      high = mid;
    }
    d_out[idx].rect = d_in[idx].bounds;
    d_out[idx].src_idx = low;
  }

    template <int N, typename T>
  __global__
  void single_map_rects(const SparsityMapEntry<N,T>* d_in,
                                  size_t             totalRects,
                                  RectDesc<N,T>*        d_out)
  {
    size_t idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= totalRects) return;
    d_out[idx].rect = d_in[idx].bounds;
    d_out[idx].src_idx = 0;
  }

template <
  int N, typename T
>
__global__
void intersect_query_bvh(
  RectDesc<N, T>* queries,
  int root,
  int *childLeft,
  int *childRight,
  uint64_t *leafIdx,
  uint64_t *targets_indices,
  Rect<N,T> *boxes,
  size_t numQueries,
  size_t numBoxes,
  uint32_t* d_targets_prefix,
  uint32_t* d_target_counters,
  RectDesc<N,T> *d_rects
) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numQueries) return;
  Rect<N, T> in_rect = queries[idx].rect;
  constexpr int MAX_STACK = 64; // max stack size for BVH traversal
  int stack[MAX_STACK];
  int sp = 0;

  // start at the root
  stack[sp++] = -1;
  int node = root;
  do
  {

    int left = childLeft[node];
    int right = childRight[node];

    bool overlapL = boxes[left].overlaps(in_rect);
    bool overlapR = boxes[right].overlaps(in_rect);

    if (overlapL && left >= numBoxes - 1) {
      // left child is a leaf
      uint64_t rect_idx = leafIdx[left - (numBoxes - 1)];
      size_t target_idx = targets_indices[rect_idx];
      if (target_idx == queries[idx].src_idx) {
        uint32_t local = atomicAdd(&d_target_counters[target_idx], 1);
        if (d_rects != nullptr) {
          RectDesc<N,T> output;
          output.src_idx = target_idx;
          output.rect = boxes[left].intersection(in_rect);
          uint32_t out_idx = d_targets_prefix[target_idx] + local;
          d_rects[out_idx] = output;
        }
      }
    }
    if (overlapR && right >= numBoxes - 1) {
      uint64_t rect_idx = leafIdx[right - (numBoxes - 1)];
      size_t target_idx = targets_indices[rect_idx];
      if (target_idx == queries[idx].src_idx) {
        uint32_t local = atomicAdd(&d_target_counters[target_idx], 1);
        if (d_rects != nullptr) {
          RectDesc<N,T> output;
          output.src_idx = target_idx;
          output.rect = boxes[right].intersection(in_rect);
          uint32_t out_idx = d_targets_prefix[target_idx] + local;
          d_rects[out_idx] = output;
        }
      }
    }

    bool traverseL = overlapL && left < numBoxes - 1;
    bool traverseR = overlapR && right < numBoxes - 1;

    if (!traverseL && !traverseR) {
      node = stack[--sp];
    } else {
      node = (traverseL ? left : right);
      if (traverseL && traverseR) {
        stack[sp++] = right;
      }
    }
  } while (node != -1);
}

template<int N, typename T>
__device__ inline void swap_ptr(RectDesc<N,T>*& a, RectDesc<N,T>*& b) {
  RectDesc<N,T>* t = a; a = b; b = t;
}
__device__ inline void swap_int(int& a, int& b) {
  int t = a; a = b; b = t;
}

  template <int N, typename T>
__device__ inline void subtract_one_frag(const RectDesc<N, T> &R, const Rect<N, T> &O, int &nextCnt)
{
  Rect<N, T> mid = R.rect;
  size_t src_idx = R.src_idx;

  if(!mid.overlaps(O)) {
    nextCnt++;
    return;
  }

#pragma unroll
  for(int d = 0; d < N; ++d) {
    // lower slab
    if(mid.lo[d] < O.lo[d]) {
      nextCnt++;
      mid.lo[d] = O.lo[d];
    }
    // upper slab
    if(mid.hi[d] > O.hi[d]) {
      nextCnt++;
      mid.hi[d] = O.hi[d];
    }
    if(mid.empty())
      break; // nothing left that overlaps O along remaining axes
  }
  // discard mid (it’s R∩O)
}

template <int N, typename T>
__device__ inline void subtract_one_frag(const RectDesc<N, T> &R, const Rect<N, T> &O,
                                         RectDesc<N, T> *nextBuf, int &nextCnt)
{
  Rect<N, T> mid = R.rect;
  size_t src_idx = R.src_idx;

  if(!mid.overlaps(O)) {
    RectDesc<N, T> output;
    output.src_idx = src_idx;
    output.rect = mid;
    nextBuf[nextCnt++] = output;
    return;
  }

#pragma unroll
  for(int d = 0; d < N; ++d) {
    // lower slab
    if(mid.lo[d] < O.lo[d]) {
      RectDesc<N, T> S;
      S.rect = mid;
      S.rect.hi[d] = T(O.lo[d]) - T(1);
      S.src_idx = src_idx;
      nextBuf[nextCnt++] = S;
      mid.lo[d] = O.lo[d];
    }
    // upper slab
    if(mid.hi[d] > O.hi[d]) {
      RectDesc<N, T> S;
      S.rect = mid;
      S.rect.lo[d] = T(O.hi[d]) + T(1);
      S.src_idx = src_idx;
      nextBuf[nextCnt++] = S;
      mid.hi[d] = O.hi[d];
    }
    if(mid.empty())
      break; // nothing left that overlaps O along remaining axes
  }
  // discard mid (it’s R∩O)
}

template <int N, typename T>
__global__ void
difference_count_hits(RectDesc<N, T> *queries, int *root, int *childLeft, int *childRight,
                      uint64_t *leafIdx, uint64_t *targets_indices, uint32_t* d_hits_prefix, Rect<N, T> *boxes,
                      size_t numQueries, size_t numBoxes, uint32_t *d_hit_counts, uint32_t *d_frag_counts, Rect<N, T> *d_hits)
{
  const size_t qid = blockIdx.x * blockDim.x + threadIdx.x;
  if(qid >= numQueries)
    return;

  const Rect<N, T> A = queries[qid].rect;
  const size_t src = queries[qid].src_idx;

  uint32_t num_hits = 0;

  // BVH traversal: collect RHS hits that share src_idx
  constexpr int MAX_STACK = 256;
  int stack[MAX_STACK];
  int sp = 0;
  stack[sp++] = -1;
  int node = *root;

  while(node != -1) {
    const int L = childLeft[node];
    const int R = childRight[node];

    const bool ol = boxes[L].overlaps(A);
    const bool orr = boxes[R].overlaps(A);

    if(ol && L >= int(numBoxes - 1)) {
      const uint64_t ridx = leafIdx[L - int(numBoxes - 1)];
      const size_t tgt = size_t(targets_indices[ridx]);
      if(tgt == src) {
        num_hits++;
        if (d_hits != nullptr) {
          d_hits[d_hits_prefix[qid] + num_hits-1] = boxes[L].intersection(A);
        }
      }
    }
    if(orr && R >= int(numBoxes - 1)) {
      const uint64_t ridx = leafIdx[R - int(numBoxes - 1)];
      const size_t tgt = size_t(targets_indices[ridx]);
      if(tgt == src) {
        num_hits++;
        if (d_hits != nullptr) {
          d_hits[d_hits_prefix[qid] + num_hits-1] = boxes[R].intersection(A);
        }
      }
    }

    const bool tL = ol && (L < int(numBoxes - 1));
    const bool tR = orr && (R < int(numBoxes - 1));
    if(!tL && !tR) {
      node = stack[--sp];
    } else {
      node = tL ? L : R;
      if(tL && tR)
        stack[sp++] = R;
    }
  }

  d_hit_counts[qid] = num_hits;
  if (d_frag_counts != nullptr) {
    uint32_t result = 1;
    for(int d = 0; d < N; d++) {
      uint32_t span = A.hi[d] - A.lo[d] + 1;
      result *= (span < 2 * num_hits + 1 ? span : 2 * num_hits + 1);
    }
    d_frag_counts[qid] = result;
  }
}

template <int BLOCK_THREADS, int N, typename T>
__global__ void difference_query_bvh(
    RectDesc<N, T> *queries, Rect<N, T> *global_hits, RectDesc<N, T>* global_frags, RectDesc<N, T>* global_frags_next, uint32_t* d_hits_prefix, uint32_t* d_hits_count,
    size_t numQueries, uint32_t maxHits, int* g_block_sum, int* g_block_base, size_t* frags_count, uint8_t* choice)
{
  RectDesc<N, T>* frags = global_frags;
  RectDesc<N, T>* frags_next = global_frags_next;

  *frags_count = numQueries;
  if (maxHits==0) {
    *choice = 2;
    return;
  }

  cg::grid_group grid = cg::this_grid();
  using BlockScan = cub::BlockScan<int, BLOCK_THREADS>;
  __shared__ typename BlockScan::TempStorage scan_smem;

  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
     idx < numQueries;
     idx += gridDim.x * blockDim.x) {
    RectDesc<N, T> query = queries[idx];
    query.src_idx = idx;
    frags[idx] = query;
  }

  __threadfence();
  grid.sync();

  size_t my_frag_count = numQueries;
  for (int i = 0; i < maxHits; i++) {
    int curr_count = 0;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < my_frag_count;
       idx += gridDim.x * blockDim.x) {
      if (i >= d_hits_count[frags[idx].src_idx]) {
        curr_count++;
        continue;
      }
      RectDesc<N, T> query = frags[idx];
      Rect<N, T> hit = global_hits[d_hits_prefix[query.src_idx] + i];
      subtract_one_frag(query, hit, curr_count);
    }
    int thread_base = 0, block_sum = 0;
    BlockScan(scan_smem).ExclusiveSum(curr_count, thread_base, block_sum);
    __threadfence();
    if (threadIdx.x == 0) g_block_sum[blockIdx.x] = block_sum;
    grid.sync();
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      int acc = 0;
      for (int b = 0; b < gridDim.x; ++b) { g_block_base[b] = acc; acc += g_block_sum[b]; }
      g_block_base[gridDim.x] = acc; // total_next
    }
    grid.sync();
    int out_base = g_block_base[blockIdx.x] + thread_base;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < my_frag_count;
       idx += gridDim.x * blockDim.x) {
      if (i >= d_hits_count[frags[idx].src_idx]) {
        frags_next[out_base++] = frags[idx];
        continue;
      }
      RectDesc<N, T> query = frags[idx];
      Rect<N, T> hit = global_hits[d_hits_prefix[query.src_idx] + i];
      subtract_one_frag(query, hit, frags_next, out_base);
    }
    grid.sync();
    my_frag_count = g_block_base[gridDim.x];
    swap_ptr(frags, frags_next);
  }
  for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
       idx < my_frag_count;
       idx += gridDim.x * blockDim.x) {
    frags[idx].src_idx = queries[frags[idx].src_idx].src_idx;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *choice = (frags == global_frags);
    *frags_count = my_frag_count;
  }
}

}