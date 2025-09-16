#pragma once
#include "partitions.h"
#include "../sparsity.h"

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

}