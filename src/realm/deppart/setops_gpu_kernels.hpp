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

  template <int N, typename T>
  __global__ void just_morton_codes(
    const RectDesc<N, T>* d_inputs_entries,
    const Rect<N,T>* d_global_bounds,
    size_t total_rects,
    uint64_t* d_morton_codes) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_rects) return;
    const auto &entry = d_inputs_entries[idx];
    d_morton_codes[idx] = bvh_morton_code(entry.rect, *d_global_bounds);
  }


  template <int N, typename T>
  __global__ void bvh_build_morton_codes(
    RectDesc<N, T>* d_inputs_entries,
    const Rect<N,T>* d_global_bounds,
    size_t total_rects,
    uint64_t* d_morton_codes,
    uint64_t* d_indices,
    uint64_t* d_rhs_indices) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_rects) return;
    const auto &entry = d_inputs_entries[idx];
    d_morton_codes[idx] = bvh_morton_code(entry.rect, *d_global_bounds);
    d_indices[idx] = idx;
    d_rhs_indices[idx] = entry.src_idx;
  }

  template <int N, typename T>
__device__ uint64_t bvh_morton_code(const Rect<N,T>& rect,
                              const Rect<N,T>& globalBounds) {
    // bits per axis (floor)
    constexpr int bits     = 64 / N;
    constexpr uint64_t maxQ = (bits == 64 ? ~0ULL
                                         : (1ULL << bits) - 1);

    uint64_t coords[N];
#pragma unroll
    for(int d = 0; d < N; ++d) {
      // 1) compute centroid in dimension d
      float center = 0.5f * (float(rect.lo[d]) + float(rect.hi[d]) + 1.0f);

      // 2) normalize into [0,1] using globalBounds
      float span = float(globalBounds.hi[d] + 1 - globalBounds.lo[d]);
      float norm = (center - float(globalBounds.lo[d])) / span;

      // 3) quantize to [0 … maxQ]
      uint64_t q = uint64_t(norm * float(maxQ) + 0.5f);
      coords[d] = (q > maxQ ? maxQ : q);
    }

    // 4) interleave bits MSB→LSB across all dims
    uint64_t code = 0;
    for(int b = bits - 1; b >= 0; --b) {
#pragma unroll
      for(int d = 0; d < N; ++d) {
        code = (code << 1) | ((coords[d] >> b) & 1ULL);
      }
    }

    return code;
  }

  __device__ __forceinline__
  int bvh_common_prefix(const uint64_t *morton, const uint64_t *leafIdx, int i, int j, int n) {
    if (j < 0 || j >= n) return -1;
    uint64_t x = morton[i] ^ morton[j];
    uint64_t y = leafIdx[i] ^ leafIdx[j];
    if (x == 0) {
      return 64 + __clzll(y);
    }
    return __clzll(x);
  }

  __global__
void bvh_build_radix_tree_kernel(
    const uint64_t *morton,    // [n]
    const uint64_t *leafIdx,   // [n]  (unused here but kept for symmetry)
    int n,
    int *childLeft,            // [2n−1]
    int *childRight,           // [2n−1]
    int *parent);               // [2n−1], pre‐initialized to −1

  __global__
void bvh_build_root_kernel(
    int *root,
    int *parent,
    size_t total_rects);

template<int N, typename T>
__global__
void bvh_merge_internal_boxes_kernel(
    size_t total_rects,
    const int *childLeft,      // [(2n−1)]
    const int *childRight,     // [(2n−1)]
    const int *parent,         // [(2n−1)]
    Rect<N,T> *boxes,                 // [(2n−1)×N]
    int *visitCount);           // [(2n−1)] initialized to zero


#ifdef REALM_DEFINE_BVH_KERNELS

__global__
void bvh_build_radix_tree_kernel(
    const uint64_t *morton,    // [n]
    const uint64_t *leafIdx,   // [n]  (unused here but kept for symmetry)
    int n,
    int *childLeft,            // [2n−1]
    int *childRight,           // [2n−1]
    int *parent)               // [2n−1], pre‐initialized to −1
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;
  int i = idx;
  if (i >= n-1) return;            // we only build n−1 internal nodes

  int left, right;
  int dL = bvh_common_prefix(morton, leafIdx, i, i-1, n);
  int dR = bvh_common_prefix(morton, leafIdx, i, i+1, n);
  int d  = (dR > dL ? +1 : -1);
  int deltaMin = (dR > dL ? dL : dR);

  // 3) find j by exponential + binary search
  int l_max = 2;
  int delta = -1;
  int i_tmp = i + d * l_max;
  if (0 <= i_tmp && i_tmp < n) {
    delta = bvh_common_prefix(morton, leafIdx, i, i_tmp, n);
  }
  while (delta > deltaMin) {
    l_max <<= 1;
    i_tmp = i + d * l_max;
    delta = -1;
    if (0 <= i_tmp && i_tmp < n) {
      delta = bvh_common_prefix(morton, leafIdx, i, i_tmp, n);
    }
  }
  int l = 0;
  int t = (l_max) >> 1;
  while (t > 0) {
    i_tmp = i + d*(l + t);
    delta = -1;
    if (0 <= i_tmp && i_tmp < n) {
      delta = bvh_common_prefix(morton, leafIdx, i, i_tmp, n);
    }
    if (delta > deltaMin) {
      l += t;
    }
    t >>= 1;
  }
  if (d < 0) {
    right = i;
    left = i + d*l;
  } else {
    left = i;
    right = i + d*l;
  }

  int gamma;
  if (morton[left] == morton[right] && leafIdx[left] == leafIdx[right]) {
    gamma = (left+right) >> 1;
  } else {
    int deltaNode = bvh_common_prefix(morton, leafIdx, left, right, n);
    int split = left;
    int stride = right - left;
    do {
      stride = (stride + 1) >> 1;
      int middle = split + stride;
      if (middle < right) {
        int delta = bvh_common_prefix(morton, leafIdx, left, middle, n);
        if (delta > deltaNode) {
          split = middle;
        }
      }
    } while (stride > 1);
    gamma = split;
  }

  int left_node = gamma;
  int right_node = gamma + 1;
  if (left == gamma) {
    left_node += n-1;
  }
  if (right == gamma + 1) {
    right_node += n-1;
  }

  childLeft [idx] = left_node;
  childRight[idx] = right_node;
  parent[left_node]  = idx;
  parent[right_node] = idx;
}

__global__
void bvh_build_root_kernel(
    int *root,
    int *parent,
    size_t total_rects) {

  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  if (tid >= 2 * total_rects - 1) return;
  if (parent[tid] == -1) {
    *root = tid;
  }
}

//
// 3) Merge internal boxes bottom-up
//
template<int N, typename T>
__global__
void bvh_merge_internal_boxes_kernel(
    size_t total_rects,
    const int *childLeft,      // [(2n−1)]
    const int *childRight,     // [(2n−1)]
    const int *parent,         // [(2n−1)]
    Rect<N,T> *boxes,                 // [(2n−1)×N]
    int *visitCount)           // [(2n−1)] initialized to zero
{
  int leaf = blockIdx.x*blockDim.x + threadIdx.x;
  if (leaf >= total_rects) return;

  int cur = leaf + total_rects - 1;
  int p   = parent[cur];

  while(p >= 0) {
    // increment visit count; the second arrival merges
    int prev = atomicAdd(&visitCount[p], 1);
    if (prev == 1) {
      // both children ready, do the merge
      int c0 = childLeft[p], c1 = childRight[p];
      boxes[p] = boxes[c0].union_bbox(boxes[c1]);
      // climb
      cur = p;
      p   = parent[cur];
    } else {
      // first child arrived, wait for sibling
      break;
    }
  }
}
#endif

  //
  // 2) Initialize leaf boxes
  //
  template<int N, typename T>
  __global__
  void bvh_init_leaf_boxes_kernel(
      const RectDesc<N,T> *rects,    // [G] all flattened Rects
      const uint64_t    *leafIdx, // [n] maps leaf→orig Rect index
      size_t total_rects,
      Rect<N,T> *boxes)                 // [(2n−1)]
  {
    int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= total_rects) return;

    size_t orig = leafIdx[k];
    boxes[k + total_rects - 1] = rects[orig].rect;
  }

template <
  int N, typename T
>
__global__
void intersect_query_bvh(
  RectDesc<N, T>* queries,
  int *root,
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
  int node = *root;
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