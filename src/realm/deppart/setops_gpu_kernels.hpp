#pragma once
#include "partitions.h"
#include "../sparsity.h"

namespace Realm {

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

}