#include "realm/deppart/setops.h"
#include "realm/deppart/setops_gpu_kernels.hpp"
#include "realm/deppart/partitions_gpu_impl.hpp"
#include <cub/cub.cuh>
#include "realm/nvtx.h"

namespace Realm {

template <int N, typename T>
void GPUUnionMicroOp<N, T>::gpu_populate(void) {
  Memory my_mem;
  bool found_gpu_memory = false;
  Machine machine = Machine::get_machine();
  std::set<Memory> all_memories;
  machine.get_all_memories(all_memories);
  for(auto& memory : all_memories) {
    if(memory.kind() == Memory::GPU_FB_MEM) {
      my_mem = memory;
      found_gpu_memory = true;
      break;
    }
  }
  assert(found_gpu_memory);
  if (sparsity_outputs.size() == 0) {
     return;
  }
    nvtx_range_push("cuda", "gpu_union_populate");

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), stream);

    nvtx_range_push("cuda", "flatten sparsity and inst entries");

    // 1) figure out the final size and build the offsets array
    std::vector<size_t> input_offsets(inputs.size() + 1);
    size_t total_size = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      input_offsets[i] = total_size;
      for (size_t j = 0; j < inputs[i].size(); ++j) {
        if (inputs[i][j].dense()) {
          total_size += 1;
        } else {
          // only call get_entries() once per input
          total_size += inputs[i][j].sparsity.impl()->get_entries().size();
        }
      }
    }
    // final end offset
    input_offsets[inputs.size()] = total_size;

    nvtx_range_pop();
    nvtx_range_push("cuda", "build device entries");

    // inputs entries allocation
    RegionInstance inputs_entries_instance = this->realm_malloc(total_size * sizeof(SparsityMapEntry<N,T>), my_mem);
    SparsityMapEntry<N,T>* d_inputs_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(inputs_entries_instance, 0).base);

    // Offsets allocation
    RegionInstance offsets_instance = this->realm_malloc((inputs.size()+1) * sizeof(size_t), my_mem);
    size_t* d_offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(offsets_instance, 0).base);

    CUDA_CHECK(cudaMemcpyAsync(d_offsets, input_offsets.data(), (inputs.size()+1) * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);

    // 3) fill in place
    size_t pos = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      for (size_t j = 0; j < inputs[i].size(); j++) {
        if (inputs[i][j].dense()) {
          // just one rect
          SparsityMapEntry<N,T> entry;
          entry.bounds = inputs[i][j].bounds;
          CUDA_CHECK(cudaMemcpyAsync(d_inputs_entries + pos, &entry, sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
          ++pos;
        } else {
            auto& tmp = inputs[i][j].sparsity.impl()->get_entries();
            CUDA_CHECK(cudaMemcpyAsync(d_inputs_entries + pos, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
            pos += tmp.size();
        }
      }
    }

    RegionInstance output_rects_instance = this->realm_malloc(total_size * sizeof(RectDesc<N,T>), my_mem);
    RectDesc<N, T>* d_output_rects = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(output_rects_instance, 0).base);

    int threads_per_block = 256;
    int grid_size = (total_size + threads_per_block - 1) / threads_per_block;

    union_map_rects<N,T><<<grid_size, threads_per_block, 0, stream>>>(d_inputs_entries, d_offsets, total_size, inputs.size(), d_output_rects);
    KERNEL_CHECK(stream);

    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    cudaStreamDestroy(stream);


    this->complete_rect_pipeline(d_output_rects, total_size, my_mem,
    /* the Container: */  sparsity_outputs,
    /* getIndex: */       [&](auto const& elem){
                            // elem is a SparsityMap<N,T> from the vector
                            return size_t(&elem - sparsity_outputs.data());
                         },
    /* getMap: */         [&](auto const& elem){
                          // return the SparsityMap key itself
                          return elem;
                       });

  nvtx_range_pop();
  nvtx_range_pop();

}

  #define DOIT(N,T) \
    template class GPUUnionMicroOp<N,T>; \

  FOREACH_NT(DOIT)

}