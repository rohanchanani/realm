#include "realm/deppart/setops.h"
#include "realm/deppart/setops_gpu_kernels.hpp"
#include "realm/deppart/partitions_gpu_impl.hpp"
#include <cub/cub.cuh>
#include "realm/nvtx.h"

namespace Realm {

/*
   *  Input (stored in MicroOp): A list of lists of input index spaces.
   *  Output: A collapsed list of all rectangles from all the input spaces combined, marked by which input they came from, which are then
   *  sent off to complete_rect_pipeline.
   *  Approach: Mark the offsets between input index spaces, copy all the entries from all spaces into a single buffer, then map that buffer
   *  from entries to marked rectangles in parallel. Finally, call complete_rect_pipeline to finish the job.
   */
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
    NVTX_DEPPART(gpu_union);

    cudaStream_t stream = Cuda::get_task_cuda_stream();

    // Figure out the final size and build the offsets array
    std::vector<size_t> input_offsets(inputs.size() + 1);
    size_t total_size = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      input_offsets[i] = total_size;
      for (size_t j = 0; j < inputs[i].size(); ++j) {
        if (inputs[i][j].dense()) {
          total_size += 1;
        } else {
          total_size += inputs[i][j].sparsity.impl()->get_entries().size();
        }
      }
    }
    input_offsets[inputs.size()] = total_size;

    // Inputs entries allocation
    RegionInstance inputs_entries_instance = this->realm_malloc(total_size * sizeof(SparsityMapEntry<N,T>), my_mem);
    SparsityMapEntry<N,T>* d_inputs_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(inputs_entries_instance, 0).base);

    // Offsets allocation
    RegionInstance offsets_instance = this->realm_malloc((inputs.size()+1) * sizeof(size_t), my_mem);
    size_t* d_offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(offsets_instance, 0).base);

    CUDA_CHECK(cudaMemcpyAsync(d_offsets, input_offsets.data(), (inputs.size()+1) * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);

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

}

  #define DOIT(N,T) \
    template class GPUUnionMicroOp<N,T>; \

  FOREACH_NT(DOIT)

}