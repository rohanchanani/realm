#include "realm/deppart/setops.h"
#define REALM_DEFINE_BVH_KERNELS
#include "realm/deppart/setops_gpu_kernels.hpp"
#undef REALM_DEFINE_BVH_KERNELS
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

    //We copy into one contiguous host buffer, then copy to device
    Memory sysmem;
    assert(find_memory(sysmem, Memory::SYSTEM_MEM));

    RegionInstance h_instance = this->realm_malloc(total_size * sizeof(SparsityMapEntry<N,T>), sysmem);
    SparsityMapEntry<N, T>* h_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(h_instance, 0).base);

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
          memcpy(h_entries + pos, &entry, sizeof(SparsityMapEntry<N,T>));
          ++pos;
        } else {
            auto& tmp = inputs[i][j].sparsity.impl()->get_entries();
            memcpy(h_entries + pos, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>));
            pos += tmp.size();
        }
      }
    }

    CUDA_CHECK(cudaMemcpyAsync(d_inputs_entries, h_entries, total_size * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);

    RegionInstance output_rects_instance = this->realm_malloc(total_size * sizeof(RectDesc<N,T>), my_mem);
    RectDesc<N, T>* d_output_rects = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(output_rects_instance, 0).base);

    union_map_rects<N,T><<<COMPUTE_GRID(total_size), THREADS_PER_BLOCK, 0, stream>>>(d_inputs_entries, d_offsets, total_size, inputs.size(), d_output_rects);
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

  template <int N, typename T>
  void GPUIntersectionMicroOp<N, T>::gpu_populate_single(void) {
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
      NVTX_DEPPART(gpu_intersection_single);

      cudaStream_t stream = Cuda::get_task_cuda_stream();

      size_t lhs_size = inputs[0][0].dense() ? 1 : inputs[0][0].sparsity.impl()->get_entries().size();

      // inputs entries allocation
      RegionInstance lhs_entries_instance = this->realm_malloc(lhs_size * sizeof(SparsityMapEntry<N,T>), my_mem);
      SparsityMapEntry<N,T>* d_lhs_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(lhs_entries_instance, 0).base);

      if (inputs[0][0].dense()) {
        // just one rect
        SparsityMapEntry<N,T> entry;
        entry.bounds = inputs[0][0].bounds;
        CUDA_CHECK(cudaMemcpyAsync(d_lhs_entries, &entry, sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
      } else {
        auto& tmp = inputs[0][0].sparsity.impl()->get_entries();
        CUDA_CHECK(cudaMemcpyAsync(d_lhs_entries, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
      }

      RegionInstance lhs_rects_instance = this->realm_malloc(lhs_size * sizeof(RectDesc<N,T>), my_mem);
      RectDesc<N, T>* d_lhs_rects = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(lhs_rects_instance, 0).base);

      single_map_rects<N,T><<<COMPUTE_GRID(lhs_size), THREADS_PER_BLOCK, 0, stream>>>(d_lhs_entries, lhs_size, d_lhs_rects);
      KERNEL_CHECK(stream);

      RegionInstance input_counters_instance = this->realm_malloc((2 * inputs.size() + 1) * sizeof(uint32_t), my_mem);
      uint32_t* d_input_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(input_counters_instance, 0).base);
      uint32_t* d_inputs_prefix = d_input_counters + inputs.size();

      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      lhs_entries_instance.destroy();

      for (size_t i = 1; i < inputs[0].size(); i++) {

        std::vector<IndexSpace<N, T>> tmp_rhs = {inputs[0][i]};

        collapsed_space<N, T> rhs_space;
        RegionInstance offsets_instance = this->realm_malloc((tmp_rhs.size()+1) * sizeof(size_t), my_mem);
        rhs_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(offsets_instance, 0).base);
        rhs_space.num_children = tmp_rhs.size();

        RegionInstance rhs_entries_instance;

        GPUMicroOp<N, T>::collapse_multi_space(tmp_rhs, rhs_entries_instance, rhs_space, my_mem, stream);

        BVH<N, T> intersect_bvh;
        RegionInstance bvh_instance;
        this->build_bvh(rhs_space, bvh_instance, intersect_bvh, my_mem, stream);

          CUDA_CHECK(cudaMemsetAsync(d_input_counters, 0, inputs.size() * sizeof(uint32_t), stream), stream);

          intersect_query_bvh< N, T ><<<COMPUTE_GRID(lhs_size), THREADS_PER_BLOCK, 0, stream>>>(d_lhs_rects, intersect_bvh.root, intersect_bvh.childLeft, intersect_bvh.childRight, intersect_bvh.indices, intersect_bvh.labels, intersect_bvh.boxes, lhs_size, intersect_bvh.num_leaves, nullptr, d_input_counters, nullptr);
          KERNEL_CHECK(stream);

          uint32_t h_input_counters[inputs.size()+1];
          h_input_counters[0] = 0; // prefix sum starts at 0
          CUDA_CHECK(cudaMemcpyAsync(h_input_counters+1, d_input_counters, inputs.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          for (size_t i = 0; i < inputs.size(); ++i) {
            h_input_counters[i+1] += h_input_counters[i];
          }

          uint32_t num_valid_rects = h_input_counters[inputs.size()];

          if (num_valid_rects==0) {
            input_counters_instance.destroy();
            lhs_rects_instance.destroy();
            rhs_entries_instance.destroy();
            offsets_instance.destroy();
            bvh_instance.destroy();
            for (auto it : sparsity_outputs) {
              SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
              impl->gpu_finalize();
            }
            return;
          }

          CUDA_CHECK(cudaMemcpyAsync(d_inputs_prefix, h_input_counters, (inputs.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

          RegionInstance output_instance = this->realm_malloc(num_valid_rects * sizeof(RectDesc<N,T>), my_mem);
          RectDesc<N, T>* d_output_rects = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(output_instance, 0).base);

          CUDA_CHECK(cudaMemsetAsync(d_input_counters, 0, (inputs.size()) * sizeof(uint32_t), stream), stream);

          intersect_query_bvh< N, T ><<<COMPUTE_GRID(lhs_size), THREADS_PER_BLOCK, 0, stream>>>(d_lhs_rects, intersect_bvh.root, intersect_bvh.childLeft, intersect_bvh.childRight, intersect_bvh.indices, intersect_bvh.labels, intersect_bvh.boxes, lhs_size, intersect_bvh.num_leaves, d_inputs_prefix, d_input_counters, d_output_rects);
          KERNEL_CHECK(stream);
          CUDA_CHECK(cudaStreamSynchronize(stream), stream);
          lhs_rects_instance.destroy();
          rhs_entries_instance.destroy();
          offsets_instance.destroy();
          bvh_instance.destroy();

          lhs_rects_instance = output_instance;
          d_lhs_rects = d_output_rects;
          lhs_size = num_valid_rects;

      }


      this->complete_rect_pipeline(d_lhs_rects, lhs_size, my_mem,
      /* the Container: */  sparsity_outputs,
      /* getIndex: */       [&](auto const& elem){
                              // elem is a SparsityMap<N,T> from the vector
                              return size_t(&elem - sparsity_outputs.data());
                           },
      /* getMap: */         [&](auto const& elem){
                            // return the SparsityMap key itself
                            return elem;
                         });

    lhs_rects_instance.destroy();
    input_counters_instance.destroy();
  }

template <int N, typename T>
void GPUIntersectionMicroOp<N, T>::gpu_populate_multiple(void) {
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
    NVTX_DEPPART(gpu_intersection_multiple);

    cudaStream_t stream = Cuda::get_task_cuda_stream();

    collapsed_space<N, T> lhs_space, rhs_space;
    RegionInstance offsets_instance = this->realm_malloc(2 * (inputs.size()+1) * sizeof(size_t), my_mem);
    lhs_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(offsets_instance, 0).base);
    rhs_space.offsets = lhs_space.offsets + (inputs.size() + 1);
    lhs_space.num_children = inputs.size() + 1;
    rhs_space.num_children = inputs.size() + 1;

    RegionInstance lhs_entries_instance, rhs_entries_instance;

    std::vector<IndexSpace<N, T>> tmp_lhs, tmp_rhs;
    for (size_t i = 0; i < inputs.size(); ++i) {
      tmp_lhs.push_back(inputs[i][0]);
      tmp_rhs.push_back(inputs[i][1]);
    }

    GPUMicroOp<N, T>::collapse_multi_space(tmp_lhs, lhs_entries_instance, lhs_space, my_mem, stream);
    GPUMicroOp<N, T>::collapse_multi_space(tmp_rhs, rhs_entries_instance, rhs_space, my_mem, stream);

    RegionInstance lhs_rects_instance = this->realm_malloc(lhs_space.num_entries * sizeof(RectDesc<N,T>), my_mem);
    RectDesc<N, T>* d_lhs_rects = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(lhs_rects_instance, 0).base);

    union_map_rects<N,T><<<COMPUTE_GRID(lhs_space.num_entries), THREADS_PER_BLOCK, 0, stream>>>(lhs_space.entries_buffer, lhs_space.offsets, lhs_space.num_entries, inputs.size(), d_lhs_rects);
    KERNEL_CHECK(stream);

      BVH<N, T> intersect_bvh;
      RegionInstance bvh_instance;
      this->build_bvh(rhs_space, bvh_instance, intersect_bvh, my_mem, stream);

      RegionInstance input_counters_instance = this->realm_malloc(inputs.size() * sizeof(uint32_t), my_mem);
      uint32_t* d_input_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(input_counters_instance, 0).base);
      CUDA_CHECK(cudaMemsetAsync(d_input_counters, 0, inputs.size() * sizeof(uint32_t), stream), stream);

      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      lhs_entries_instance.destroy();
      rhs_entries_instance.destroy();
      offsets_instance.destroy();



      intersect_query_bvh< N, T ><<<COMPUTE_GRID(lhs_space.num_entries), THREADS_PER_BLOCK, 0, stream>>>(d_lhs_rects, intersect_bvh.root, intersect_bvh.childLeft, intersect_bvh.childRight, intersect_bvh.indices, intersect_bvh.labels, intersect_bvh.boxes, lhs_space.num_entries, intersect_bvh.num_leaves, nullptr, d_input_counters, nullptr);
      KERNEL_CHECK(stream);

      uint32_t h_input_counters[inputs.size()+1];
      h_input_counters[0] = 0; // prefix sum starts at 0
      CUDA_CHECK(cudaMemcpyAsync(h_input_counters+1, d_input_counters, inputs.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      for (size_t i = 0; i < inputs.size(); ++i) {
        h_input_counters[i+1] += h_input_counters[i];
      }

      uint32_t num_valid_rects = h_input_counters[inputs.size()];

      if (num_valid_rects==0) {
        for (auto it : sparsity_outputs) {
          SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
          impl->gpu_finalize();
        }
        input_counters_instance.destroy();
        bvh_instance.destroy();
        lhs_rects_instance.destroy();
        return;
      }

      RegionInstance inputs_prefix_instance = this->realm_malloc((inputs.size() + 1) * sizeof(uint32_t), my_mem);
      uint32_t* d_inputs_prefix = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inputs_prefix_instance, 0).base);
      CUDA_CHECK(cudaMemcpyAsync(d_inputs_prefix, h_input_counters, (inputs.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

      RegionInstance output_instance = this->realm_malloc(num_valid_rects * sizeof(RectDesc<N,T>), my_mem);
      RectDesc<N, T>* d_output_rects = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(output_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_input_counters, 0, (inputs.size()) * sizeof(uint32_t), stream), stream);

      intersect_query_bvh< N, T ><<<COMPUTE_GRID(lhs_space.num_entries), THREADS_PER_BLOCK, 0, stream>>>(d_lhs_rects, intersect_bvh.root, intersect_bvh.childLeft, intersect_bvh.childRight, intersect_bvh.indices, intersect_bvh.labels, intersect_bvh.boxes, lhs_space.num_entries, intersect_bvh.num_leaves, d_inputs_prefix, d_input_counters, d_output_rects);
      KERNEL_CHECK(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      input_counters_instance.destroy();
      bvh_instance.destroy();
      lhs_rects_instance.destroy();
      inputs_prefix_instance.destroy();


    this->complete_rect_pipeline(d_output_rects, num_valid_rects, my_mem,
    /* the Container: */  sparsity_outputs,
    /* getIndex: */       [&](auto const& elem){
                            // elem is a SparsityMap<N,T> from the vector
                            return size_t(&elem - sparsity_outputs.data());
                         },
    /* getMap: */         [&](auto const& elem){
                          // return the SparsityMap key itself
                          return elem;
                       });

  output_instance.destroy();

}


  #define DOIT(N,T) \
    template class GPUUnionMicroOp<N,T>; \
    template class GPUIntersectionMicroOp<N,T>;

  FOREACH_NT(DOIT)

}