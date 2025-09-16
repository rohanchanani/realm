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

          std::vector<uint32_t> h_input_counters(inputs.size()+1);
          h_input_counters[0] = 0; // prefix sum starts at 0
          CUDA_CHECK(cudaMemcpyAsync(h_input_counters.data()+1, d_input_counters, inputs.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
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

          CUDA_CHECK(cudaMemcpyAsync(d_inputs_prefix, h_input_counters.data(), (inputs.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

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

template <int N, typename T>
void GPUDifferenceMicroOp<N, T>::gpu_populate(void) {
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

    cudaStream_t stream = Cuda::get_task_cuda_stream();
    NVTX_DEPPART(gpu_difference);

  collapsed_space<N, T> lhs_space, rhs_space;
  RegionInstance offsets_instance = this->realm_malloc(2 * (lhss.size()+1) * sizeof(size_t), my_mem);
  lhs_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(offsets_instance, 0).base);
  rhs_space.offsets = lhs_space.offsets + (lhss.size() + 1);
  lhs_space.num_children = lhss.size() + 1;
  rhs_space.num_children = lhss.size() + 1;

  RegionInstance lhs_entries_instance, rhs_entries_instance;

  GPUMicroOp<N, T>::collapse_multi_space(lhss, lhs_entries_instance, lhs_space, my_mem, stream);
  GPUMicroOp<N, T>::collapse_multi_space(rhss, rhs_entries_instance, rhs_space, my_mem, stream);

  RegionInstance lhs_rects_instance = this->realm_malloc(lhs_space.num_entries * sizeof(RectDesc<N,T>), my_mem);
  RectDesc<N, T>* d_lhs_rects = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(lhs_rects_instance, 0).base);

  union_map_rects<N,T><<<COMPUTE_GRID(lhs_space.num_entries), THREADS_PER_BLOCK, 0, stream>>>(lhs_space.entries_buffer, lhs_space.offsets, lhs_space.num_entries, lhss.size(), d_lhs_rects);
  KERNEL_CHECK(stream);

  BVH<N, T> diff_bvh;
  RegionInstance bvh_instance;
  this->build_bvh(rhs_space, bvh_instance, diff_bvh, my_mem, stream);

  CUDA_CHECK(cudaStreamSynchronize(stream), stream);
  lhs_entries_instance.destroy();
  rhs_entries_instance.destroy();
  offsets_instance.destroy();

      RegionInstance counts_instance = this->realm_malloc(2 * lhs_space.num_entries * sizeof(uint32_t), my_mem);
      uint32_t* d_hit_counts = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(counts_instance, 0).base);
      uint32_t* d_frag_counts = d_hit_counts + lhs_space.num_entries;

      difference_count_hits< N, T ><<<COMPUTE_GRID(lhs_space.num_entries), THREADS_PER_BLOCK, 0, stream>>>(d_lhs_rects, diff_bvh.root, diff_bvh.childLeft, diff_bvh.childRight, diff_bvh.indices, diff_bvh.labels, nullptr, diff_bvh.boxes, lhs_space.num_entries, diff_bvh.num_leaves, d_hit_counts, d_frag_counts, nullptr);
      KERNEL_CHECK(stream);

      RegionInstance prefix_instance = this->realm_malloc(2 * lhs_space.num_entries * sizeof(uint32_t), my_mem);
      uint32_t* d_hits_prefix = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(prefix_instance, 0).base);
      uint32_t* d_frags_prefix = d_hits_prefix + lhs_space.num_entries;

      size_t temp_bytes = 0;
      RegionInstance tmp_instance;
      void* tmp_storage;
      cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_hit_counts, d_hits_prefix, lhs_space.num_entries, stream);
      tmp_instance = this->realm_malloc(temp_bytes, my_mem);
      tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
      cub::DeviceScan::ExclusiveSum(tmp_storage, temp_bytes, d_hit_counts, d_hits_prefix, lhs_space.num_entries, stream);

      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_frag_counts, d_frags_prefix, lhs_space.num_entries, stream);
      tmp_instance.destroy();
      tmp_instance = this->realm_malloc(temp_bytes, my_mem);
      tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
      cub::DeviceScan::ExclusiveSum(tmp_storage, temp_bytes, d_frag_counts, d_frags_prefix, lhs_space.num_entries, stream);

      uint32_t num_hits;
      uint32_t num_frags;
      uint32_t last_hit;
      uint32_t last_frag;

       CUDA_CHECK(cudaMemcpyAsync(&num_hits, d_hits_prefix + lhs_space.num_entries - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaMemcpyAsync(&num_frags, d_frags_prefix + lhs_space.num_entries - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaMemcpyAsync(&last_hit, d_hit_counts + lhs_space.num_entries - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaMemcpyAsync(&last_frag, d_frag_counts + lhs_space.num_entries - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      num_hits += last_hit;
      num_frags += last_frag;

      if (num_hits==0) {
        tmp_instance.destroy();
        prefix_instance.destroy();
        counts_instance.destroy();
        bvh_instance.destroy();

        this->complete_rect_pipeline(d_lhs_rects, lhs_space.num_entries, my_mem,
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
        return;
      }

      RegionInstance hits_instance = this->realm_malloc((num_hits) * sizeof(Rect<N,T>), my_mem);
      Rect<N, T>* global_hits = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(hits_instance, 0).base);

      RegionInstance frags_instance = this->realm_malloc((2*num_frags) * sizeof(RectDesc<N,T>), my_mem);
      RectDesc<N, T>* global_frags_curr = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(frags_instance, 0).base);
      RectDesc<N, T>* global_frags_next = global_frags_curr + num_frags;

      difference_count_hits< N, T ><<<COMPUTE_GRID(lhs_space.num_entries), THREADS_PER_BLOCK, 0, stream>>>(d_lhs_rects, diff_bvh.root, diff_bvh.childLeft, diff_bvh.childRight, diff_bvh.indices, diff_bvh.labels, d_hits_prefix, diff_bvh.boxes, lhs_space.num_entries, diff_bvh.num_leaves, d_hit_counts, d_frag_counts, global_hits);
      KERNEL_CHECK(stream);

      Memory zcpy_mem;
      assert(find_memory(zcpy_mem, Memory::Z_COPY_MEM));

      RegionInstance counter_instance = this->realm_malloc(sizeof(size_t), zcpy_mem);
      RegionInstance choice_instance = this->realm_malloc(sizeof(uint8_t), zcpy_mem);
      size_t* d_counter = reinterpret_cast<size_t*>(AffineAccessor<char,1>(counter_instance, 0).base);
      uint8_t* d_choice = reinterpret_cast<uint8_t*>(AffineAccessor<char,1>(choice_instance, 0).base);

      int dev=0; cudaGetDevice(&dev);
      cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);

      int maxBlocksPerSM = 0;
      int blocksPerSM=0;
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocksPerSM,
          difference_query_bvh<256, N, T>,  // note: template instantiated here
          256, /*dynamicSmemBytes=*/0);

      if (blocksPerSM > 0) {
        maxBlocksPerSM = blocksPerSM;
      } else {
        assert(false);
      }

      // 3) Grid must fit resident at once
      const int sms = prop.multiProcessorCount;
      const int blocksPerSMWanted = std::min(maxBlocksPerSM, 2);  // persistent style
      int gridBlocks = sms * blocksPerSMWanted;

      RegionInstance block_inst = this->realm_malloc((2* gridBlocks + 1) * sizeof(int), my_mem);
      int* g_block_sum = reinterpret_cast<int*>(AffineAccessor<char,1>(block_inst, 0).base);
      int* g_block_base = g_block_sum + gridBlocks;

      RegionInstance max_hits_instance = this->realm_malloc(sizeof(size_t), my_mem);
      uint32_t* d_maxHits = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(max_hits_instance, 0).base);

      void *d_temp_storage = nullptr;
      size_t temp_storage_bytes = 0;

      // First pass: get temp storage size
      cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_hit_counts, d_maxHits, lhs_space.num_entries, stream);

      // Allocate temp storage
      tmp_instance.destroy();
      tmp_instance = this->realm_malloc(temp_storage_bytes, my_mem);
      d_temp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);

      // Actual reduction
      cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_hit_counts, d_maxHits, lhs_space.num_entries, stream);

      // Copy result to host
      uint32_t maxHits;
      cudaMemcpyAsync(&maxHits, d_maxHits, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);

      // 5) Pack args
      void* args[] = {
        (void*)&d_lhs_rects, (void*)&global_hits,
        (void*)&global_frags_curr, (void*)&global_frags_next,
        (void*)&d_hits_prefix, (void*)&d_hit_counts,
        (void*)&lhs_space.num_entries, (void*)&maxHits,
        (void*)&g_block_sum, (void*)&g_block_base,
        (void*)&d_counter, (void*)&d_choice
      };

      // 6) Launch cooperatively
      cudaError_t st;
      st = cudaLaunchCooperativeKernel((void*)difference_query_bvh<256, N, T>,
            dim3(gridBlocks), dim3(256),
                   args, /*sharedMemBytes*/0, /*stream*/stream);

      if (st == cudaErrorCooperativeLaunchTooLarge) {
        // grid too large for full residency → shrink grid or fallback
        gridBlocks = sms * 1;  // try 1 block/SM
        st = cudaLaunchCooperativeKernel((void*)difference_query_bvh<256, N, T>,
              dim3(gridBlocks), dim3(256),
                     args, /*sharedMemBytes*/0, /*stream*/stream);
      }
      if (st != cudaSuccess) {
        // Final fallback if coop still fails (e.g., Windows WDDM, MPS, etc.)
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__  << "cooperative launch failed with " << cudaGetErrorString(st) << " (" << st << ")\n";
        assert(false);
      }

      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      uint8_t choice = *d_choice;
      choice_instance.destroy();

      size_t num_valid_rects = *d_counter;
      counter_instance.destroy();

      RectDesc<N, T>* d_output_rects;
      if (choice==2) {
        d_output_rects = d_lhs_rects; // already in final output
      } else if (choice==1) {
        d_output_rects = global_frags_curr;
      } else {
        d_output_rects = global_frags_next;
      }

      bvh_instance.destroy();
      hits_instance.destroy();
      tmp_instance.destroy();
      prefix_instance.destroy();
      counts_instance.destroy();

      if (num_valid_rects==0) {
        for (auto it : sparsity_outputs) {
          SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
          impl->gpu_finalize();
        }
        lhs_rects_instance.destroy();
        frags_instance.destroy();
        return;
      }


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
  lhs_rects_instance.destroy();
  frags_instance.destroy();

}


  #define DOIT(N,T) \
    template class GPUUnionMicroOp<N,T>; \
    template class GPUIntersectionMicroOp<N,T>; \
    template class GPUDifferenceMicroOp<N, T>;

  FOREACH_NT(DOIT)

}