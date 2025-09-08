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

    // 1) figure out the final size and build the offsets array
    std::vector<size_t> lhs_offsets(lhss.size() + 1);
    std::vector<size_t> rhs_offsets(rhss.size() + 1);
    size_t lhs_size = 0;
    size_t rhs_size = 0;
    for (size_t i = 0; i < lhss.size(); ++i) {
      lhs_offsets[i] = lhs_size;
      rhs_offsets[i] = rhs_size;
      if (lhss[i].dense()) {
        lhs_size += 1;
      } else {
        // only call get_entries() once per input
        lhs_size += lhss[i].sparsity.impl()->get_entries().size();
      }
      if (rhss[i].dense()) {
        rhs_size += 1;
      } else {
        // only call get_entries() once per input
        rhs_size += rhss[i].sparsity.impl()->get_entries().size();
      }
    }
    // final end offset
    lhs_offsets[lhss.size()] = lhs_size;
    rhs_offsets[rhss.size()] = rhs_size;


    // inputs entries allocation
    RegionInstance lhs_entries_instance = this->realm_malloc(lhs_size * sizeof(SparsityMapEntry<N,T>), my_mem);
    SparsityMapEntry<N,T>* d_lhs_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(lhs_entries_instance, 0).base);

    RegionInstance rhs_entries_instance = this->realm_malloc(rhs_size * sizeof(SparsityMapEntry<N,T>), my_mem);
    SparsityMapEntry<N,T>* d_rhs_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(rhs_entries_instance, 0).base);

    // Offsets allocation
    RegionInstance offsets_instance = this->realm_malloc(2*(lhss.size()+1) * sizeof(size_t), my_mem);
    size_t* d_lhs_offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(offsets_instance, 0).base);
    size_t* d_rhs_offsets = d_lhs_offsets + (lhss.size() + 1);

    CUDA_CHECK(cudaMemcpyAsync(d_lhs_offsets, lhs_offsets.data(), (lhss.size()+1) * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);
    CUDA_CHECK(cudaMemcpyAsync(d_rhs_offsets, rhs_offsets.data(), (rhss.size()+1) * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);

    // 3) fill in place
    size_t l_pos = 0;
    size_t r_pos = 0;
    for (size_t i = 0; i < lhss.size(); ++i) {
      if (lhss[i].dense()) {
        // just one rect
        SparsityMapEntry<N,T> entry;
        entry.bounds = lhss[i].bounds;
        CUDA_CHECK(cudaMemcpyAsync(d_lhs_entries + l_pos, &entry, sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
        ++l_pos;
      } else {
          auto& tmp = lhss[i].sparsity.impl()->get_entries();
          CUDA_CHECK(cudaMemcpyAsync(d_lhs_entries + l_pos, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
          l_pos += tmp.size();
      }
      if (rhss[i].dense()) {
        // just one rect
        SparsityMapEntry<N,T> entry;
        entry.bounds = rhss[i].bounds;
        CUDA_CHECK(cudaMemcpyAsync(d_rhs_entries + r_pos, &entry, sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
        ++r_pos;
      } else {
        auto& tmp = rhss[i].sparsity.impl()->get_entries();
        CUDA_CHECK(cudaMemcpyAsync(d_rhs_entries + r_pos, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
        r_pos += tmp.size();
      }
    }

    RegionInstance output_rects_instance = this->realm_malloc((lhs_size+rhs_size) * sizeof(RectDesc<N,T>), my_mem);
    RectDesc<N, T>* d_lhs_rects = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(output_rects_instance, 0).base);
    RectDesc<N, T>* d_rhs_rects = d_lhs_rects + lhs_size;

    int threads_per_block = 256;
    int grid_size = (lhs_size + threads_per_block - 1) / threads_per_block;

    RegionInstance l_out_instance = this->realm_malloc(lhs_size * sizeof(RectDesc<N,T>), my_mem);
    RectDesc<N, T>* d_l_out_rects = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(l_out_instance, 0).base);

    union_map_rects<N,T><<<grid_size, threads_per_block, 0, stream>>>(d_lhs_entries, d_lhs_offsets, lhs_size, lhss.size(), d_l_out_rects);
    KERNEL_CHECK(stream);

    grid_size = (rhs_size + threads_per_block - 1) / threads_per_block;
    union_map_rects<N,T><<<grid_size, threads_per_block, 0, stream>>>(d_rhs_entries, d_rhs_offsets, rhs_size, lhss.size(), d_rhs_rects);
    KERNEL_CHECK(stream);

    Rect<N, T> l_global_bounds = lhss[0].bounds;
    for (size_t i = 1; i < lhss.size(); ++i) {
      l_global_bounds = l_global_bounds.union_bbox(lhss[i].bounds);
    }

    RegionInstance l_bounds_instance = this->realm_malloc(sizeof(Rect<N,T>), my_mem);
    Rect<N,T>* d_l_bounds = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(l_bounds_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_l_bounds, &l_global_bounds, sizeof(Rect<N,T>), cudaMemcpyHostToDevice, stream), stream);

    RegionInstance l_codes_instance = this->realm_malloc(2 * lhs_size * sizeof(uint64_t), my_mem);
    uint64_t* d_l_codes = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(l_codes_instance, 0).base);
    uint64_t* d_l_codes_out = d_l_codes + lhs_size;

    just_morton_codes<<<grid_size, threads_per_block, 0, stream>>>(d_l_out_rects, d_l_bounds, lhs_size, d_l_codes);
    KERNEL_CHECK(stream);


    void *morton_temp = nullptr;
    size_t morton_temp_bytes = 0;
    cub::DeviceRadixSort::SortPairs(morton_temp, morton_temp_bytes, d_l_codes, d_l_codes_out, d_l_out_rects,
                                    d_lhs_rects, lhs_size, 0, 64, stream);
    RegionInstance morton_temp_instance = this->realm_malloc(morton_temp_bytes, my_mem);
    morton_temp = reinterpret_cast<void*>(AffineAccessor<char,1>(morton_temp_instance, 0).base);
    cub::DeviceRadixSort::SortPairs(morton_temp, morton_temp_bytes, d_l_codes, d_l_codes_out, d_l_out_rects,
                                    d_lhs_rects, lhs_size, 0, 64, stream);

    CUDA_CHECK(cudaStreamSynchronize(stream), stream);

    morton_temp_instance.destroy();
    l_bounds_instance.destroy();
    l_codes_instance.destroy();
    l_out_instance.destroy();

    Rect<N, T> global_bounds;
    global_bounds = rhss[0].bounds;
    for (size_t i = 1; i < rhss.size(); ++i) {
      global_bounds = global_bounds.union_bbox(rhss[i].bounds);
    }

    RegionInstance global_bounds_instance = this->realm_malloc(sizeof(Rect<N,T>), my_mem);
    Rect<N,T>* d_global_bounds = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(global_bounds_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_global_bounds, &global_bounds, sizeof(Rect<N,T>), cudaMemcpyHostToDevice, stream), stream);

    RegionInstance morton_codes_instance = this->realm_malloc(rhs_size * sizeof(uint64_t), my_mem);
    uint64_t* d_morton_codes = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(morton_codes_instance, 0).base);

    RegionInstance indices_instance = this->realm_malloc(rhs_size * sizeof(uint64_t), my_mem);
    uint64_t* d_indices = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(indices_instance, 0).base);

    RegionInstance rhs_indices_instance = this->realm_malloc(rhs_size * sizeof(size_t), my_mem);
    size_t* d_rhs_indices = reinterpret_cast<size_t*>(AffineAccessor<char,1>(rhs_indices_instance, 0).base);

    threads_per_block = 256;
    grid_size = (rhs_size + threads_per_block - 1) / threads_per_block;

    bvh_build_morton_codes<<<grid_size, threads_per_block, 0, stream>>>(d_rhs_rects, d_global_bounds, rhs_size, d_morton_codes, d_indices, d_rhs_indices);
    KERNEL_CHECK(stream);

      RegionInstance morton_codes_out_instance = this->realm_malloc(rhs_size * sizeof(uint64_t), my_mem);
      uint64_t* d_morton_codes_out = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(morton_codes_out_instance, 0).base);

      RegionInstance indices_out_instance = this->realm_malloc(rhs_size * sizeof(uint64_t), my_mem);
      uint64_t* d_indices_out = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(indices_out_instance, 0).base);

      void *bvh_temp = nullptr;
      size_t bvh_temp_bytes = 0;
      cub::DeviceRadixSort::SortPairs(bvh_temp, bvh_temp_bytes, d_morton_codes, d_morton_codes_out, d_indices,
                                      d_indices_out, rhs_size, 0, 64, stream);
      RegionInstance bvh_temp_instance = this->realm_malloc(bvh_temp_bytes, my_mem);
      bvh_temp = reinterpret_cast<void*>(AffineAccessor<char,1>(bvh_temp_instance, 0).base);
      cub::DeviceRadixSort::SortPairs(bvh_temp, bvh_temp_bytes, d_morton_codes, d_morton_codes_out, d_indices,
                                      d_indices_out, rhs_size, 0, 64, stream);

      std::swap(d_morton_codes, d_morton_codes_out);
      std::swap(d_indices, d_indices_out);

      RegionInstance childLeft_instance = this->realm_malloc((2*rhs_size - 1) * sizeof(int), my_mem);
      int* d_childLeft = reinterpret_cast<int*>(AffineAccessor<char,1>(childLeft_instance, 0).base);

      RegionInstance childRight_instance = this->realm_malloc((2*rhs_size - 1) * sizeof(int), my_mem);
      int* d_childRight = reinterpret_cast<int*>(AffineAccessor<char,1>(childRight_instance, 0).base);

      RegionInstance parent_instance = this->realm_malloc((2*rhs_size - 1) * sizeof(int), my_mem);
      int* d_parent = reinterpret_cast<int*>(AffineAccessor<char,1>(parent_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_parent, -1, (2*rhs_size - 1) * sizeof(int), stream), stream);

      threads_per_block = 256;
      grid_size = ((rhs_size - 1) + threads_per_block - 1) / threads_per_block;

      int n = (int) rhs_size;
      bvh_build_radix_tree_kernel<<< grid_size, threads_per_block, 0, stream>>>(d_morton_codes, d_indices, n, d_childLeft, d_childRight, d_parent);
      KERNEL_CHECK(stream);

      RegionInstance root_instance = this->realm_malloc(sizeof(int), my_mem);
      int* d_root = reinterpret_cast<int*>(AffineAccessor<char,1>(root_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_root, -1, sizeof(int), stream), stream);

      threads_per_block = 256;
      grid_size = (2 * rhs_size - 1 + threads_per_block - 1) / threads_per_block;
      bvh_build_root_kernel<<< grid_size, threads_per_block, 0, stream>>>(d_root, d_parent, rhs_size);
      KERNEL_CHECK(stream);

      int root;
      CUDA_CHECK(cudaMemcpyAsync(&root, d_root, sizeof(int), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);


      RegionInstance boxes_instance = this->realm_malloc((2*rhs_size - 1) * sizeof(Rect<N,T>), my_mem);
      Rect<N,T>* d_boxes = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(boxes_instance, 0).base);

      threads_per_block = 256;
      grid_size = ((rhs_size) + threads_per_block - 1) / threads_per_block;
      bvh_init_leaf_boxes_kernel<N, T><<<grid_size, threads_per_block, 0, stream>>>(d_rhs_rects, d_indices, rhs_size, d_boxes);
      KERNEL_CHECK(stream);

      RegionInstance visitCount_instance = this->realm_malloc((2*rhs_size - 1) * sizeof(int), my_mem);
      int* d_visitCount = reinterpret_cast<int*>(AffineAccessor<char,1>(visitCount_instance, 0).base);
      CUDA_CHECK(cudaMemsetAsync(d_visitCount, 0, (2*rhs_size - 1) * sizeof(int), stream), stream);

      threads_per_block = 256;
      grid_size = (rhs_size + threads_per_block - 1) / threads_per_block;
      bvh_merge_internal_boxes_kernel < N, T ><<< grid_size, threads_per_block, 0, stream>>>(rhs_size, d_childLeft, d_childRight, d_parent, d_boxes, d_visitCount);
      KERNEL_CHECK(stream);

      RegionInstance hit_counts_instance = this->realm_malloc(lhs_size * sizeof(uint32_t), my_mem);
      uint32_t* d_hit_counts = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(hit_counts_instance, 0).base);

      RegionInstance frag_counts_instance = this->realm_malloc(lhs_size * sizeof(uint32_t), my_mem);
      uint32_t* d_frag_counts = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(frag_counts_instance, 0).base);

      grid_size = (lhs_size + threads_per_block - 1) / threads_per_block;
      difference_count_hits< N, T ><<<grid_size, threads_per_block, 0, stream>>>(d_lhs_rects, d_root, d_childLeft, d_childRight, d_indices, d_rhs_indices, nullptr, d_boxes, lhs_size, rhs_size, d_hit_counts, d_frag_counts, nullptr);
      KERNEL_CHECK(stream);

      RegionInstance prefix_instance = this->realm_malloc(2 * lhs_size * sizeof(uint32_t), my_mem);
      uint32_t* d_hits_prefix = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(prefix_instance, 0).base);
      uint32_t* d_frags_prefix = d_hits_prefix + lhs_size;

      size_t temp_bytes = 0;
      RegionInstance tmp_instance;
      void* tmp_storage;
      cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_hit_counts, d_hits_prefix, lhs_size, stream);
      tmp_instance = this->realm_malloc(temp_bytes, my_mem);
      tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
      cub::DeviceScan::ExclusiveSum(tmp_storage, temp_bytes, d_hit_counts, d_hits_prefix, lhs_size, stream);

      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      cub::DeviceScan::ExclusiveSum(nullptr, temp_bytes, d_frag_counts, d_frags_prefix, lhs_size, stream);
      tmp_instance.destroy();
      tmp_instance = this->realm_malloc(temp_bytes, my_mem);
      tmp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);
      cub::DeviceScan::ExclusiveSum(tmp_storage, temp_bytes, d_frag_counts, d_frags_prefix, lhs_size, stream);

      uint32_t num_hits;
      uint32_t num_frags;
      uint32_t last_hit;
      uint32_t last_frag;

       CUDA_CHECK(cudaMemcpyAsync(&num_hits, d_hits_prefix + lhs_size - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaMemcpyAsync(&num_frags, d_frags_prefix + lhs_size - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaMemcpyAsync(&last_hit, d_hit_counts + lhs_size - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaMemcpyAsync(&last_frag, d_frag_counts + lhs_size - 1, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      num_hits += last_hit;
      num_frags += last_frag;

      if (num_hits==0) {
        morton_codes_instance.destroy();
        indices_instance.destroy();
        rhs_indices_instance.destroy();
        morton_codes_out_instance.destroy();
        indices_out_instance.destroy();
        bvh_temp_instance.destroy();
        childLeft_instance.destroy();
        childRight_instance.destroy();
        parent_instance.destroy();
        root_instance.destroy();
        boxes_instance.destroy();
        visitCount_instance.destroy();
        tmp_instance.destroy();
        prefix_instance.destroy();
        hit_counts_instance.destroy();
        frag_counts_instance.destroy();


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
        lhs_entries_instance.destroy();
        rhs_entries_instance.destroy();
        offsets_instance.destroy();
        output_rects_instance.destroy();
        global_bounds_instance.destroy();
        return;
      }

      RegionInstance hits_instance = this->realm_malloc((num_hits) * sizeof(Rect<N,T>), my_mem);
      Rect<N, T>* global_hits = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(hits_instance, 0).base);

      RegionInstance frags_instance = this->realm_malloc((2*num_frags) * sizeof(RectDesc<N,T>), my_mem);
      RectDesc<N, T>* global_frags_curr = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(frags_instance, 0).base);
      RectDesc<N, T>* global_frags_next = global_frags_curr + num_frags;

      difference_count_hits< N, T ><<<grid_size, threads_per_block, 0, stream>>>(d_lhs_rects, d_root, d_childLeft, d_childRight, d_indices, d_rhs_indices, d_hits_prefix, d_boxes, lhs_size, rhs_size, d_hit_counts, nullptr, global_hits);
      KERNEL_CHECK(stream);

      RegionInstance counter_instance = this->realm_malloc(sizeof(size_t), my_mem);
      RegionInstance choice_instance = this->realm_malloc(sizeof(uint8_t), my_mem);
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
      cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_hit_counts, d_maxHits, lhs_size, stream);

      // Allocate temp storage
      tmp_instance.destroy();
      tmp_instance = this->realm_malloc(temp_storage_bytes, my_mem);
      d_temp_storage = reinterpret_cast<void*>(AffineAccessor<char,1>(tmp_instance, 0).base);

      // Actual reduction
      cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_hit_counts, d_maxHits, lhs_size, stream);

      // Copy result to host
      uint32_t maxHits;
      cudaMemcpyAsync(&maxHits, d_maxHits, sizeof(uint32_t), cudaMemcpyDeviceToHost, stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);

      // 5) Pack args
      void* args[] = {
        (void*)&d_lhs_rects, (void*)&global_hits,
        (void*)&global_frags_curr, (void*)&global_frags_next,
        (void*)&d_hits_prefix, (void*)&d_hit_counts,
        (void*)&lhs_size, (void*)&maxHits,
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

      uint8_t choice;
      CUDA_CHECK(cudaMemcpyAsync(&choice, d_choice, sizeof(uint8_t), cudaMemcpyDeviceToHost, stream), stream);

      size_t num_valid_rects;
      CUDA_CHECK(cudaMemcpyAsync(&num_valid_rects, d_counter, sizeof(size_t), cudaMemcpyDeviceToHost, stream), stream);

      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      RectDesc<N, T>* d_output_rects;
      if (choice==2) {
        d_output_rects = d_lhs_rects; // already in final output
      } else if (choice==1) {
        d_output_rects = global_frags_curr;
      } else {
        d_output_rects = global_frags_next;
      }

      if (num_valid_rects==0) {
        for (auto it : sparsity_outputs) {
          SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
          impl->gpu_finalize();
        }
        morton_codes_instance.destroy();
        indices_instance.destroy();
        rhs_indices_instance.destroy();
        morton_codes_out_instance.destroy();
        indices_out_instance.destroy();
        bvh_temp_instance.destroy();
        childLeft_instance.destroy();
        childRight_instance.destroy();
        parent_instance.destroy();
        root_instance.destroy();
        boxes_instance.destroy();
        visitCount_instance.destroy();
        hits_instance.destroy();
        frags_instance.destroy();
        tmp_instance.destroy();
        prefix_instance.destroy();
        hit_counts_instance.destroy();
        frag_counts_instance.destroy();
        lhs_entries_instance.destroy();
        rhs_entries_instance.destroy();
        offsets_instance.destroy();
        output_rects_instance.destroy();
        global_bounds_instance.destroy();
        return;
      }
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      morton_codes_instance.destroy();
      indices_instance.destroy();
      rhs_indices_instance.destroy();
      morton_codes_out_instance.destroy();
      indices_out_instance.destroy();
      bvh_temp_instance.destroy();
      childLeft_instance.destroy();
      childRight_instance.destroy();
      parent_instance.destroy();
      root_instance.destroy();
      boxes_instance.destroy();
      visitCount_instance.destroy();
      hits_instance.destroy();
      tmp_instance.destroy();
      prefix_instance.destroy();
      hit_counts_instance.destroy();
      frag_counts_instance.destroy();


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
  lhs_entries_instance.destroy();
  rhs_entries_instance.destroy();
  offsets_instance.destroy();
  output_rects_instance.destroy();
  global_bounds_instance.destroy();
  frags_instance.destroy();

}


  #define DOIT(N,T) \
    template class GPUUnionMicroOp<N,T>; \
    template class GPUIntersectionMicroOp<N,T>; \
    template class GPUDifferenceMicroOp<N, T>;

  FOREACH_NT(DOIT)

}