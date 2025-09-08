#pragma once
#include "realm/deppart/preimage.h"
#include "realm/deppart/preimage_gpu_kernels.hpp"
#include "realm/deppart/byfield_gpu_kernels.hpp"
#include "realm/deppart/partitions_gpu_impl.hpp"
#include <cub/cub.cuh>
#include <sstream>
#include "realm/nvtx.h"

namespace Realm {
  template<int N2, typename T2>
  struct RectVolumeOp {
    __device__ __forceinline__ size_t operator()(const Rect<N2, T2> &rd) const {
      return rd.volume();
    }
  };

  template<int N, typename T, int N2, typename T2>
  void GPUPreimageMicroOp<N, T, N2, T2>::gpu_populate_ranges()
  {
    if (targets.size() == 0) {
      return;
    }
    Memory my_mem = domain_transform.range_data[0].inst.get_location();

    nvtx_range_push("cuda", "preimage_populate_bitmasks_ranges");

    nvtx_range_push("cuda", "create stream");
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), stream);
    nvtx_range_pop();
    nvtx_range_push("cuda", "build parent and instance entries");


    std::vector<size_t> inst_offsets(domain_transform.range_data.size() + 1);
    size_t inst_size = 0;
    for (size_t i = 0; i < domain_transform.range_data.size(); ++i) {
      inst_offsets[i] = inst_size;
      if (domain_transform.range_data[i].index_space.dense()) {
        inst_size += 1;
      } else {
        // only call get_entries() once per source
        inst_size += domain_transform.range_data[i].index_space.sparsity.impl()->get_entries().size();
      }
    }
    // final end offset
    inst_offsets[domain_transform.range_data.size()] = inst_size;

    RegionInstance inst_entries_instance = this->realm_malloc(sizeof(SparsityMapEntry<N, T>) * inst_size, my_mem);
    SparsityMapEntry<N, T> *d_inst_entries = reinterpret_cast<SparsityMapEntry<N, T> *>(AffineAccessor<char, 1>(inst_entries_instance, 0).base);

    size_t inst_pos = 0;
    for (size_t i = 0; i < domain_transform.range_data.size(); ++i) {
      IndexSpace<N, T> &inst_space = domain_transform.range_data[i].index_space;
      if (inst_space.dense()) {
        // just one rect
        SparsityMapEntry<N, T> entry;
        entry.bounds = inst_space.bounds;
        CUDA_CHECK(
          cudaMemcpyAsync(d_inst_entries + inst_pos, &entry, sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice,
            stream), stream);
        ++inst_pos;
      } else {
        auto tmp = inst_space.sparsity.impl()->get_entries();
        CUDA_CHECK(
          cudaMemcpyAsync(d_inst_entries + inst_pos, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>),
            cudaMemcpyHostToDevice, stream), stream);
        inst_pos += tmp.size();
      }
    }

    span<SparsityMapEntry<N,T>> parent_entries;
    std::vector<SparsityMapEntry<N,T>> parent_entries_vec;
    RegionInstance parent_entries_instance;
    SparsityMapEntry<N,T>* d_parent_entries;
    if (parent_space.dense()) {
      SparsityMapEntry<N,T> entry;
      entry.bounds = parent_space.bounds;
      parent_entries_vec = {entry};
      parent_entries = span<SparsityMapEntry<N,T>>(parent_entries_vec.data(), 1);
    } else {
      parent_entries = parent_space.sparsity.impl()->get_entries();
    }
    parent_entries_instance = this->realm_malloc(parent_entries.size() * sizeof(SparsityMapEntry<N,T>), my_mem);
    d_parent_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(parent_entries_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_parent_entries, parent_entries.data(), parent_entries.size() * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
    nvtx_range_pop();

    nvtx_range_push("cuda", "Build and intersect input rectangles");


    RegionInstance inst_counters_instance = this->realm_malloc((domain_transform.range_data.size()) * sizeof(uint32_t), my_mem);
    uint32_t* d_inst_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inst_counters_instance, 0).base);
    CUDA_CHECK(cudaMemsetAsync(d_inst_counters, 0, (domain_transform.range_data.size()) * sizeof(uint32_t), stream), stream);

    RegionInstance inst_offsets_instance = this->realm_malloc((domain_transform.range_data.size() + 1) * sizeof(size_t), my_mem);
    size_t* d_inst_offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(inst_offsets_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_inst_offsets, inst_offsets.data(), (domain_transform.range_data.size() + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);

    int flattened_threads = 256;
    int grid_size = (parent_entries.size() * inst_size + flattened_threads - 1) / flattened_threads;
    intersect_input_rects<N,T><<<grid_size, flattened_threads, 0, stream>>>(d_inst_entries, d_parent_entries, d_inst_offsets, nullptr, inst_size, parent_entries.size(), domain_transform.range_data.size(), d_inst_counters, nullptr);
    KERNEL_CHECK(stream);

    uint32_t h_inst_counters[domain_transform.range_data.size()+1];
    h_inst_counters[0] = 0; // prefix sum starts at 0
    CUDA_CHECK(cudaMemcpyAsync(h_inst_counters+1, d_inst_counters, domain_transform.range_data.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    for (size_t i = 0; i < domain_transform.range_data.size(); ++i) {
      h_inst_counters[i+1] += h_inst_counters[i];
    }

    size_t num_valid_rects = h_inst_counters[domain_transform.range_data.size()];

    if (num_valid_rects == 0) {
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      for (auto it : sparsity_outputs) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
        impl->gpu_finalize();
      }
      nvtx_range_pop();
      nvtx_range_pop();
      cudaStreamDestroy(stream);
      return;
    }

    RegionInstance inst_prefix_instance = this->realm_malloc((domain_transform.range_data.size() + 1) * sizeof(uint32_t), my_mem);
    uint32_t* d_inst_prefix = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inst_prefix_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_inst_prefix, h_inst_counters, (domain_transform.range_data.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

    RegionInstance valid_rects_instance = this->realm_malloc(num_valid_rects * sizeof(Rect<N,T>), my_mem);
    Rect<N,T>* d_valid_rects = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(valid_rects_instance, 0).base);

    CUDA_CHECK(cudaMemsetAsync(d_inst_counters, 0, (domain_transform.range_data.size()) * sizeof(uint32_t), stream), stream);

    flattened_threads = 256;
    grid_size = (parent_entries.size() * inst_size + flattened_threads - 1) / flattened_threads;
    intersect_input_rects<N,T><<<grid_size, flattened_threads, 0, stream>>>(d_inst_entries, d_parent_entries, d_inst_offsets, d_inst_prefix, inst_size, parent_entries.size(), domain_transform.range_data.size(), d_inst_counters, d_valid_rects);
    KERNEL_CHECK(stream);

    nvtx_range_pop();
    nvtx_range_push("cuda", "Collect valid rectangles");

    RegionInstance prefix_rects_instance = this->realm_malloc((num_valid_rects + 1) * sizeof(size_t), my_mem);
    size_t* d_prefix_rects = reinterpret_cast<size_t*>(AffineAccessor<char,1>(prefix_rects_instance, 0).base);
    CUDA_CHECK(cudaMemsetAsync(d_prefix_rects, 0, sizeof(size_t), stream), stream);

    // 3) build the CUB transform‐iterator
    using VolIter = cub::TransformInputIterator<
      size_t, // output type
      RectVolumeOp<N, T>, // functor
      Rect<N, T> * // underlying input iterator
    >;
    VolIter d_volumes(d_valid_rects, RectVolumeOp<N, T>());

    // 3) get temporary storage size
    void *d_temp = nullptr;
    size_t rect_temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(
      /* d_temp_storage */ nullptr,
                           /* temp_bytes */ rect_temp_bytes,
                           /* d_in */ d_volumes,
                           /* d_out */ d_prefix_rects + 1, // shift by one so prefix[1]..prefix[n]
                           /* num_items */ num_valid_rects, stream);

    // 4) allocate and run
    RegionInstance temp_instance = this->realm_malloc(rect_temp_bytes, my_mem);
    d_temp = reinterpret_cast<void*>(AffineAccessor<char,1>(temp_instance, 0).base);
    cub::DeviceScan::InclusiveSum(
      /* d_temp_storage */ d_temp,
                           /* temp_bytes */ rect_temp_bytes,
                           /* d_in */ d_volumes,
                           /* d_out */ d_prefix_rects + 1,
                           /* num_items */ num_valid_rects, stream);

    nvtx_range_pop();

    nvtx_range_push("cuda", "Prepare to populate bitmasks");
    size_t total_pts;
    CUDA_CHECK(
      cudaMemcpyAsync(&total_pts, &d_prefix_rects[num_valid_rects], sizeof(size_t), cudaMemcpyDeviceToHost, stream),
      stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);

    nvtx_range_pop();

    nvtx_range_push("cuda", "build target entries");
    std::vector<size_t> target_offsets(targets.size() + 1);
    size_t total_size = 0;
    for (size_t i = 0; i < targets.size(); ++i) {
      target_offsets[i] = total_size;
      if (targets[i].dense()) {
        total_size += 1;
      } else {
        // only call get_entries() once per source
        total_size += targets[i].sparsity.impl()->get_entries().size();
      }
    }

    // final end offset
    target_offsets[targets.size()] = total_size;

    Rect<N2, T2> global_bounds = targets[0].bounds;
    for (size_t i = 1; i < targets.size(); i++) {
      global_bounds.union_bbox(targets[i].bounds);
    }

    RegionInstance global_bounds_instance = this->realm_malloc(sizeof(Rect<N2,T2>), my_mem);
    Rect<N2,T2>* d_global_bounds = reinterpret_cast<Rect<N2,T2>*>(AffineAccessor<char,1>(global_bounds_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_global_bounds, &global_bounds, sizeof(Rect<N2,T2>), cudaMemcpyHostToDevice, stream),
               stream);


    RegionInstance targets_entries_instance = this->realm_malloc(total_size * sizeof(SparsityMapEntry<N2,T2>), my_mem);
    SparsityMapEntry<N2,T2>* d_targets_entries = reinterpret_cast<SparsityMapEntry<N2,T2>*>(AffineAccessor<char,1>(targets_entries_instance, 0).base);

    RegionInstance offsets_instance = this->realm_malloc((targets.size() + 1) * sizeof(size_t), my_mem);
    size_t* d_offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(offsets_instance, 0).base);

    CUDA_CHECK(
      cudaMemcpyAsync(d_offsets, target_offsets.data(), (targets.size()+1) * sizeof(size_t), cudaMemcpyHostToDevice,
        stream), stream);

    // 3) fill in place
    size_t pos = 0;
    for (size_t i = 0; i < targets.size(); ++i) {
      if (targets[i].dense()) {
        // just one rect
        SparsityMapEntry<N2, T2> entry;
        entry.bounds = targets[i].bounds;
        CUDA_CHECK(
          cudaMemcpyAsync(d_targets_entries + pos, &entry, sizeof(SparsityMapEntry<N2,T2>), cudaMemcpyHostToDevice,
            stream), stream);
        ++pos;
      } else {
        auto tmp = targets[i].sparsity.impl()->get_entries();
        CUDA_CHECK(
          cudaMemcpyAsync(d_targets_entries + pos, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N2,T2>),
            cudaMemcpyHostToDevice, stream), stream);
        pos += tmp.size();
      }
      // pos should now equal source_offsets[i+1]
    }

    AffineAccessor<Rect<N2, T2>, N, T> h_accessors[domain_transform.range_data.size()];
    for (size_t i = 0; i < domain_transform.range_data.size(); ++i) {
      h_accessors[i] = AffineAccessor<Rect<N2, T2>, N, T>(domain_transform.range_data[i].inst,
                                                           domain_transform.range_data[i].field_offset);
    }

    RegionInstance accessors_instance = this->realm_malloc(domain_transform.range_data.size() * sizeof(AffineAccessor<Rect<N2,T2>,N,T>), my_mem);
    AffineAccessor<Rect<N2,T2>,N,T>* d_accessors = reinterpret_cast<AffineAccessor<Rect<N2,T2>,N,T>*>(AffineAccessor<char,1>(accessors_instance, 0).base);
    CUDA_CHECK(
      cudaMemcpyAsync(d_accessors, h_accessors, domain_transform.range_data.size() * sizeof(AffineAccessor<Rect<N2,T2>,N,
        T>), cudaMemcpyHostToDevice, stream), stream);

    nvtx_range_pop();

    RegionInstance points_instance;
    PointDesc<N,T>* d_points;
    size_t num_valid_points;

    if (total_size > targets.size()) {
      nvtx_range_push("cuda", "Sparse case");
      RegionInstance morton_codes_instance = this->realm_malloc(total_size * sizeof(uint64_t), my_mem);
      uint64_t* d_morton_codes = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(morton_codes_instance, 0).base);

      RegionInstance indices_instance = this->realm_malloc(total_size * sizeof(uint64_t), my_mem);
      uint64_t* d_indices = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(indices_instance, 0).base);

      RegionInstance targets_indices_instance = this->realm_malloc(total_size * sizeof(size_t), my_mem);
      size_t* d_targets_indices = reinterpret_cast<size_t*>(AffineAccessor<char,1>(targets_indices_instance, 0).base);

      flattened_threads = 256;
      grid_size = (total_size + flattened_threads - 1) / flattened_threads;

      preimage_build_morton_codes<N2, T2><<<grid_size, flattened_threads, 0, stream>>>(
        d_targets_entries, d_offsets, d_global_bounds, total_size, targets.size(), d_morton_codes, d_indices,
        d_targets_indices);
      KERNEL_CHECK(stream);

      RegionInstance morton_codes_out_instance = this->realm_malloc(total_size * sizeof(uint64_t), my_mem);
      uint64_t* d_morton_codes_out = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(morton_codes_out_instance, 0).base);

      RegionInstance indices_out_instance = this->realm_malloc(total_size * sizeof(uint64_t), my_mem);
      uint64_t* d_indices_out = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(indices_out_instance, 0).base);

      void *bvh_temp = nullptr;
      size_t bvh_temp_bytes = 0;
      cub::DeviceRadixSort::SortPairs(bvh_temp, bvh_temp_bytes, d_morton_codes, d_morton_codes_out, d_indices,
                                      d_indices_out, total_size, 0, 64, stream);
      RegionInstance bvh_temp_instance = this->realm_malloc(bvh_temp_bytes, my_mem);
      bvh_temp = reinterpret_cast<void*>(AffineAccessor<char,1>(bvh_temp_instance, 0).base);
      cub::DeviceRadixSort::SortPairs(bvh_temp, bvh_temp_bytes, d_morton_codes, d_morton_codes_out, d_indices,
                                      d_indices_out, total_size, 0, 64, stream);

      std::swap(d_morton_codes, d_morton_codes_out);
      std::swap(d_indices, d_indices_out);

      RegionInstance childLeft_instance = this->realm_malloc((2*total_size - 1) * sizeof(int), my_mem);
      int* d_childLeft = reinterpret_cast<int*>(AffineAccessor<char,1>(childLeft_instance, 0).base);

      RegionInstance childRight_instance = this->realm_malloc((2*total_size - 1) * sizeof(int), my_mem);
      int* d_childRight = reinterpret_cast<int*>(AffineAccessor<char,1>(childRight_instance, 0).base);

      RegionInstance parent_instance = this->realm_malloc((2*total_size - 1) * sizeof(int), my_mem);
      int* d_parent = reinterpret_cast<int*>(AffineAccessor<char,1>(parent_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_parent, -1, (2*total_size - 1) * sizeof(int), stream), stream);

      flattened_threads = 256;
      grid_size = ((total_size - 1) + flattened_threads - 1) / flattened_threads;

      int n = (int) total_size;
      bvh_build_radix_tree_kernel<<< grid_size, flattened_threads, 0, stream>>>(d_morton_codes, d_indices, n, d_childLeft, d_childRight, d_parent);
      KERNEL_CHECK(stream);

      RegionInstance root_instance = this->realm_malloc(sizeof(int), my_mem);
      int* d_root = reinterpret_cast<int*>(AffineAccessor<char,1>(root_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_root, -1, sizeof(int), stream), stream);

      flattened_threads = 256;
      grid_size = (2 * total_size - 1 + flattened_threads - 1) / flattened_threads;
      bvh_build_root_kernel<<< grid_size, flattened_threads, 0, stream>>>(d_root, d_parent, total_size);
      KERNEL_CHECK(stream);

      int root;
      CUDA_CHECK(cudaMemcpyAsync(&root, d_root, sizeof(int), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);


      RegionInstance boxes_instance = this->realm_malloc((2*total_size - 1) * sizeof(Rect<N2,T2>), my_mem);
      Rect<N2,T2>* d_boxes = reinterpret_cast<Rect<N2,T2>*>(AffineAccessor<char,1>(boxes_instance, 0).base);

      flattened_threads = 256;
      grid_size = ((total_size) + flattened_threads - 1) / flattened_threads;
      preimage_init_leaf_boxes_kernel<N2, T2><<<grid_size, flattened_threads, 0, stream>>>(d_targets_entries, d_indices, total_size, d_boxes);
      KERNEL_CHECK(stream);

      RegionInstance visitCount_instance = this->realm_malloc((2*total_size - 1) * sizeof(int), my_mem);
      int* d_visitCount = reinterpret_cast<int*>(AffineAccessor<char,1>(visitCount_instance, 0).base);
      CUDA_CHECK(cudaMemsetAsync(d_visitCount, 0, (2*total_size - 1) * sizeof(int), stream), stream);

      flattened_threads = 256;
      grid_size = (total_size + flattened_threads - 1) / flattened_threads;
      bvh_merge_internal_boxes_kernel < N2, T2 ><<< grid_size, flattened_threads, 0, stream>>>(total_size, d_childLeft, d_childRight, d_parent, d_boxes, d_visitCount);
      KERNEL_CHECK(stream);

      RegionInstance target_counters_instance = this->realm_malloc(targets.size() * sizeof(uint32_t), my_mem);
      uint32_t* d_target_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(target_counters_instance, 0).base);
      CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, targets.size() * sizeof(uint32_t), stream), stream);


      int threads_per_block = 256;
      int num_blocks = (total_pts + threads_per_block - 1) / threads_per_block;
      preimage_gpuPopulateBitmasksPtrsKernel < N, T, N2, T2 ><<<num_blocks, threads_per_block, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, d_root, d_childLeft, d_childRight, d_indices,
       d_targets_indices, d_boxes, total_pts, num_valid_rects, domain_transform.range_data.size(), total_size, nullptr, d_target_counters, nullptr);
      KERNEL_CHECK(stream);

      uint32_t h_target_counters[targets.size()+1];
      h_target_counters[0] = 0; // prefix sum starts at 0
      CUDA_CHECK(cudaMemcpyAsync(h_target_counters+1, d_target_counters, targets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      for (size_t i = 0; i < targets.size(); ++i) {
        h_target_counters[i+1] += h_target_counters[i];
      }

      num_valid_points = h_target_counters[targets.size()];

      if (num_valid_points == 0) {
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        for (auto it : sparsity_outputs) {
          SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
          impl->gpu_finalize();
        }
        nvtx_range_pop();
        nvtx_range_pop();
        cudaStreamDestroy(stream);
        return;
      }

      RegionInstance targets_prefix_instance = this->realm_malloc((targets.size() + 1) * sizeof(uint32_t), my_mem);
      uint32_t* d_targets_prefix = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(targets_prefix_instance, 0).base);
      CUDA_CHECK(cudaMemcpyAsync(d_targets_prefix, h_target_counters, (targets.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

      points_instance = this->realm_malloc(num_valid_points * sizeof(PointDesc<N,T>), my_mem);
      d_points = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(points_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, (targets.size()) * sizeof(uint32_t), stream), stream);

      preimage_gpuPopulateBitmasksPtrsKernel < N, T, N2, T2 ><<<num_blocks, threads_per_block, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, d_root, d_childLeft, d_childRight, d_indices,
       d_targets_indices, d_boxes, total_pts, num_valid_rects, domain_transform.range_data.size(), total_size, d_targets_prefix, d_target_counters, d_points);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      morton_codes_instance.destroy();
      indices_instance.destroy();
      targets_indices_instance.destroy();
      morton_codes_out_instance.destroy();
      indices_out_instance.destroy();
      bvh_temp_instance.destroy();
      childLeft_instance.destroy();
      childRight_instance.destroy();
      parent_instance.destroy();
      root_instance.destroy();
      boxes_instance.destroy();
      visitCount_instance.destroy();
      target_counters_instance.destroy();
      targets_prefix_instance.destroy();
      nvtx_range_pop();
    } else {
      nvtx_range_push("cuda", "dense case");

      RegionInstance target_counters_instance = this->realm_malloc(targets.size() * sizeof(uint32_t), my_mem);
      uint32_t* d_target_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(target_counters_instance, 0).base);
      CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, targets.size() * sizeof(uint32_t), stream), stream);

      int threads_per_block = 256;
      int num_blocks = (total_pts + threads_per_block - 1) / threads_per_block;
      preimage_dense_populate_bitmasks_kernel < N, T, N2, T2 ><<< num_blocks, threads_per_block, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, d_targets_entries, d_offsets, total_pts,
       num_valid_rects, domain_transform.range_data.size(), targets.size(), nullptr, d_target_counters, nullptr);
      KERNEL_CHECK(stream);

      uint32_t h_target_counters[targets.size()+1];
      h_target_counters[0] = 0; // prefix sum starts at 0
      CUDA_CHECK(cudaMemcpyAsync(h_target_counters+1, d_target_counters, targets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      for (size_t i = 0; i < targets.size(); ++i) {
        h_target_counters[i+1] += h_target_counters[i];
      }

      num_valid_points = h_target_counters[targets.size()];

      if (num_valid_points == 0) {
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        for (auto it : sparsity_outputs) {
          SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
          impl->gpu_finalize();
        }
        nvtx_range_pop();
        nvtx_range_pop();
        cudaStreamDestroy(stream);
        return;
      }

      RegionInstance targets_prefix_instance = this->realm_malloc((targets.size() + 1) * sizeof(uint32_t), my_mem);
      uint32_t* d_targets_prefix = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(targets_prefix_instance, 0).base);
      CUDA_CHECK(cudaMemcpyAsync(d_targets_prefix, h_target_counters, (targets.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

      points_instance = this->realm_malloc(num_valid_points * sizeof(PointDesc<N,T>), my_mem);
      d_points = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(points_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, (targets.size()) * sizeof(uint32_t), stream), stream);

      preimage_dense_populate_bitmasks_kernel < N, T, N2, T2 ><<< num_blocks, threads_per_block, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, d_targets_entries, d_offsets, total_pts,
       num_valid_rects, domain_transform.range_data.size(), targets.size(), d_targets_prefix, d_target_counters, d_points);
      KERNEL_CHECK(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);

      nvtx_range_pop();
      target_counters_instance.destroy();
      targets_prefix_instance.destroy();
    }
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    cudaStreamDestroy(stream);
    inst_entries_instance.destroy();
    parent_entries_instance.destroy();
    inst_offsets_instance.destroy();
    valid_rects_instance.destroy();
    prefix_rects_instance.destroy();
    temp_instance.destroy();
    global_bounds_instance.destroy();
    targets_entries_instance.destroy();
    offsets_instance.destroy();
    accessors_instance.destroy();
    nvtx_range_pop();

    this->complete_pipeline(d_points, num_valid_points, my_mem,
    /* the Container: */  sparsity_outputs,
    /* getIndex: */       [&](auto const& elem){
                            // elem is a SparsityMap<N,T> from the vector
                            return size_t(&elem - sparsity_outputs.data());
                         },
    /* getMap: */         [&](auto const& elem){
                          // return the SparsityMap key itself
                          return elem;
                       });

    nvtx_range_push("cuda", "free");
    // Destroy all Realm RegionInstances (except loop‐allocated ones)
    points_instance.destroy();
    nvtx_range_pop();
    nvtx_range_pop();
  }

  template<int N, typename T, int N2, typename T2>
  void GPUPreimageMicroOp<N, T, N2, T2>::gpu_populate_bitmasks() {
    if (targets.size() == 0) {
      return;
    }

    Memory my_mem = domain_transform.ptr_data[0].inst.get_location();

    nvtx_range_push("cuda", "preimage_populate_bitmasks_ptrs");

    nvtx_range_push("cuda", "create stream");
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream), stream);
    nvtx_range_pop();
    nvtx_range_push("cuda", "build parent and instance entries");


    std::vector<size_t> inst_offsets(domain_transform.ptr_data.size() + 1);
    size_t inst_size = 0;
    for (size_t i = 0; i < domain_transform.ptr_data.size(); ++i) {
      inst_offsets[i] = inst_size;
      if (domain_transform.ptr_data[i].index_space.dense()) {
        inst_size += 1;
      } else {
        // only call get_entries() once per source
        inst_size += domain_transform.ptr_data[i].index_space.sparsity.impl()->get_entries().size();
      }
    }
    // final end offset
    inst_offsets[domain_transform.ptr_data.size()] = inst_size;

    RegionInstance inst_entries_instance = this->realm_malloc(sizeof(SparsityMapEntry<N, T>) * inst_size, my_mem);
    SparsityMapEntry<N, T> *d_inst_entries = reinterpret_cast<SparsityMapEntry<N, T> *>(AffineAccessor<char, 1>(inst_entries_instance, 0).base);

    size_t inst_pos = 0;
    for (size_t i = 0; i < domain_transform.ptr_data.size(); ++i) {
      IndexSpace<N, T> &inst_space = domain_transform.ptr_data[i].index_space;
      if (inst_space.dense()) {
        // just one rect
        SparsityMapEntry<N, T> entry;
        entry.bounds = inst_space.bounds;
        CUDA_CHECK(
          cudaMemcpyAsync(d_inst_entries + inst_pos, &entry, sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice,
            stream), stream);
        ++inst_pos;
      } else {
         auto tmp = inst_space.sparsity.impl()->get_entries();
         CUDA_CHECK(
           cudaMemcpyAsync(d_inst_entries + inst_pos, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N,T>),
             cudaMemcpyHostToDevice, stream), stream);
         inst_pos += tmp.size();
      }
    }

    span<SparsityMapEntry<N,T>> parent_entries;
    std::vector<SparsityMapEntry<N,T>> parent_entries_vec;
    RegionInstance parent_entries_instance;
    SparsityMapEntry<N,T>* d_parent_entries;
    if (parent_space.dense()) {
      SparsityMapEntry<N,T> entry;
      entry.bounds = parent_space.bounds;
      parent_entries_vec = {entry};
      parent_entries = span<SparsityMapEntry<N,T>>(parent_entries_vec.data(), 1);
    } else {
      parent_entries = parent_space.sparsity.impl()->get_entries();
    }
    parent_entries_instance = this->realm_malloc(parent_entries.size() * sizeof(SparsityMapEntry<N,T>), my_mem);
    d_parent_entries = reinterpret_cast<SparsityMapEntry<N,T>*>(AffineAccessor<char,1>(parent_entries_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_parent_entries, parent_entries.data(), parent_entries.size() * sizeof(SparsityMapEntry<N,T>), cudaMemcpyHostToDevice, stream), stream);
    nvtx_range_pop();

    nvtx_range_push("cuda", "Build and intersect input rectangles");

    RegionInstance inst_counters_instance = this->realm_malloc((domain_transform.ptr_data.size()) * sizeof(uint32_t), my_mem);
    uint32_t* d_inst_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inst_counters_instance, 0).base);
    CUDA_CHECK(cudaMemsetAsync(d_inst_counters, 0, (domain_transform.ptr_data.size()) * sizeof(uint32_t), stream), stream);

    RegionInstance inst_offsets_instance = this->realm_malloc((domain_transform.ptr_data.size() + 1) * sizeof(size_t), my_mem);
    size_t* d_inst_offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(inst_offsets_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_inst_offsets, inst_offsets.data(), (domain_transform.ptr_data.size() + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);

    int flattened_threads = 256;
    int grid_size = (parent_entries.size() * inst_size + flattened_threads - 1) / flattened_threads;
    intersect_input_rects<N,T><<<grid_size, flattened_threads, 0, stream>>>(d_inst_entries, d_parent_entries, d_inst_offsets, nullptr, inst_size, parent_entries.size(), domain_transform.ptr_data.size(), d_inst_counters, nullptr);
    KERNEL_CHECK(stream);

    uint32_t h_inst_counters[domain_transform.ptr_data.size()+1];
    h_inst_counters[0] = 0; // prefix sum starts at 0
    CUDA_CHECK(cudaMemcpyAsync(h_inst_counters+1, d_inst_counters, domain_transform.ptr_data.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    for (size_t i = 0; i < domain_transform.ptr_data.size(); ++i) {
      h_inst_counters[i+1] += h_inst_counters[i];
    }

    size_t num_valid_rects = h_inst_counters[domain_transform.ptr_data.size()];

    if (num_valid_rects == 0) {
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      for (auto it : sparsity_outputs) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
        impl->gpu_finalize();
      }
      nvtx_range_pop();
      nvtx_range_pop();
      cudaStreamDestroy(stream);
      return;
    }

    RegionInstance inst_prefix_instance = this->realm_malloc((domain_transform.ptr_data.size() + 1) * sizeof(uint32_t), my_mem);
    uint32_t* d_inst_prefix = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inst_prefix_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_inst_prefix, h_inst_counters, (domain_transform.ptr_data.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

    RegionInstance valid_rects_instance = this->realm_malloc(num_valid_rects * sizeof(Rect<N,T>), my_mem);
    Rect<N,T>* d_valid_rects = reinterpret_cast<Rect<N,T>*>(AffineAccessor<char,1>(valid_rects_instance, 0).base);

    CUDA_CHECK(cudaMemsetAsync(d_inst_counters, 0, (domain_transform.ptr_data.size()) * sizeof(uint32_t), stream), stream);

    flattened_threads = 256;
    grid_size = (parent_entries.size() * inst_size + flattened_threads - 1) / flattened_threads;
    intersect_input_rects<N,T><<<grid_size, flattened_threads, 0, stream>>>(d_inst_entries, d_parent_entries, d_inst_offsets, d_inst_prefix, inst_size, parent_entries.size(), domain_transform.ptr_data.size(), d_inst_counters, d_valid_rects);
    KERNEL_CHECK(stream);

    nvtx_range_pop();
    nvtx_range_push("cuda", "Collect valid rectangles");

    RegionInstance prefix_rects_instance = this->realm_malloc((num_valid_rects + 1) * sizeof(size_t), my_mem);
    size_t* d_prefix_rects = reinterpret_cast<size_t*>(AffineAccessor<char,1>(prefix_rects_instance, 0).base);
    CUDA_CHECK(cudaMemsetAsync(d_prefix_rects, 0, sizeof(size_t), stream), stream);

    // 3) build the CUB transform‐iterator
    using VolIter = cub::TransformInputIterator<
      size_t, // output type
      RectVolumeOp<N, T>, // functor
      Rect<N, T> * // underlying input iterator
    >;
    VolIter d_volumes(d_valid_rects, RectVolumeOp<N, T>());

    // 3) get temporary storage size
    void *d_temp = nullptr;
    size_t rect_temp_bytes = 0;
    cub::DeviceScan::InclusiveSum(
      /* d_temp_storage */ nullptr,
                           /* temp_bytes */ rect_temp_bytes,
                           /* d_in */ d_volumes,
                           /* d_out */ d_prefix_rects + 1, // shift by one so prefix[1]..prefix[n]
                           /* num_items */ num_valid_rects, stream);

    // 4) allocate and run
    RegionInstance temp_instance = this->realm_malloc(rect_temp_bytes, my_mem);
    d_temp = reinterpret_cast<void*>(AffineAccessor<char,1>(temp_instance, 0).base);
    cub::DeviceScan::InclusiveSum(
      /* d_temp_storage */ d_temp,
                           /* temp_bytes */ rect_temp_bytes,
                           /* d_in */ d_volumes,
                           /* d_out */ d_prefix_rects + 1,
                           /* num_items */ num_valid_rects, stream);

    nvtx_range_pop();

    nvtx_range_push("cuda", "Prepare to populate bitmasks");
    size_t total_pts;
    CUDA_CHECK(
      cudaMemcpyAsync(&total_pts, &d_prefix_rects[num_valid_rects], sizeof(size_t), cudaMemcpyDeviceToHost, stream),
      stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);

    nvtx_range_pop();

    nvtx_range_push("cuda", "build target entries");
    std::vector<size_t> target_offsets(targets.size() + 1);
    size_t total_size = 0;
    for (size_t i = 0; i < targets.size(); ++i) {
      target_offsets[i] = total_size;
      if (targets[i].dense()) {
        total_size += 1;
      } else {
        // only call get_entries() once per source
        total_size += targets[i].sparsity.impl()->get_entries().size();
      }
    }

    // final end offset
    target_offsets[targets.size()] = total_size;

    Rect<N2, T2> global_bounds = targets[0].bounds;
    for (size_t i = 1; i < targets.size(); i++) {
      global_bounds.union_bbox(targets[i].bounds);
    }

    RegionInstance global_bounds_instance = this->realm_malloc(sizeof(Rect<N2,T2>), my_mem);
    Rect<N2,T2>* d_global_bounds = reinterpret_cast<Rect<N2,T2>*>(AffineAccessor<char,1>(global_bounds_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_global_bounds, &global_bounds, sizeof(Rect<N2,T2>), cudaMemcpyHostToDevice, stream),
               stream);


    RegionInstance targets_entries_instance = this->realm_malloc(total_size * sizeof(SparsityMapEntry<N2,T2>), my_mem);
    SparsityMapEntry<N2,T2>* d_targets_entries = reinterpret_cast<SparsityMapEntry<N2,T2>*>(AffineAccessor<char,1>(targets_entries_instance, 0).base);

    RegionInstance offsets_instance = this->realm_malloc((targets.size() + 1) * sizeof(size_t), my_mem);
    size_t* d_offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(offsets_instance, 0).base);

    CUDA_CHECK(
      cudaMemcpyAsync(d_offsets, target_offsets.data(), (targets.size()+1) * sizeof(size_t), cudaMemcpyHostToDevice,
        stream), stream);

    // 3) fill in place
    size_t pos = 0;
    for (size_t i = 0; i < targets.size(); ++i) {
      if (targets[i].dense()) {
        // just one rect
        SparsityMapEntry<N2, T2> entry;
        entry.bounds = targets[i].bounds;
        CUDA_CHECK(
          cudaMemcpyAsync(d_targets_entries + pos, &entry, sizeof(SparsityMapEntry<N2,T2>), cudaMemcpyHostToDevice,
            stream), stream);
        ++pos;
      } else {
        auto tmp = targets[i].sparsity.impl()->get_entries();
          CUDA_CHECK(
            cudaMemcpyAsync(d_targets_entries + pos, tmp.data(), tmp.size() * sizeof(SparsityMapEntry<N2,T2>),
              cudaMemcpyHostToDevice, stream), stream);
          pos += tmp.size();
      }
      // pos should now equal source_offsets[i+1]
    }

    AffineAccessor<Point<N2, T2>, N, T> h_accessors[domain_transform.ptr_data.size()];
    for (size_t i = 0; i < domain_transform.ptr_data.size(); ++i) {
      h_accessors[i] = AffineAccessor<Point<N2, T2>, N, T>(domain_transform.ptr_data[i].inst,
                                                           domain_transform.ptr_data[i].field_offset);
    }

    RegionInstance accessors_instance = this->realm_malloc(domain_transform.ptr_data.size() * sizeof(AffineAccessor<Point<N2,T2>,N,T>), my_mem);
    AffineAccessor<Point<N2,T2>,N,T>* d_accessors = reinterpret_cast<AffineAccessor<Point<N2,T2>,N,T>*>(AffineAccessor<char,1>(accessors_instance, 0).base);
    CUDA_CHECK(
      cudaMemcpyAsync(d_accessors, h_accessors, domain_transform.ptr_data.size() * sizeof(AffineAccessor<Point<N2,T2>,N,
        T>), cudaMemcpyHostToDevice, stream), stream);

    nvtx_range_pop();

    RegionInstance points_instance;
    PointDesc<N,T>* d_points;
    size_t num_valid_points;

    if (total_size > targets.size()) {
      nvtx_range_push("cuda", "Sparse case");
      RegionInstance morton_codes_instance = this->realm_malloc(total_size * sizeof(uint64_t), my_mem);
      uint64_t* d_morton_codes = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(morton_codes_instance, 0).base);

      RegionInstance indices_instance = this->realm_malloc(total_size * sizeof(uint64_t), my_mem);
      uint64_t* d_indices = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(indices_instance, 0).base);

      RegionInstance targets_indices_instance = this->realm_malloc(total_size * sizeof(size_t), my_mem);
      size_t* d_targets_indices = reinterpret_cast<size_t*>(AffineAccessor<char,1>(targets_indices_instance, 0).base);

      flattened_threads = 256;
      grid_size = (total_size + flattened_threads - 1) / flattened_threads;

      preimage_build_morton_codes<N2, T2><<<grid_size, flattened_threads, 0, stream>>>(
        d_targets_entries, d_offsets, d_global_bounds, total_size, targets.size(), d_morton_codes, d_indices,
        d_targets_indices);
      KERNEL_CHECK(stream);

      RegionInstance morton_codes_out_instance = this->realm_malloc(total_size * sizeof(uint64_t), my_mem);
      uint64_t* d_morton_codes_out = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(morton_codes_out_instance, 0).base);

      RegionInstance indices_out_instance = this->realm_malloc(total_size * sizeof(uint64_t), my_mem);
      uint64_t* d_indices_out = reinterpret_cast<uint64_t*>(AffineAccessor<char,1>(indices_out_instance, 0).base);

      void *bvh_temp = nullptr;
      size_t bvh_temp_bytes = 0;
      cub::DeviceRadixSort::SortPairs(bvh_temp, bvh_temp_bytes, d_morton_codes, d_morton_codes_out, d_indices,
                                      d_indices_out, total_size, 0, 64, stream);
      RegionInstance bvh_temp_instance = this->realm_malloc(bvh_temp_bytes, my_mem);
      bvh_temp = reinterpret_cast<void*>(AffineAccessor<char,1>(bvh_temp_instance, 0).base);
      cub::DeviceRadixSort::SortPairs(bvh_temp, bvh_temp_bytes, d_morton_codes, d_morton_codes_out, d_indices,
                                      d_indices_out, total_size, 0, 64, stream);

      std::swap(d_morton_codes, d_morton_codes_out);
      std::swap(d_indices, d_indices_out);

      RegionInstance childLeft_instance = this->realm_malloc((2*total_size - 1) * sizeof(int), my_mem);
      int* d_childLeft = reinterpret_cast<int*>(AffineAccessor<char,1>(childLeft_instance, 0).base);

      RegionInstance childRight_instance = this->realm_malloc((2*total_size - 1) * sizeof(int), my_mem);
      int* d_childRight = reinterpret_cast<int*>(AffineAccessor<char,1>(childRight_instance, 0).base);

      RegionInstance parent_instance = this->realm_malloc((2*total_size - 1) * sizeof(int), my_mem);
      int* d_parent = reinterpret_cast<int*>(AffineAccessor<char,1>(parent_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_parent, -1, (2*total_size - 1) * sizeof(int), stream), stream);

      flattened_threads = 256;
      grid_size = ((total_size - 1) + flattened_threads - 1) / flattened_threads;

      int n = (int) total_size;
      bvh_build_radix_tree_kernel<<< grid_size, flattened_threads, 0, stream>>>(d_morton_codes, d_indices, n, d_childLeft, d_childRight, d_parent);
      KERNEL_CHECK(stream);

      RegionInstance root_instance = this->realm_malloc(sizeof(int), my_mem);
      int* d_root = reinterpret_cast<int*>(AffineAccessor<char,1>(root_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_root, -1, sizeof(int), stream), stream);

      flattened_threads = 256;
      grid_size = (2 * total_size - 1 + flattened_threads - 1) / flattened_threads;
      bvh_build_root_kernel<<< grid_size, flattened_threads, 0, stream>>>(d_root, d_parent, total_size);
      KERNEL_CHECK(stream);

      int root;
      CUDA_CHECK(cudaMemcpyAsync(&root, d_root, sizeof(int), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);


      RegionInstance boxes_instance = this->realm_malloc((2*total_size - 1) * sizeof(Rect<N2,T2>), my_mem);
      Rect<N2,T2>* d_boxes = reinterpret_cast<Rect<N2,T2>*>(AffineAccessor<char,1>(boxes_instance, 0).base);


      flattened_threads = 256;
      grid_size = ((total_size) + flattened_threads - 1) / flattened_threads;
      preimage_init_leaf_boxes_kernel<N2, T2><<<grid_size, flattened_threads, 0, stream>>>(d_targets_entries, d_indices, total_size, d_boxes);
      KERNEL_CHECK(stream);

      RegionInstance visitCount_instance = this->realm_malloc((2*total_size - 1) * sizeof(int), my_mem);
      int* d_visitCount = reinterpret_cast<int*>(AffineAccessor<char,1>(visitCount_instance, 0).base);
      CUDA_CHECK(cudaMemsetAsync(d_visitCount, 0, (2*total_size - 1) * sizeof(int), stream), stream);

      flattened_threads = 256;
      grid_size = (total_size + flattened_threads - 1) / flattened_threads;
      bvh_merge_internal_boxes_kernel < N2, T2 ><<< grid_size, flattened_threads, 0, stream>>>(total_size, d_childLeft, d_childRight, d_parent, d_boxes, d_visitCount);
      KERNEL_CHECK(stream);

      RegionInstance target_counters_instance = this->realm_malloc(targets.size() * sizeof(uint32_t), my_mem);
      uint32_t* d_target_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(target_counters_instance, 0).base);
      CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, targets.size() * sizeof(uint32_t), stream), stream);


      int threads_per_block = 256;
      int num_blocks = (total_pts + threads_per_block - 1) / threads_per_block;
      preimage_gpuPopulateBitmasksPtrsKernel < N, T, N2, T2 ><<<num_blocks, threads_per_block, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, d_root, d_childLeft, d_childRight, d_indices,
       d_targets_indices, d_boxes, total_pts, num_valid_rects, domain_transform.ptr_data.size(), total_size, nullptr, d_target_counters, nullptr);
      KERNEL_CHECK(stream);

      uint32_t h_target_counters[targets.size()+1];
      h_target_counters[0] = 0; // prefix sum starts at 0
      CUDA_CHECK(cudaMemcpyAsync(h_target_counters+1, d_target_counters, targets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      for (size_t i = 0; i < targets.size(); ++i) {
        h_target_counters[i+1] += h_target_counters[i];
      }

      num_valid_points = h_target_counters[targets.size()];

      if (num_valid_points == 0) {
        CUDA_CHECK(cudaStreamSynchronize(stream), stream);
        for (auto it : sparsity_outputs) {
          SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
          impl->gpu_finalize();
        }
        nvtx_range_pop();
        nvtx_range_pop();
        cudaStreamDestroy(stream);
        return;
      }

      RegionInstance targets_prefix_instance = this->realm_malloc((targets.size() + 1) * sizeof(uint32_t), my_mem);
      uint32_t* d_targets_prefix = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(targets_prefix_instance, 0).base);
      CUDA_CHECK(cudaMemcpyAsync(d_targets_prefix, h_target_counters, (targets.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

      points_instance = this->realm_malloc(num_valid_points * sizeof(PointDesc<N,T>), my_mem);
      d_points = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(points_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, (targets.size()) * sizeof(uint32_t), stream), stream);

      preimage_gpuPopulateBitmasksPtrsKernel < N, T, N2, T2 ><<<num_blocks, threads_per_block, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, d_root, d_childLeft, d_childRight, d_indices,
       d_targets_indices, d_boxes, total_pts, num_valid_rects, domain_transform.ptr_data.size(), total_size, d_targets_prefix, d_target_counters, d_points);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      morton_codes_instance.destroy();
      indices_instance.destroy();
      targets_indices_instance.destroy();
      morton_codes_out_instance.destroy();
      indices_out_instance.destroy();
      bvh_temp_instance.destroy();
      childLeft_instance.destroy();
      childRight_instance.destroy();
      parent_instance.destroy();
      root_instance.destroy();
      boxes_instance.destroy();
      visitCount_instance.destroy();
      target_counters_instance.destroy();
      targets_prefix_instance.destroy();
      nvtx_range_pop();
    } else {
      nvtx_range_push("cuda", "dense case");

      RegionInstance target_counters_instance = this->realm_malloc(targets.size() * sizeof(uint32_t), my_mem);
      uint32_t* d_target_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(target_counters_instance, 0).base);
      CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, targets.size() * sizeof(uint32_t), stream), stream);

      int threads_per_block = 256;
      int num_blocks = (total_pts + threads_per_block - 1) / threads_per_block;
      preimage_dense_populate_bitmasks_kernel < N, T, N2, T2 ><<< num_blocks, threads_per_block, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, d_targets_entries, d_offsets, total_pts,
       num_valid_rects, domain_transform.ptr_data.size(), targets.size(), nullptr, d_target_counters, nullptr);
      KERNEL_CHECK(stream);

      uint32_t h_target_counters[targets.size()+1];
      h_target_counters[0] = 0; // prefix sum starts at 0
      CUDA_CHECK(cudaMemcpyAsync(h_target_counters+1, d_target_counters, targets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);
      for (size_t i = 0; i < targets.size(); ++i) {
        h_target_counters[i+1] += h_target_counters[i];
      }

      num_valid_points = h_target_counters[targets.size()];

      RegionInstance targets_prefix_instance = this->realm_malloc((targets.size() + 1) * sizeof(uint32_t), my_mem);
      uint32_t* d_targets_prefix = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(targets_prefix_instance, 0).base);
      CUDA_CHECK(cudaMemcpyAsync(d_targets_prefix, h_target_counters, (targets.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

      points_instance = this->realm_malloc(num_valid_points * sizeof(PointDesc<N,T>), my_mem);
      d_points = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(points_instance, 0).base);

      CUDA_CHECK(cudaMemsetAsync(d_target_counters, 0, (targets.size()) * sizeof(uint32_t), stream), stream);

      preimage_dense_populate_bitmasks_kernel < N, T, N2, T2 ><<< num_blocks, threads_per_block, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, d_targets_entries, d_offsets, total_pts,
       num_valid_rects, domain_transform.ptr_data.size(), targets.size(), d_targets_prefix, d_target_counters, d_points);
      KERNEL_CHECK(stream);
      CUDA_CHECK(cudaStreamSynchronize(stream), stream);

      nvtx_range_pop();
      target_counters_instance.destroy();
      targets_prefix_instance.destroy();
    }

    cudaStreamDestroy(stream);

    this->complete_pipeline(d_points, num_valid_points, my_mem,
    /* the Container: */  sparsity_outputs,
    /* getIndex: */       [&](auto const& elem){
                            // elem is a SparsityMap<N,T> from the vector
                            return size_t(&elem - sparsity_outputs.data());
                         },
    /* getMap: */         [&](auto const& elem){
                          // return the SparsityMap key itself
                          return elem;
                       });

    nvtx_range_push("cuda", "free");
    // Destroy all Realm RegionInstances (except loop‐allocated ones)
    inst_entries_instance.destroy();
    parent_entries_instance.destroy();
    inst_offsets_instance.destroy();
    valid_rects_instance.destroy();
    prefix_rects_instance.destroy();
    temp_instance.destroy();
    points_instance.destroy();
    global_bounds_instance.destroy();
    targets_entries_instance.destroy();
    offsets_instance.destroy();
    accessors_instance.destroy();
    inst_counters_instance.destroy();
    inst_prefix_instance.destroy();
    nvtx_range_pop();
    nvtx_range_pop();
  }
}
