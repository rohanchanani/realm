#pragma once
#include "realm/deppart/image.h"
#include "realm/deppart/image_gpu_kernels.hpp"
#include "realm/deppart/partitions_gpu_impl.hpp"
#include <cub/cub.cuh>
#include "realm/nvtx.h"

namespace Realm {

//TODO: INTERSECTING INPUT/OUTPUT RECTS CAN BE DONE WITH BVH IF BECOME EXPENSIVE

template <int N2, typename T2>
struct RectDescVolumeOp {
  __device__ __forceinline__
  size_t operator()(const RectDesc<N2,T2>& rd) const {
    return rd.rect.volume();
  }
};

template <int N2, typename T2>
struct SparsityMapEntryVolumeOp {
  __device__ __forceinline__
  size_t operator()(const SparsityMapEntry<N2,T2>& entry) const {
    return entry.bounds.volume();
  }
};

  /*
   *  Input (stored in MicroOp): Array of field instances, a parent index space, and a list of source index spaces
   *  Output: A list of (potentially overlapping) rectangles that result from chasing all the pointers in the source index spaces
   *  through the provided instances and emitting only those that intersect the parent index space labeled by which source they came from,
   *  which are then sent off to complete_rect_pipeline.
   *  Approach: Intersect all instance rectangles with source rectangles in parallel. Prefix sum + binary search to iterate over these in
   *  parallel and chase all the pointers in the source rectangles to their corresponding rectangle. Finally, intersect the output rectangles
   *  with the parent rectangles in parallel.
   */
template <int N, typename T, int N2, typename T2>
void GPUImageMicroOp<N,T,N2,T2>::gpu_populate_rngs()
{

    if (sources.size() == 0) {
      return;
    }

    NVTX_DEPPART(gpu_image);

    Memory my_mem = domain_transform.range_data[0].inst.get_location();

    cudaStream_t stream = Cuda::get_task_cuda_stream();

    collapsed_space<N2, T2> src_space;
    RegionInstance offsets_instance = this->realm_malloc((sources.size()+1) * sizeof(size_t), my_mem);
    src_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(offsets_instance, 0).base);
    src_space.num_children = sources.size();

    RegionInstance sources_entries_instance;

    GPUMicroOp<N2, T2>::collapse_multi_space(sources, sources_entries_instance, src_space, my_mem, stream);

    collapsed_space<N2, T2> inst_space;
  
    // We combine all of our instances into one to batch work, tracking the offsets between instances.
    RegionInstance inst_offsets_instance = this->realm_malloc((domain_transform.range_data.size() + 1) * sizeof(size_t), my_mem);
    inst_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(inst_offsets_instance, 0).base);
    inst_space.num_children = domain_transform.range_data.size();
  
    RegionInstance inst_entries_instance;

    GPUMicroOp<N2, T2>::collapse_multi_space(domain_transform.range_data, inst_entries_instance, inst_space, my_mem, stream);

    // This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
    // to figure out where to write each rectangle.
    RegionInstance inst_counters_instance = this->realm_malloc((2*domain_transform.range_data.size() + 1) * sizeof(uint32_t), my_mem);
    uint32_t* d_inst_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inst_counters_instance, 0).base);
  
    // This will be a prefix sum over the counters, used first to figure out where to write in the emit phase, and second
    // to track which instance each rectangle came from in the populate phase.
    uint32_t* d_inst_prefix = d_inst_counters + domain_transform.range_data.size();
    RegionInstance valid_rects_instance;
    size_t num_valid_rects;
  
    // Here we intersect the instance spaces with the parent space, and make sure we know which instance each resulting rectangle came from.
    GPUMicroOp<N2, T2>::template construct_input_rectlist<RectDesc<N2, T2>>(inst_space, src_space, valid_rects_instance, num_valid_rects, d_inst_counters, d_inst_prefix, my_mem, stream);
    inst_entries_instance.destroy();
    sources_entries_instance.destroy();
    inst_offsets_instance.destroy();

    if (num_valid_rects == 0) {
      for (SparsityMap<N, T> it : sparsity_outputs) {
            SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
            if (this->exclusive) {
              impl->gpu_finalize();
            } else {
              impl->contribute_nothing();
            }
      }
      valid_rects_instance.destroy();
      inst_counters_instance.destroy();
      return;
    }

    RectDesc<N2, T2>* d_valid_rects = reinterpret_cast<RectDesc<N2,T2>*>(AffineAccessor<char,1>(valid_rects_instance, 0).base);

    // Prefix sum the valid rectangles by volume.
    RegionInstance prefix_rects_instance;
    size_t total_pts;

    GPUMicroOp<N2, T2>::volume_prefix_sum(d_valid_rects, num_valid_rects, prefix_rects_instance, total_pts, my_mem, stream);

    size_t* d_prefix_rects = reinterpret_cast<size_t*>(AffineAccessor<char,1>(prefix_rects_instance, 0).base);

    RegionInstance rngs_instance = this->realm_malloc(total_pts * sizeof(RectDesc<N,T>), my_mem);
    RectDesc<N,T>* d_rngs = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(rngs_instance, 0).base);


    Memory zcpy_mem;
    assert(find_memory(zcpy_mem, Memory::Z_COPY_MEM));
    RegionInstance accessors_instance = this->realm_malloc(domain_transform.ptr_data.size() * sizeof(AffineAccessor<Rect<N,T>,N2,T2>), zcpy_mem);
    AffineAccessor<Rect<N,T>,N2,T2>* d_accessors = reinterpret_cast<AffineAccessor<Rect<N,T>,N2,T2>*>(AffineAccessor<char,1>(accessors_instance, 0).base);
    for (size_t i = 0; i < domain_transform.ptr_data.size(); ++i) {
      d_accessors[i] = AffineAccessor<Rect<N,T>,N2,T2>(domain_transform.ptr_data[i].inst, domain_transform.ptr_data[i].field_offset);
    }

    image_gpuPopulateBitmasksRngsKernel<N,T,N2,T2><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, d_prefix_rects, d_inst_prefix, total_pts, num_valid_rects, domain_transform.range_data.size(), d_rngs);
    KERNEL_CHECK(stream);

    RegionInstance parent_entries_instance;
    collapsed_space<N, T> collapsed_parent;

    // We collapse the parent space to undifferentiate between dense and sparse and match downstream APIs.
    GPUMicroOp<N, T>::collapse_parent_space(parent_space, parent_entries_instance, collapsed_parent, my_mem, stream);
  

    RegionInstance src_counters_instance = this->realm_malloc(sources.size() * sizeof(uint32_t), my_mem);
    uint32_t* d_src_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(src_counters_instance, 0).base);
    CUDA_CHECK(cudaMemsetAsync(d_src_counters, 0, sources.size() * sizeof(uint32_t), stream), stream);


    //Finally, we do another two pass count + emit to intersect with the parent rectangles
    image_intersect_output<N,T><<<COMPUTE_GRID(collapsed_parent.num_entries * total_pts), THREADS_PER_BLOCK, 0, stream>>>(collapsed_parent.entries_buffer, d_rngs, nullptr, collapsed_parent.num_entries, total_pts, d_src_counters, nullptr);
    KERNEL_CHECK(stream);

    std::vector<uint32_t> h_src_counters(sources.size()+1);
    h_src_counters[0] = 0; // prefix sum starts at 0
    CUDA_CHECK(cudaMemcpyAsync(h_src_counters.data()+1, d_src_counters, sources.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);

    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    valid_rects_instance.destroy();
    prefix_rects_instance.destroy();
    accessors_instance.destroy();

    for (size_t i = 0; i < sources.size(); ++i) {
      h_src_counters[i+1] += h_src_counters[i];
    }

    size_t num_valid_output = h_src_counters[sources.size()];

    if (num_valid_output == 0) {
      for (SparsityMap<N, T> it : sparsity_outputs) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
        if (this->exclusive) {
          impl->gpu_finalize();
        } else {
          impl->contribute_nothing();
        }
      }
      parent_entries_instance.destroy();
      src_counters_instance.destroy();
      rngs_instance.destroy();
      return;
    }


    RegionInstance valid_intersect_instance = this->realm_malloc(num_valid_output * sizeof(RectDesc<N,T>), my_mem);
    RectDesc<N,T>* d_valid_intersect = reinterpret_cast<RectDesc<N,T>*>(AffineAccessor<char,1>(valid_intersect_instance, 0).base);

    RegionInstance src_prefix_instance = this->realm_malloc((sources.size() + 1) * sizeof(uint32_t), my_mem);
    uint32_t* d_src_prefix = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(src_prefix_instance, 0).base);
    CUDA_CHECK(cudaMemcpyAsync(d_src_prefix, h_src_counters.data(), (sources.size() + 1) * sizeof(size_t), cudaMemcpyHostToDevice, stream), stream);

    CUDA_CHECK(cudaMemsetAsync(d_src_counters, 0, sources.size() * sizeof(uint32_t), stream), stream);

    image_intersect_output<N,T><<<COMPUTE_GRID(collapsed_parent.num_entries * total_pts), THREADS_PER_BLOCK, 0, stream>>>(collapsed_parent.entries_buffer, d_rngs, d_src_prefix, collapsed_parent.num_entries, total_pts, d_src_counters, d_valid_intersect);
    KERNEL_CHECK(stream);

    CUDA_CHECK(cudaStreamSynchronize(stream), stream);

    src_prefix_instance.destroy();
    parent_entries_instance.destroy();
    src_counters_instance.destroy();
    rngs_instance.destroy();

    this->complete_rect_pipeline(d_valid_intersect, num_valid_output, my_mem,
    /* the Container: */  sparsity_outputs,
    /* getIndex: */       [&](auto const& elem){
                            // elem is a SparsityMap<N,T> from the vector
                            return size_t(&elem - sparsity_outputs.data());
                         },
    /* getMap: */         [&](auto const& elem){
                          // return the SparsityMap key itself
                          return elem;
                       });

  valid_intersect_instance.destroy();

}

  /*
   *  Input (stored in MicroOp): Array of field instances, a parent index space, and a list of source index spaces
   *  Output: A list of (potentially overlapping) points that result from chasing all the pointers in the source index spaces
   *  through the provided instances and emitting only points in the parent index space labeled by which source they came from,
   *  which are then sent off to complete_pipeline.
   *  Approach: Intersect all instance rectangles with source rectangles in parallel. Prefix sum + binary search to iterate over these in
   *  parallel and chase all the pointers in the source rectangles to their corresponding point. Here, the pointer chasing is also a count + emit,
   *  where only points that are in the parent index space are counted/emitted.
   */
template <int N, typename T, int N2, typename T2>
void GPUImageMicroOp<N,T,N2,T2>::gpu_populate_ptrs()
{
    if (sources.size() == 0) {
      return;
    }

    NVTX_DEPPART(gpu_image);

    Memory my_mem = domain_transform.ptr_data[0].inst.get_location();

    cudaStream_t stream = Cuda::get_task_cuda_stream();

    collapsed_space<N2, T2> src_space;
    RegionInstance offsets_instance = this->realm_malloc((sources.size()+1) * sizeof(size_t), my_mem);
    src_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(offsets_instance, 0).base);
    src_space.num_children = sources.size();

    RegionInstance sources_entries_instance;

    GPUMicroOp<N2, T2>::collapse_multi_space(sources, sources_entries_instance, src_space, my_mem, stream);

    collapsed_space<N2, T2> inst_space;
  
    // We combine all of our instances into one to batch work, tracking the offsets between instances.
    RegionInstance inst_offsets_instance = this->realm_malloc((domain_transform.ptr_data.size() + 1) * sizeof(size_t), my_mem);
    inst_space.offsets = reinterpret_cast<size_t*>(AffineAccessor<char,1>(inst_offsets_instance, 0).base);
    inst_space.num_children = domain_transform.ptr_data.size();
  
    RegionInstance inst_entries_instance;

    GPUMicroOp<N2, T2>::collapse_multi_space(domain_transform.ptr_data, inst_entries_instance, inst_space, my_mem, stream);

    // This is used for count + emit: first pass counts how many rectangles survive intersection, second pass uses the counter
    // to figure out where to write each rectangle.
    RegionInstance inst_counters_instance = this->realm_malloc((2*domain_transform.ptr_data.size() + 1) * sizeof(uint32_t), my_mem);
    uint32_t* d_inst_counters = reinterpret_cast<uint32_t*>(AffineAccessor<char,1>(inst_counters_instance, 0).base);
  
    // This will be a prefix sum over the counters, used first to figure out where to write in the emit phase, and second
    // to track which instance each rectangle came from in the populate phase.
    uint32_t* d_inst_prefix = d_inst_counters + domain_transform.ptr_data.size();
    RegionInstance valid_rects_instance;
    size_t num_valid_rects;
  
    // Here we intersect the instance spaces with the parent space, and make sure we know which instance each resulting rectangle came from.
    GPUMicroOp<N2, T2>::template construct_input_rectlist<RectDesc<N2, T2>>(inst_space, src_space, valid_rects_instance, num_valid_rects, d_inst_counters, d_inst_prefix, my_mem, stream);
    inst_entries_instance.destroy();
    sources_entries_instance.destroy();
    inst_offsets_instance.destroy();

    if (num_valid_rects == 0) {
      for (SparsityMap<N, T> it : sparsity_outputs) {
            SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
            if (this->exclusive) {
              impl->gpu_finalize();
            } else {
              impl->contribute_nothing();
            }
      }
      valid_rects_instance.destroy();
      inst_counters_instance.destroy();
      return;
    }

    RectDesc<N2, T2>* d_valid_rects = reinterpret_cast<RectDesc<N2,T2>*>(AffineAccessor<char,1>(valid_rects_instance, 0).base);

    // Prefix sum the valid rectangles by volume.
    RegionInstance prefix_rects_instance;
    size_t total_pts;

    GPUMicroOp<N2, T2>::volume_prefix_sum(d_valid_rects, num_valid_rects, prefix_rects_instance, total_pts, my_mem, stream);

    size_t* d_prefix_rects = reinterpret_cast<size_t*>(AffineAccessor<char,1>(prefix_rects_instance, 0).base);

    RegionInstance parent_entries_instance;
    collapsed_space<N, T> collapsed_parent;

    // We collapse the parent space to undifferentiate between dense and sparse and match downstream APIs.
    GPUMicroOp<N, T>::collapse_parent_space(parent_space, parent_entries_instance, collapsed_parent, my_mem, stream);

    Memory zcpy_mem;
    assert(find_memory(zcpy_mem, Memory::Z_COPY_MEM));
    RegionInstance accessors_instance = this->realm_malloc(domain_transform.ptr_data.size() * sizeof(AffineAccessor<Point<N,T>,N2,T2>), zcpy_mem);
    AffineAccessor<Point<N,T>,N2,T2>* d_accessors = reinterpret_cast<AffineAccessor<Point<N,T>,N2,T2>*>(AffineAccessor<char,1>(accessors_instance, 0).base);
    for (size_t i = 0; i < domain_transform.ptr_data.size(); ++i) {
      d_accessors[i] = AffineAccessor<Point<N,T>,N2,T2>(domain_transform.ptr_data[i].inst, domain_transform.ptr_data[i].field_offset);
    }

    CUDA_CHECK(cudaMemsetAsync(d_inst_counters, 0, (domain_transform.ptr_data.size()) * sizeof(uint32_t), stream), stream);


    //We do a two pass count + emit to chase all the pointers in parallel and check for membership in the parent index space
    image_gpuPopulateBitmasksPtrsKernel<N,T,N2,T2><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, collapsed_parent.entries_buffer, d_prefix_rects, d_inst_prefix, nullptr, total_pts, num_valid_rects, domain_transform.ptr_data.size(), collapsed_parent.num_entries, d_inst_counters, nullptr);
    KERNEL_CHECK(stream);

    std::vector<uint32_t> h_inst_counters(domain_transform.ptr_data.size()+1);
    h_inst_counters[0] = 0; // prefix sum starts at 0
    CUDA_CHECK(cudaMemcpyAsync(h_inst_counters.data()+1, d_inst_counters, domain_transform.ptr_data.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream), stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    for (size_t i = 0; i < domain_transform.ptr_data.size(); ++i) {
      h_inst_counters[i+1] += h_inst_counters[i];
    }

    size_t num_valid_points = h_inst_counters[domain_transform.ptr_data.size()];

    if (num_valid_points == 0) {
      for (SparsityMap<N, T> it : sparsity_outputs) {
        SparsityMapImpl<N, T> *impl = SparsityMapImpl<N, T>::lookup(it);
        if (this->exclusive) {
          impl->gpu_finalize();
        } else {
          impl->contribute_nothing();
        }
      }
      valid_rects_instance.destroy();
      prefix_rects_instance.destroy();
      parent_entries_instance.destroy();
      accessors_instance.destroy();
      inst_counters_instance.destroy();
      return;
    }

    RegionInstance prefix_points_instance = this->realm_malloc((domain_transform.ptr_data.size() + 1) * sizeof(uint32_t), my_mem);
    uint32_t* d_prefix_points = reinterpret_cast<uint32_t *>(AffineAccessor<char,1>(prefix_points_instance, 0).base);

    CUDA_CHECK(cudaMemcpyAsync(d_prefix_points, h_inst_counters.data(), (domain_transform.ptr_data.size() + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, stream), stream);

    RegionInstance valid_points_instance = this->realm_malloc(num_valid_points * sizeof(PointDesc<N,T>), my_mem);
    PointDesc<N,T>* d_valid_points = reinterpret_cast<PointDesc<N,T>*>(AffineAccessor<char,1>(valid_points_instance, 0).base);

    CUDA_CHECK(cudaMemsetAsync(d_inst_counters, 0, (domain_transform.ptr_data.size()) * sizeof(uint32_t), stream), stream);

    image_gpuPopulateBitmasksPtrsKernel<N,T,N2,T2><<<COMPUTE_GRID(total_pts), THREADS_PER_BLOCK, 0, stream>>>(d_accessors, d_valid_rects, collapsed_parent.entries_buffer, d_prefix_rects, d_inst_prefix, d_prefix_points, total_pts, num_valid_rects, domain_transform.ptr_data.size(), collapsed_parent.num_entries, d_inst_counters, d_valid_points);
    KERNEL_CHECK(stream);
    CUDA_CHECK(cudaStreamSynchronize(stream), stream);
    valid_rects_instance.destroy();
    prefix_rects_instance.destroy();
    parent_entries_instance.destroy();
    accessors_instance.destroy();
    inst_counters_instance.destroy();
    prefix_points_instance.destroy();

    //Send it off for processing
    this->complete_pipeline(d_valid_points, num_valid_points, my_mem,
    /* the Container: */  sparsity_outputs,
    /* getIndex: */       [&](auto const& elem){
                            // elem is a SparsityMap<N,T> from the vector
                            return size_t(&elem - sparsity_outputs.data());
                         },
    /* getMap: */         [&](auto const& elem){
                          // return the SparsityMap key itself
                          return elem;
                       });

    valid_points_instance.destroy();
}
}