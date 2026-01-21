#ifndef CAGRA_ADAPTER_CUH
#define CAGRA_ADAPTER_CUH

#include <raft/core/device_resources.hpp>
#include <raft/core/device_mdspan.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <iostream>

// 构建并保存 CAGRA 索引 (IdxT 默认为 uint32_t)
inline void build_cagra_index(const float* dataset, int n_rows, int dim, const char* filename) {
    raft::device_resources handle;
    // 输入数据视图
    auto dataset_view = raft::make_device_matrix_view<const float, int64_t>(dataset, n_rows, dim);
    
    cuvs::neighbors::cagra::index_params params;
    params.graph_degree = 8;               
    params.intermediate_graph_degree = 16; 
    // cuVS 默认使用 NN_DESCENT 算法，无需显式指定 

    std::cout << "  [cuVS] Building CAGRA Index..." << std::endl;
    // build 返回的 index 类型会自动推导，通常是 index<float, uint32_t>
    auto index = cuvs::neighbors::cagra::build(handle, params, dataset_view);
    
    std::cout << "  [cuVS] Serializing to " << filename << "..." << std::endl;
    cuvs::neighbors::cagra::serialize(handle, std::string(filename), index);
}

// 加载 CAGRA 索引
// 使用单 GPU 版本的 deserialize API，接受输出指针
inline cuvs::neighbors::cagra::index<float, uint32_t> load_cagra_index(raft::device_resources& handle, const char* filename) {
    cuvs::neighbors::cagra::index<float, uint32_t> index(handle);
    cuvs::neighbors::cagra::deserialize(handle, std::string(filename), &index);
    return index;
}

// 搜索 CAGRA 索引
// [修复] out_indices 改为 uint32_t*
template <typename TIndex>
inline void search_cagra_index(
    raft::device_resources& handle,
    const TIndex& index,
    const float* queries,
    int n_queries,
    int dim,
    int top_k,
    uint32_t* out_indices,   // <--- Change int* to uint32_t*
    float* out_dists
) {
    auto queries_view = raft::make_device_matrix_view<const float, int64_t>(queries, n_queries, dim);
    auto indices_view = raft::make_device_matrix_view<uint32_t, int64_t>(out_indices, n_queries, top_k);
    auto dists_view   = raft::make_device_matrix_view<float, int64_t>(out_dists, n_queries, top_k);

    cuvs::neighbors::cagra::search_params params;
    params.max_queries = n_queries;
    
    // === 优化参数 ===
    params.itopk_size = std::max(32, top_k);  // 内部候选队列
    params.search_width = 4;                   // 并行搜索宽度
    params.max_iterations = 200;               // 限制最大迭代
    
    // [新增] 使用 SINGLE_CTA 算法 - 对小数据集更高效
    params.algo = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
    
    // [新增] 优化 team_size - 每个查询使用的线程数 (0=自动)
    params.team_size = 0;

    cuvs::neighbors::cagra::search(handle, params, index, queries_view, indices_view, dists_view);
}

#endif // CAGRA_ADAPTER_CUH