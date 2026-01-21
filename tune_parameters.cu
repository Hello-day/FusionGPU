#include "defs.h"
#include "utils.h"
#include "cagra_adapter.cuh"
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>
#include <random>
#include <numeric>
#include <set>

// 参数调优配置
struct TuneConfig {
    int n_samples = 1000;           // 采样查询数量
    int final_k = 10;               // 最终返回的 top-k
    float p98_percentile = 0.98f;   // P98 分位数
    float p99_percentile = 0.99f;   // P99 分位数
};

// 调优结果
struct TuneResults {
    int recommended_top_m;          // 搜索的最大聚类数 (P98/P99 分位数方法)
    int recommended_p1_lists;       // Phase 1 应该搜索的列表数量 (拐点检测方法)
    int recommended_limit_k;        // Phase 1 列表内截断保留的元素数量
    float threshold_coeff;          // Phase 2 过滤阈值放宽系数
    
    // 统计信息
    std::vector<int> cluster_ranks;      // 真实结果所在聚类的排名
    std::vector<int> pq_ranks;           // 真实结果在聚类内 PQ 排序的排名
    std::vector<int> exact_ranks;        // 真实结果在聚类内精确距离排序的排名
    std::vector<float> pq_ratios;        // PQ 距离比率
    std::vector<int> rank_diffs;         // PQ排名与精确排名的差异
    std::vector<int> hits_per_list;      // 每个列表位置的命中数 (用于拐点分析)
};

// 计算欧氏距离平方
__device__ __host__ inline float euclidean_distance_sq(const float* a, const float* b, int dim) {
    float dist = 0.0f;
    for (int i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// GPU Kernel: 暴力搜索计算 Ground Truth
__global__ void brute_force_search_kernel(
    const float* queries,      // [n_queries, DIM]
    const float* database,     // [n_base, DIM]
    int* gt_ids,              // [n_queries, k]
    float* gt_dists,          // [n_queries, k]
    int n_queries,
    int n_base,
    int k,
    int dim
) {
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (qid >= n_queries) return;
    
    const float* query = queries + qid * dim;
    
    // 使用局部数组存储 top-k
    float local_dists[100];  // 假设 k <= 100
    int local_ids[100];
    
    for (int i = 0; i < k; ++i) {
        local_dists[i] = 1e30f;
        local_ids[i] = -1;
    }
    
    // 遍历所有数据库向量
    for (int i = 0; i < n_base; ++i) {
        const float* base_vec = database + i * dim;
        float dist = euclidean_distance_sq(query, base_vec, dim);
        
        // 插入排序维护 top-k
        if (dist < local_dists[k-1]) {
            int pos = k - 1;
            while (pos > 0 && dist < local_dists[pos-1]) {
                local_dists[pos] = local_dists[pos-1];
                local_ids[pos] = local_ids[pos-1];
                pos--;
            }
            local_dists[pos] = dist;
            local_ids[pos] = i;
        }
    }
    
    // 写回全局内存
    for (int i = 0; i < k; ++i) {
        gt_ids[qid * k + i] = local_ids[i];
        gt_dists[qid * k + i] = local_dists[i];
    }
}

// GPU Kernel: 预计算残差距离查找表 (LUT)
// 与 gpu_global.cu 中的 batch_residual_precompute_kernel 逻辑一致
// LUT[m][k] = ||residual_m - codebook[m][k]||²
__global__ void precompute_lut_kernel(
    const float* query,             // [DIM]
    const float* centroid,          // [DIM]
    const float* pq_codebook,       // [PQ_M, PQ_K, PQ_SUB_DIM]
    float* lut                      // [PQ_M, PQ_K]
) {
    int m = blockIdx.x;   // 子空间索引
    int k = threadIdx.x;  // 码字索引
    
    if (m >= PQ_M || k >= PQ_K) return;
    
    int dim_offset = m * PQ_SUB_DIM;
    const float* cb_vec = pq_codebook + (m * PQ_K + k) * PQ_SUB_DIM;
    
    float dist = 0.0f;
    for (int d = 0; d < PQ_SUB_DIM; ++d) {
        float residual = query[dim_offset + d] - centroid[dim_offset + d];
        float diff = residual - cb_vec[d];
        dist += diff * diff;
    }
    
    lut[m * PQ_K + k] = dist;
}

// GPU Kernel: 使用 LUT 查表计算 PQ 距离
// 与 gpu_global.cu 中的 compute_pq_dist_vectorized 逻辑一致
__global__ void compute_pq_dist_with_lut_kernel(
    const uint8_t* pq_codes,        // [n_vecs, PQ_M]
    const float* lut,               // [PQ_M, PQ_K]
    float* pq_dists,                // [n_vecs]
    int n_vecs
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= n_vecs) return;
    
    const uint8_t* code = pq_codes + vid * PQ_M;
    
    float dist = 0.0f;
    #pragma unroll 8
    for (int m = 0; m < PQ_M; ++m) {
        dist += lut[m * PQ_K + code[m]];
    }
    
    pq_dists[vid] = dist;
}

// GPU Kernel: 计算精确欧氏距离 (用于聚类内精确排名)
__global__ void compute_exact_distances_kernel(
    const float* query,             // [DIM]
    const float* base_vecs,         // [n_vecs, DIM]
    float* exact_dists,             // [n_vecs]
    int n_vecs
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= n_vecs) return;
    
    const float* vec = base_vecs + vid * DIM;
    
    float dist = 0.0f;
    for (int d = 0; d < DIM; ++d) {
        float diff = query[d] - vec[d];
        dist += diff * diff;
    }
    
    exact_dists[vid] = dist;
}

// 计算分位数
float compute_percentile(std::vector<float>& data, float percentile) {
    if (data.empty()) return 0.0f;
    std::sort(data.begin(), data.end());
    int idx = static_cast<int>(data.size() * percentile);
    if (idx >= data.size()) idx = data.size() - 1;
    return data[idx];
}

int compute_percentile_int(std::vector<int>& data, float percentile) {
    if (data.empty()) return 0;
    std::sort(data.begin(), data.end());
    int idx = static_cast<int>(data.size() * percentile);
    if (idx >= data.size()) idx = data.size() - 1;
    return data[idx];
}

// ===== 拐点检测算法 =====
// 分析 k 近邻在各列表位置的分布，找到密集区域的拐点
struct ElbowAnalysisResult {
    int elbow_point;                    // 拐点位置
    float coverage_at_elbow;            // 拐点处的覆盖率
    std::vector<float> cumulative_coverage;  // 累积覆盖率
    std::vector<float> marginal_gain;        // 边际增益 (一阶导数)
};

// 核心思想：找到累积覆盖率曲线的"膝盖点"
// 即增长速度明显放缓的位置
ElbowAnalysisResult find_elbow_point(const std::vector<int>& hits_per_list, int total_hits) {
    ElbowAnalysisResult result;
    int n = hits_per_list.size();
    
    if (n == 0 || total_hits == 0) {
        result.elbow_point = 1;
        result.coverage_at_elbow = 0.0f;
        return result;
    }
    
    // 计算累积覆盖率和边际增益
    result.cumulative_coverage.resize(n);
    result.marginal_gain.resize(n);
    
    int cumsum = 0;
    for (int i = 0; i < n; ++i) {
        cumsum += hits_per_list[i];
        result.cumulative_coverage[i] = static_cast<float>(cumsum) / total_hits;
        result.marginal_gain[i] = static_cast<float>(hits_per_list[i]) / total_hits;
    }
    
    // ===== 方法: 基于边际增益衰减的拐点检测 =====
    // 思路：前几个列表的边际增益很高，找到增益显著下降的点
    
    // 计算前几个列表的平均增益作为基准
    int baseline_count = std::min(5, n);
    float baseline_gain = 0.0f;
    for (int i = 0; i < baseline_count; ++i) {
        baseline_gain += result.marginal_gain[i];
    }
    baseline_gain /= baseline_count;
    
    // 找到边际增益下降到基准的某个比例的位置
    // 同时使用滑动窗口平滑噪声
    int window_size = 3;
    result.elbow_point = n;
    
    for (int i = baseline_count; i < n - window_size + 1; ++i) {
        // 计算窗口内的平均增益
        float window_gain = 0.0f;
        for (int j = 0; j < window_size; ++j) {
            window_gain += result.marginal_gain[i + j];
        }
        window_gain /= window_size;
        
        // 当窗口增益下降到基准的 10% 以下时，认为是拐点
        if (window_gain < baseline_gain * 0.1f) {
            result.elbow_point = i;
            break;
        }
        
        // 或者当累积覆盖率已经很高且增益很低时
        if (result.cumulative_coverage[i] > 0.60f && window_gain < baseline_gain * 0.2f) {
            result.elbow_point = i;
            break;
        }
    }
    
    // 确保拐点至少覆盖 60% 的命中
    while (result.elbow_point > 1 && 
           result.cumulative_coverage[result.elbow_point - 1] < 0.60f) {
        result.elbow_point++;
    }
    
    // 限制最大值，不超过 P98 的合理范围
    result.elbow_point = std::min(result.elbow_point, n);
    
    if (result.elbow_point > 0 && result.elbow_point <= n) {
        result.coverage_at_elbow = result.cumulative_coverage[result.elbow_point - 1];
    } else {
        result.elbow_point = 1;
        result.coverage_at_elbow = result.cumulative_coverage[0];
    }
    
    return result;
}

// 主调优函数
TuneResults tune_search_parameters(
    const float* base_data,        // 原始数据集 [n_base, DIM]
    int n_base,
    const float* centroids,        // IVF 聚类中心 [n_clusters, DIM]
    int n_clusters,
    const int* cluster_offsets,    // 聚类偏移 [n_clusters + 1]
    const int* vector_ids,         // 向量 ID [total_vecs]
    const uint8_t* pq_codes,       // PQ 编码 [total_vecs, PQ_M]
    const float* pq_codebook,      // PQ 码本 [PQ_M, PQ_K, PQ_SUB_DIM]
    const TuneConfig& config
) {
    TuneResults results;
    
    std::cout << "\n========== 参数调优开始 ==========" << std::endl;
    std::cout << "采样查询数量: " << config.n_samples << std::endl;
    std::cout << "目标 Top-K: " << config.final_k << std::endl;
    
    // ===== 步骤 1: 采样查询向量 =====
    std::cout << "\n[1/4] 采样查询向量..." << std::endl;
    std::vector<int> sample_indices(config.n_samples);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n_base - 1);
    
    for (int i = 0; i < config.n_samples; ++i) {
        sample_indices[i] = dis(gen);
    }
    
    std::vector<float> sample_queries(config.n_samples * DIM);
    for (int i = 0; i < config.n_samples; ++i) {
        std::memcpy(sample_queries.data() + i * DIM, 
                   base_data + sample_indices[i] * DIM, 
                   DIM * sizeof(float));
    }
    
    // ===== 步骤 2: 计算 Ground Truth =====
    std::cout << "[2/4] 计算 Ground Truth (GPU 暴力搜索)..." << std::endl;
    
    float *d_queries, *d_base, *d_gt_dists;
    int *d_gt_ids;
    
    cudaMalloc(&d_queries, config.n_samples * DIM * sizeof(float));
    cudaMalloc(&d_base, n_base * DIM * sizeof(float));
    cudaMalloc(&d_gt_ids, config.n_samples * config.final_k * sizeof(int));
    cudaMalloc(&d_gt_dists, config.n_samples * config.final_k * sizeof(float));
    
    cudaMemcpy(d_queries, sample_queries.data(), 
               config.n_samples * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_base, base_data, 
               n_base * DIM * sizeof(float), cudaMemcpyHostToDevice);
    
    int threads = 256;
    int blocks = (config.n_samples + threads - 1) / threads;
    brute_force_search_kernel<<<blocks, threads>>>(
        d_queries, d_base, d_gt_ids, d_gt_dists,
        config.n_samples, n_base, config.final_k, DIM
    );
    cudaDeviceSynchronize();
    
    std::vector<int> gt_ids(config.n_samples * config.final_k);
    std::vector<float> gt_dists(config.n_samples * config.final_k);
    cudaMemcpy(gt_ids.data(), d_gt_ids, 
               config.n_samples * config.final_k * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(gt_dists.data(), d_gt_dists, 
               config.n_samples * config.final_k * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "Ground Truth 计算完成" << std::endl;

    // ===== 步骤 3: 分析聚类排名 (RECOMMENDED_P1_LISTS) =====
    std::cout << "\n[3/4] 分析聚类排名..." << std::endl;
    
    // 为每个查询的每个 GT 结果记录聚类排名
    std::vector<int> all_cluster_ranks;
    all_cluster_ranks.reserve(config.n_samples * config.final_k);
    
    for (int i = 0; i < config.n_samples; ++i) {
        const float* query = sample_queries.data() + i * DIM;
        
        // 计算查询向量到所有聚类中心的距离（只需计算一次）
        std::vector<std::pair<float, int>> cluster_dists(n_clusters);
        for (int c = 0; c < n_clusters; ++c) {
            const float* centroid = centroids + c * DIM;
            float dist = euclidean_distance_sq(query, centroid, DIM);
            cluster_dists[c] = {dist, c};
        }
        
        // 排序聚类中心
        std::sort(cluster_dists.begin(), cluster_dists.end());
        
        // 为该查询的所有 K 个真实近邻找到其聚类排名
        for (int k = 0; k < config.final_k; ++k) {
            int true_id = gt_ids[i * config.final_k + k];
            
            // 找到 true_id 所属的聚类
            int true_cluster = -1;
            for (int c = 0; c < n_clusters; ++c) {
                int start = cluster_offsets[c];
                int end = cluster_offsets[c + 1];
                for (int j = start; j < end; ++j) {
                    if (vector_ids[j] == true_id) {
                        true_cluster = c;
                        break;
                    }
                }
                if (true_cluster != -1) break;
            }
            
            if (true_cluster == -1) {
                std::cerr << "警告: 无法找到 GT ID " << true_id << " 所属的聚类" << std::endl;
                continue;
            }
            
            // 找到该聚类的排名
            int rank = -1;
            for (int r = 0; r < n_clusters; ++r) {
                if (cluster_dists[r].second == true_cluster) {
                    rank = r + 1;  // 排名从 1 开始
                    break;
                }
            }
            
            if (rank > 0) {
                all_cluster_ranks.push_back(rank);
            }
        }
    }
    
    // 保存用于统计
    results.cluster_ranks = all_cluster_ranks;
    
    // ===== 方法1: P98 分位数确定 RECOMMENDED_TOP_M =====
    results.recommended_top_m = compute_percentile_int(all_cluster_ranks, config.p98_percentile);
    
    // ===== 方法2: 拐点检测确定 RECOMMENDED_P1_LISTS =====
    // 统计每个列表位置的命中数
    int max_rank = *std::max_element(all_cluster_ranks.begin(), all_cluster_ranks.end());
    std::vector<int> hits_per_list(max_rank, 0);
    
    for (int rank : all_cluster_ranks) {
        if (rank > 0 && rank <= max_rank) {
            hits_per_list[rank - 1]++;  // rank 从 1 开始，数组从 0 开始
        }
    }
    
    // 保存用于统计输出
    results.hits_per_list = hits_per_list;
    
    // 执行拐点检测
    int total_hits = all_cluster_ranks.size();
    ElbowAnalysisResult elbow_result = find_elbow_point(hits_per_list, total_hits);
    results.recommended_p1_lists = elbow_result.elbow_point;
    
    std::cout << "聚类排名分析完成" << std::endl;
    std::cout << "  - 总样本数 (queries × k): " << all_cluster_ranks.size() << std::endl;
    std::cout << "  - 平均排名: " << std::accumulate(all_cluster_ranks.begin(), all_cluster_ranks.end(), 0.0) / all_cluster_ranks.size() << std::endl;
    std::cout << "  - 最大排名: " << max_rank << std::endl;
    std::cout << "\n  [方法1] P98 分位数 (RECOMMENDED_TOP_M): " << results.recommended_top_m << std::endl;
    std::cout << "  [方法2] 拐点检测 (RECOMMENDED_P1_LISTS): " << results.recommended_p1_lists << std::endl;
    std::cout << "          拐点处覆盖率: " << std::fixed << std::setprecision(2) 
              << elbow_result.coverage_at_elbow * 100 << "%" << std::endl;
    
    // 打印前20个列表的命中分布
    std::cout << "\n  列表命中分布 (前20):" << std::endl;
    int cumsum = 0;
    for (int i = 0; i < std::min(20, (int)hits_per_list.size()); ++i) {
        cumsum += hits_per_list[i];
        float coverage = static_cast<float>(cumsum) / total_hits * 100;
        std::cout << "    List " << std::setw(2) << (i + 1) << ": " 
                  << std::setw(5) << hits_per_list[i] << " hits, "
                  << "累积覆盖: " << std::fixed << std::setprecision(1) << coverage << "%" << std::endl;
    }
    
    // ===== 步骤 4: 分析 PQ 排序误差 (基于 P1_LISTS 内的 k 近邻) =====
    std::cout << "\n[4/4] 分析 PQ 排序误差 (仅统计前 " << results.recommended_p1_lists << " 个聚类)..." << std::endl;
    
    // 将 PQ 相关数据传输到 GPU
    float *d_centroids, *d_pq_codebook;
    
    cudaMalloc(&d_centroids, n_clusters * DIM * sizeof(float));
    cudaMalloc(&d_pq_codebook, PQ_M * PQ_K * PQ_SUB_DIM * sizeof(float));
    
    cudaMemcpy(d_centroids, centroids, n_clusters * DIM * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pq_codebook, pq_codebook, PQ_M * PQ_K * PQ_SUB_DIM * sizeof(float), cudaMemcpyHostToDevice);
    
    // 预分配 LUT 空间 (每个查询复用)
    float *d_lut;
    cudaMalloc(&d_lut, PQ_M * PQ_K * sizeof(float));
    
    // 收集落在前 P1_LISTS 个聚类中的 k 近邻的 PQ 排名
    std::vector<int> p1_pq_ranks;           // 前 P1_LISTS 聚类内的 PQ 排名
    std::vector<int> all_pq_ranks;          // 所有 k 近邻的 PQ 排名 (用于对比)
    std::vector<int> all_exact_ranks;
    std::vector<int> all_rank_diffs;
    
    for (int i = 0; i < config.n_samples; ++i) {
        const float* query = sample_queries.data() + i * DIM;
        
        // 计算查询向量到所有聚类中心的距离并排序
        std::vector<std::pair<float, int>> cluster_dists(n_clusters);
        for (int c = 0; c < n_clusters; ++c) {
            const float* centroid = centroids + c * DIM;
            float dist = euclidean_distance_sq(query, centroid, DIM);
            cluster_dists[c] = {dist, c};
        }
        std::sort(cluster_dists.begin(), cluster_dists.end());
        
        // 获取前 P1_LISTS 个聚类的 ID
        std::set<int> p1_clusters;
        for (int r = 0; r < std::min(results.recommended_p1_lists, n_clusters); ++r) {
            p1_clusters.insert(cluster_dists[r].second);
        }
        
        // 处理该查询的所有 K 个真实近邻
        for (int k = 0; k < config.final_k; ++k) {
            int true_id = gt_ids[i * config.final_k + k];
            
            // 找到 true_id 所属的聚类
            int true_cluster = -1;
            for (int c = 0; c < n_clusters; ++c) {
                int start = cluster_offsets[c];
                int end = cluster_offsets[c + 1];
                for (int j = start; j < end; ++j) {
                    if (vector_ids[j] == true_id) {
                        true_cluster = c;
                        break;
                    }
                }
                if (true_cluster != -1) break;
            }
            
            if (true_cluster == -1) continue;
            
            int cluster_start = cluster_offsets[true_cluster];
            int cluster_end = cluster_offsets[true_cluster + 1];
            int cluster_size = cluster_end - cluster_start;
            
            if (cluster_size == 0) continue;
            
            // ===== 使用 LUT 方式计算 PQ 距离 (与 gpu_global.cu 一致) =====
            std::vector<float> pq_dists_host(cluster_size);
            
            float *d_query_single, *d_pq_dists, *d_centroid_single;
            uint8_t *d_cluster_codes;
            
            cudaMalloc(&d_query_single, DIM * sizeof(float));
            cudaMalloc(&d_centroid_single, DIM * sizeof(float));
            cudaMalloc(&d_pq_dists, cluster_size * sizeof(float));
            cudaMalloc(&d_cluster_codes, cluster_size * PQ_M * sizeof(uint8_t));
            
            cudaMemcpy(d_query_single, query, DIM * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_centroid_single, centroids + true_cluster * DIM, DIM * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_cluster_codes, pq_codes + cluster_start * PQ_M, 
                       cluster_size * PQ_M * sizeof(uint8_t), cudaMemcpyHostToDevice);
            
            // Step 1: 预计算 LUT (grid: PQ_M blocks, block: PQ_K threads)
            precompute_lut_kernel<<<PQ_M, PQ_K>>>(
                d_query_single, d_centroid_single, d_pq_codebook, d_lut
            );
            
            // Step 2: 使用 LUT 查表计算 PQ 距离
            int threads_pq = 256;
            int blocks_pq = (cluster_size + threads_pq - 1) / threads_pq;
            compute_pq_dist_with_lut_kernel<<<blocks_pq, threads_pq>>>(
                d_cluster_codes, d_lut, d_pq_dists, cluster_size
            );
            cudaDeviceSynchronize();
            
            cudaMemcpy(pq_dists_host.data(), d_pq_dists, 
                       cluster_size * sizeof(float), cudaMemcpyDeviceToHost);
            
            // ===== 计算精确欧氏距离 =====
            std::vector<float> exact_dists_host(cluster_size);
            float *d_exact_dists, *d_cluster_vecs;
            
            cudaMalloc(&d_exact_dists, cluster_size * sizeof(float));
            cudaMalloc(&d_cluster_vecs, cluster_size * DIM * sizeof(float));
            
            // 收集聚类内的原始向量
            std::vector<float> cluster_vecs(cluster_size * DIM);
            for (int j = 0; j < cluster_size; ++j) {
                int vec_id = vector_ids[cluster_start + j];
                std::memcpy(cluster_vecs.data() + j * DIM, 
                           base_data + vec_id * DIM, DIM * sizeof(float));
            }
            cudaMemcpy(d_cluster_vecs, cluster_vecs.data(), 
                       cluster_size * DIM * sizeof(float), cudaMemcpyHostToDevice);
            
            compute_exact_distances_kernel<<<blocks_pq, threads_pq>>>(
                d_query_single, d_cluster_vecs, d_exact_dists, cluster_size
            );
            cudaDeviceSynchronize();
            
            cudaMemcpy(exact_dists_host.data(), d_exact_dists, 
                       cluster_size * sizeof(float), cudaMemcpyDeviceToHost);
            
            cudaFree(d_query_single);
            cudaFree(d_centroid_single);
            cudaFree(d_pq_dists);
            cudaFree(d_cluster_codes);
            cudaFree(d_exact_dists);
            cudaFree(d_cluster_vecs);
            
            // 找到 true_id 在该聚类中的局部索引
            int true_local_idx = -1;
            for (int j = 0; j < cluster_size; ++j) {
                if (vector_ids[cluster_start + j] == true_id) {
                    true_local_idx = j;
                    break;
                }
            }
            
            if (true_local_idx == -1) continue;
            
            float true_pq_dist = pq_dists_host[true_local_idx];
            float true_exact_dist = exact_dists_host[true_local_idx];
            
            // 计算 PQ 排名 (基于残差 PQ 距离)
            int pq_rank = 1;
            for (int j = 0; j < cluster_size; ++j) {
                if (pq_dists_host[j] < true_pq_dist) {
                    pq_rank++;
                }
            }
            
            // 计算精确排名 (基于精确欧氏距离)
            int exact_rank = 1;
            for (int j = 0; j < cluster_size; ++j) {
                if (exact_dists_host[j] < true_exact_dist) {
                    exact_rank++;
                }
            }
            
            all_pq_ranks.push_back(pq_rank);
            all_exact_ranks.push_back(exact_rank);
            all_rank_diffs.push_back(pq_rank - exact_rank);
            
            // 如果该聚类在前 P1_LISTS 中，记录 PQ 排名 (用于计算 LIMIT_K)
            if (p1_clusters.count(true_cluster) > 0) {
                p1_pq_ranks.push_back(pq_rank);
            }
        }
    }
    
    // 保存用于统计
    results.pq_ranks = all_pq_ranks;
    results.exact_ranks = all_exact_ranks;
    results.rank_diffs = all_rank_diffs;
    
    // ===== 计算 RECOMMENDED_LIMIT_K (前 P1_LISTS 聚类内 PQ 排名的 100% 置信度) =====
    if (!p1_pq_ranks.empty()) {
        results.recommended_limit_k = *std::max_element(p1_pq_ranks.begin(), p1_pq_ranks.end());
    } else {
        // 回退到 P98 分位数
        results.recommended_limit_k = compute_percentile_int(all_pq_ranks, config.p98_percentile);
    }
    
    std::cout << "PQ 排名分析完成" << std::endl;
    std::cout << "  - 总样本数 (queries × k): " << all_pq_ranks.size() << std::endl;
    std::cout << "  - 前 P1_LISTS 聚类内样本数: " << p1_pq_ranks.size() << std::endl;
    std::cout << "  - 平均 PQ 排名 (全部): " << std::accumulate(all_pq_ranks.begin(), all_pq_ranks.end(), 0.0) / all_pq_ranks.size() << std::endl;
    std::cout << "  - 平均精确排名: " << std::accumulate(all_exact_ranks.begin(), all_exact_ranks.end(), 0.0) / all_exact_ranks.size() << std::endl;
    std::cout << "  - 平均排名差异 (PQ - 精确): " << std::accumulate(all_rank_diffs.begin(), all_rank_diffs.end(), 0.0) / all_rank_diffs.size() << std::endl;
    if (!p1_pq_ranks.empty()) {
        std::cout << "  - 平均 PQ 排名 (P1_LISTS 内): " << std::accumulate(p1_pq_ranks.begin(), p1_pq_ranks.end(), 0.0) / p1_pq_ranks.size() << std::endl;
        std::cout << "  - 最大 PQ 排名 (P1_LISTS 内, 100%置信度): " << results.recommended_limit_k << std::endl;
    }
    std::cout << "  - RECOMMENDED_LIMIT_K: " << results.recommended_limit_k << std::endl;
    
    // ===== 计算 THRESHOLD_COEFF =====
    // 逻辑：对于每个查询的每个 k 近邻，计算其 PQ 距离与其所在聚类第 LIMIT_K 名 PQ 距离的比值
    // 只统计落在前 P1_LISTS 聚类中的 k 近邻
    std::cout << "\n计算阈值系数 (基于 P1_LISTS=" << results.recommended_p1_lists 
              << ", LIMIT_K=" << results.recommended_limit_k << ")..." << std::endl;
    
    std::vector<float> all_ratios;
    
    for (int i = 0; i < config.n_samples; ++i) {
        const float* query = sample_queries.data() + i * DIM;
        
        // 计算查询向量到所有聚类中心的距离并排序
        std::vector<std::pair<float, int>> cluster_dists(n_clusters);
        for (int c = 0; c < n_clusters; ++c) {
            const float* centroid = centroids + c * DIM;
            float dist = euclidean_distance_sq(query, centroid, DIM);
            cluster_dists[c] = {dist, c};
        }
        std::sort(cluster_dists.begin(), cluster_dists.end());
        
        // 获取前 P1_LISTS 个聚类的 ID
        std::set<int> p1_clusters;
        for (int r = 0; r < std::min(results.recommended_p1_lists, n_clusters); ++r) {
            p1_clusters.insert(cluster_dists[r].second);
        }
        
        // 处理该查询的所有 K 个真实近邻
        for (int k = 0; k < config.final_k; ++k) {
            int true_id = gt_ids[i * config.final_k + k];
            
            // 找到 true_id 所属的聚类
            int true_cluster = -1;
            for (int c = 0; c < n_clusters; ++c) {
                int start = cluster_offsets[c];
                int end = cluster_offsets[c + 1];
                for (int j = start; j < end; ++j) {
                    if (vector_ids[j] == true_id) {
                        true_cluster = c;
                        break;
                    }
                }
                if (true_cluster != -1) break;
            }
            
            if (true_cluster == -1) continue;
            
            // 只统计落在前 P1_LISTS 聚类中的 k 近邻
            if (p1_clusters.count(true_cluster) == 0) continue;
            
            int cluster_start = cluster_offsets[true_cluster];
            int cluster_end = cluster_offsets[true_cluster + 1];
            int cluster_size = cluster_end - cluster_start;
            
            if (cluster_size == 0) continue;
            
            // 计算该聚类内所有向量的 PQ 距离
            std::vector<float> pq_dists_host(cluster_size);
            
            float *d_query_single, *d_pq_dists, *d_centroid_single;
            uint8_t *d_cluster_codes;
            
            cudaMalloc(&d_query_single, DIM * sizeof(float));
            cudaMalloc(&d_centroid_single, DIM * sizeof(float));
            cudaMalloc(&d_pq_dists, cluster_size * sizeof(float));
            cudaMalloc(&d_cluster_codes, cluster_size * PQ_M * sizeof(uint8_t));
            
            cudaMemcpy(d_query_single, query, DIM * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_centroid_single, centroids + true_cluster * DIM, DIM * sizeof(float), cudaMemcpyHostToDevice);
            cudaMemcpy(d_cluster_codes, pq_codes + cluster_start * PQ_M, 
                       cluster_size * PQ_M * sizeof(uint8_t), cudaMemcpyHostToDevice);
            
            precompute_lut_kernel<<<PQ_M, PQ_K>>>(
                d_query_single, d_centroid_single, d_pq_codebook, d_lut
            );
            
            int threads_pq = 256;
            int blocks_pq = (cluster_size + threads_pq - 1) / threads_pq;
            compute_pq_dist_with_lut_kernel<<<blocks_pq, threads_pq>>>(
                d_cluster_codes, d_lut, d_pq_dists, cluster_size
            );
            cudaDeviceSynchronize();
            
            cudaMemcpy(pq_dists_host.data(), d_pq_dists, 
                       cluster_size * sizeof(float), cudaMemcpyDeviceToHost);
            
            cudaFree(d_query_single);
            cudaFree(d_centroid_single);
            cudaFree(d_pq_dists);
            cudaFree(d_cluster_codes);
            
            // 找到 true_id 的 PQ 距离
            float true_pq_dist = -1.0f;
            for (int j = 0; j < cluster_size; ++j) {
                if (vector_ids[cluster_start + j] == true_id) {
                    true_pq_dist = pq_dists_host[j];
                    break;
                }
            }
            
            if (true_pq_dist < 0) continue;
            
            // 排序该聚类内的 PQ 距离，获取第 LIMIT_K 名的距离
            std::vector<float> sorted_pq_dists = pq_dists_host;
            std::sort(sorted_pq_dists.begin(), sorted_pq_dists.end());
            
            int limit_k_idx = std::min(results.recommended_limit_k - 1, cluster_size - 1);
            if (limit_k_idx < 0) limit_k_idx = 0;
            float limit_k_pq_dist = sorted_pq_dists[limit_k_idx];
            
            // 计算比值: k近邻的 PQ 距离 / 该聚类第 LIMIT_K 名的 PQ 距离
            if (limit_k_pq_dist > 0) {
                float ratio = true_pq_dist / limit_k_pq_dist;
                all_ratios.push_back(ratio);
            }
        }
    }
    
    // 保存用于统计
    results.pq_ratios = all_ratios;
    
    // 计算 THRESHOLD_COEFF (取最大值，确保 100% 覆盖)
    if (!all_ratios.empty()) {
        // 使用 P99 或最大值
        float max_ratio = *std::max_element(all_ratios.begin(), all_ratios.end());
        float p99_ratio = compute_percentile(all_ratios, config.p99_percentile);
        results.threshold_coeff = std::max(max_ratio, p99_ratio);  // 取最大值确保覆盖
    } else {
        results.threshold_coeff = 2.0f;  // 默认值
    }
    
    std::cout << "阈值系数分析完成" << std::endl;
    std::cout << "  - 有效样本数 (P1_LISTS 内的 k 近邻): " << all_ratios.size() << std::endl;
    if (!all_ratios.empty()) {
        std::cout << "  - 平均比值 (kNN_PQ / LIMIT_K_PQ): " 
                  << std::accumulate(all_ratios.begin(), all_ratios.end(), 0.0f) / all_ratios.size() << std::endl;
        std::cout << "  - 最大比值: " << *std::max_element(all_ratios.begin(), all_ratios.end()) << std::endl;
        std::cout << "  - P99 比值: " << compute_percentile(all_ratios, config.p99_percentile) << std::endl;
        std::cout << "  - THRESHOLD_COEFF (100%覆盖): " << results.threshold_coeff << std::endl;
    }
    
    // 清理 GPU 内存
    cudaFree(d_queries);
    cudaFree(d_base);
    cudaFree(d_gt_ids);
    cudaFree(d_gt_dists);
    cudaFree(d_centroids);
    cudaFree(d_pq_codebook);
    cudaFree(d_lut);
    
    std::cout << "\n========== 参数调优完成 ==========" << std::endl;
    
    return results;
}

// 保存调优结果到配置文件
void save_tune_results(const TuneResults& results, const std::string& output_file) {
    std::ofstream ofs(output_file);
    if (!ofs) {
        std::cerr << "无法创建配置文件: " << output_file << std::endl;
        return;
    }
    
    ofs << "# 搜索参数调优结果\n";
    ofs << "# 该文件由 tune_parameters 工具自动生成\n\n";
    
    ofs << "# ===== 聚类搜索范围参数 =====\n\n";
    
    ofs << "# TOP_M: 搜索的最大聚类数 (P98 分位数方法)\n";
    ofs << "# 含义: 保证 98% 的查询，其真实最近邻所在的聚类被包含在搜索范围内\n";
    ofs << "RECOMMENDED_TOP_M=" << results.recommended_top_m << "\n\n";
    
    ofs << "# P1_LISTS: Phase 1 密集搜索的聚类数量 (拐点检测方法)\n";
    ofs << "# 含义: k近邻分布的密集区域，在此范围内命中率高，超过此范围边际收益递减\n";
    ofs << "RECOMMENDED_P1_LISTS=" << results.recommended_p1_lists << "\n\n";
    
    ofs << "# ===== PQ 距离相关参数 =====\n\n";
    
    ofs << "# LIMIT_K: 列表内截断保留的元素数量\n";
    ofs << "# 含义: 由于 PQ 距离存在误差，需要保留足够多的候选者以防止真实结果被过早截断\n";
    ofs << "RECOMMENDED_LIMIT_K=" << results.recommended_limit_k << "\n\n";
    
    ofs << "# THRESHOLD_COEFF: Phase 2 过滤阈值放宽系数\n";
    ofs << "# 含义: Phase 2 阈值应该是最小 PQ 距离的多少倍，以容忍 PQ 误差\n";
    ofs << "THRESHOLD_COEFF=" << std::fixed << std::setprecision(3) << results.threshold_coeff << "\n\n";
    
    ofs << "# ===== 使用建议 =====\n";
    ofs << "# 1. TOP_M 用于 CAGRA 搜索返回的候选聚类数量上限\n";
    ofs << "# 2. P1_LISTS 用于 Phase 1 精细扫描的聚类数量\n";
    ofs << "# 3. LIMIT_K 用于每个聚类内 PQ 排序后的截断阈值\n";
    ofs << "# 4. THRESHOLD_COEFF 用于 Phase 2 阈值过滤的放宽系数\n";
    
    ofs.close();
    std::cout << "\n配置文件已保存到: " << output_file << std::endl;
}

// 打印详细统计信息
void print_detailed_stats(const TuneResults& results) {
    std::cout << "\n========== 详细统计信息 ==========" << std::endl;
    
    // 聚类排名分布
    std::cout << "\n聚类排名分布 (前 10 个百分位):" << std::endl;
    std::vector<int> cluster_ranks_copy = results.cluster_ranks;
    std::sort(cluster_ranks_copy.begin(), cluster_ranks_copy.end());
    
    for (int p = 10; p <= 100; p += 10) {
        int idx = static_cast<int>(cluster_ranks_copy.size() * p / 100.0) - 1;
        if (idx < 0) idx = 0;
        if (idx >= cluster_ranks_copy.size()) idx = cluster_ranks_copy.size() - 1;
        std::cout << "  P" << p << ": " << cluster_ranks_copy[idx] << std::endl;
    }
    
    // PQ 排名分布
    std::cout << "\nPQ 排名分布 (聚类内, 基于残差PQ距离):" << std::endl;
    std::vector<int> pq_ranks_copy = results.pq_ranks;
    std::sort(pq_ranks_copy.begin(), pq_ranks_copy.end());
    
    for (int p = 10; p <= 100; p += 10) {
        int idx = static_cast<int>(pq_ranks_copy.size() * p / 100.0) - 1;
        if (idx < 0) idx = 0;
        if (idx >= pq_ranks_copy.size()) idx = pq_ranks_copy.size() - 1;
        std::cout << "  P" << p << ": " << pq_ranks_copy[idx] << std::endl;
    }
    
    // 精确排名分布
    std::cout << "\n精确排名分布 (聚类内, 基于精确欧氏距离):" << std::endl;
    std::vector<int> exact_ranks_copy = results.exact_ranks;
    std::sort(exact_ranks_copy.begin(), exact_ranks_copy.end());
    
    for (int p = 10; p <= 100; p += 10) {
        int idx = static_cast<int>(exact_ranks_copy.size() * p / 100.0) - 1;
        if (idx < 0) idx = 0;
        if (idx >= exact_ranks_copy.size()) idx = exact_ranks_copy.size() - 1;
        std::cout << "  P" << p << ": " << exact_ranks_copy[idx] << std::endl;
    }
    
    // 排名差异分布 (PQ排名 - 精确排名)
    std::cout << "\n排名差异分布 (PQ排名 - 精确排名):" << std::endl;
    std::vector<int> rank_diffs_copy = results.rank_diffs;
    std::sort(rank_diffs_copy.begin(), rank_diffs_copy.end());
    
    for (int p = 10; p <= 100; p += 10) {
        int idx = static_cast<int>(rank_diffs_copy.size() * p / 100.0) - 1;
        if (idx < 0) idx = 0;
        if (idx >= rank_diffs_copy.size()) idx = rank_diffs_copy.size() - 1;
        std::cout << "  P" << p << ": " << rank_diffs_copy[idx] << std::endl;
    }
    
    // 距离比率分布
    std::cout << "\n距离比率分布 (前 10 个百分位):" << std::endl;
    std::vector<float> ratios_copy = results.pq_ratios;
    std::sort(ratios_copy.begin(), ratios_copy.end());
    
    for (int p = 10; p <= 100; p += 10) {
        int idx = static_cast<int>(ratios_copy.size() * p / 100.0) - 1;
        if (idx < 0) idx = 0;
        if (idx >= ratios_copy.size()) idx = ratios_copy.size() - 1;
        std::cout << "  P" << p << ": " << std::fixed << std::setprecision(3) << ratios_copy[idx] << std::endl;
    }
    
    std::cout << "\n===================================" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "用法: ./tune_parameters <base.fvecs> [options]" << std::endl;
        std::cerr << "选项:" << std::endl;
        std::cerr << "  -n <num>     采样查询数量 (默认: 1000)" << std::endl;
        std::cerr << "  -k <num>     目标 Top-K (默认: 10)" << std::endl;
        std::cerr << "  -o <file>    输出配置文件 (默认: search_config.txt)" << std::endl;
        return 1;
    }
    
    // 解析命令行参数
    std::string base_file = argv[1];
    TuneConfig config;
    std::string output_file = "search_config.txt";
    
    for (int i = 2; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-n" && i + 1 < argc) {
            config.n_samples = std::atoi(argv[++i]);
        } else if (arg == "-k" && i + 1 < argc) {
            config.final_k = std::atoi(argv[++i]);
        } else if (arg == "-o" && i + 1 < argc) {
            output_file = argv[++i];
        }
    }
    
    std::cout << "========== 参数调优工具 ==========" << std::endl;
    std::cout << "数据集: " << base_file << std::endl;
    std::cout << "采样数量: " << config.n_samples << std::endl;
    std::cout << "目标 Top-K: " << config.final_k << std::endl;
    std::cout << "输出文件: " << output_file << std::endl;
    
    // 加载数据集
    std::cout << "\n加载数据集..." << std::endl;
    float* base_data;
    int n_base = read_vecs<float>(base_file, base_data, DIM);
    std::cout << "数据集大小: " << n_base << " 个向量" << std::endl;
    
    // 加载 IVF 索引
    std::cout << "\n加载 IVF 索引..." << std::endl;
    std::ifstream ifs("../res/ivf_data.bin", std::ios::binary);
    if (!ifs) {
        std::cerr << "错误: 无法打开 ivf_data.bin，请先运行 build_global" << std::endl;
        delete[] base_data;
        return 1;
    }
    
    int total_vecs, n_clusters;
    ifs.read((char*)&total_vecs, sizeof(int));
    ifs.read((char*)&n_clusters, sizeof(int));
    
    std::vector<int> cluster_offsets(n_clusters + 1);
    std::vector<int> vector_ids(total_vecs);
    std::vector<uint8_t> pq_codes(total_vecs * PQ_M);
    std::vector<float> centroids(n_clusters * DIM);
    
    ifs.read((char*)cluster_offsets.data(), cluster_offsets.size() * sizeof(int));
    ifs.read((char*)vector_ids.data(), vector_ids.size() * sizeof(int));
    ifs.read((char*)pq_codes.data(), pq_codes.size() * sizeof(uint8_t));
    ifs.read((char*)centroids.data(), centroids.size() * sizeof(float));
    ifs.close();
    
    std::cout << "聚类数量: " << n_clusters << std::endl;
    std::cout << "总向量数: " << total_vecs << std::endl;
    
    // 加载 PQ 码本
    std::cout << "\n加载 PQ 码本..." << std::endl;
    std::vector<float> pq_codebook(PQ_M * PQ_K * PQ_SUB_DIM);
    read_binary_vector("../res/global_pq_codebook.bin", pq_codebook);
    std::cout << "PQ 码本大小: " << pq_codebook.size() << " 个浮点数" << std::endl;
    
    // 执行参数调优
    TuneResults results = tune_search_parameters(
        base_data, n_base,
        centroids.data(), n_clusters,
        cluster_offsets.data(),
        vector_ids.data(),
        pq_codes.data(),
        pq_codebook.data(),
        config
    );
    
    // 打印结果
    std::cout << "\n========== 推荐参数 ==========" << std::endl;
    std::cout << "RECOMMENDED_TOP_M     = " << results.recommended_top_m << " (P98 分位数)" << std::endl;
    std::cout << "RECOMMENDED_P1_LISTS  = " << results.recommended_p1_lists << " (拐点检测)" << std::endl;
    std::cout << "RECOMMENDED_LIMIT_K   = " << results.recommended_limit_k << std::endl;
    std::cout << "THRESHOLD_COEFF       = " << std::fixed << std::setprecision(3) 
              << results.threshold_coeff << std::endl;
    std::cout << "==============================" << std::endl;
    
    // 打印详细统计
    print_detailed_stats(results);
    
    // 保存配置文件
    save_tune_results(results, output_file);
    
    // 清理
    delete[] base_data;
    
    std::cout << "\n调优完成！" << std::endl;
    return 0;
}
