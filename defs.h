#ifndef DEFS_H_
#define DEFS_H_

#include <cstdint>
#include <vector>
#include <cuda_runtime.h> 

// --- 基础配置 ---
#define DIM 128
#define PQ_M 32
#define PQ_K 256
#define PQ_SUB_DIM (DIM / PQ_M)

// --- 默认参数 ---
#define K_CLUSTERS  2048 
#define DEFAULT_BATCH_SIZE 128
#define DEFAULT_TOP_M 20     
#define DEFAULT_RERANK_M 10000 

// --- 搜索参数默认值 (可通过参数调优工具获得推荐值) ---
#define DEFAULT_P1_LISTS 12           // Phase 1 密集搜索的聚类数量
#define DEFAULT_LIMIT_K 50            // 每个聚类内保留的候选数量（增加以提高召回率）
#define DEFAULT_THRESHOLD_COEFF 1.0f    // Phase 2 阈值放宽系数 

// --- Shared Memory 使用说明 ---
// Phase1 Kernel:
//   - s_residual: 128 * 4B = 512 bytes (用于直接计算)
//   - s_shared_buffer: 32 * 256 * 4B = 32KB (用于预计算表 LUT)
//   - CUB sort storage: ~16KB (小聚类排序时)
//   总计: ~49KB (在 A100 默认 48KB 限制内，需要 opt-in 到 64KB)
//
// Phase2 Kernel:
//   - s_residual: 512 bytes
//   - s_shared_buffer: 32KB (用于预计算表 LUT)
//   - s_ids/s_dists: 128 * 8B = 1KB (批量收集缓冲区)
//   总计: ~34KB (在 48KB 限制内)

// --- Heuristic Re-ranking 默认参数 ---
#define DEFAULT_RERANK_MINI_BATCH 128  
#define DEFAULT_RERANK_EPSILON 0.1f        
#define DEFAULT_RERANK_BETA 2              

using pq_code_t = uint8_t;

template<typename T>
struct point_t {
    T coordinates[DIM];
};

// --- GPU 索引结构 (只读) ---
struct IVFIndexGPU {
    float* d_pq_codebook;       // [PQ_M, PQ_K, PQ_SUB_DIM] 
    int* d_cluster_offsets;     // [K_CLUSTERS + 1]
    uint8_t* d_all_pq_codes;    // [Total_Vectors, PQ_M] 
    int* d_all_vector_ids;      // [Total_Vectors]
    float* d_centroids;         // [K_CLUSTERS, DIM]
    int num_clusters;           // 聚类数量
};

// 细粒度耗时统计事件
struct BatchLogicEvents {
    cudaEvent_t evt_start;
    cudaEvent_t evt_prune_count;     // [Fused] Prune + Count 融合
    // 目前代码中统一使用 evt_precompute 来表示 Precompute 阶段
    // 为了兼容旧版本，这里保留原有的 evt_precompute_p1 命名，但实际上只用 evt_precompute
    cudaEvent_t evt_precompute_p1;   // (兼容) 旧命名：Phase 1 预计算
    cudaEvent_t evt_precompute;      // 新代码使用的事件名
    cudaEvent_t evt_scan_offset;
    cudaEvent_t evt_resize_sync;
    
    cudaEvent_t evt_scan_phase1_end; // Phase 1 扫描结束（无排序）
    cudaEvent_t evt_d2h_p1_start;    // Phase 1 D2H 开始
    cudaEvent_t evt_d2h_p1_end;      // Phase 1 D2H 结束
    cudaEvent_t evt_precompute_p2;   // Phase 2 预计算（与 CPU 排序并行）
    cudaEvent_t evt_h2d_threshold;   // 阈值 H2D
    cudaEvent_t evt_scan_cand;       // Phase 2 结束
    
    cudaEvent_t evt_compact;         // [New] Compaction 结束
    cudaEvent_t evt_sort;
    cudaEvent_t evt_gather;
};

// 耗时统计结果
struct GPUStageTimings {
    float prune_count_ms = 0;        // [Fused] Prune + Count 融合
    // 旧版本拆成 precompute_p1 / precompute_p2，这里增加统一的 precompute_ms 字段
    // 以兼容当前 search_pipeline.cu 中的使用方式
    float precompute_p1_ms = 0;      // (兼容) Phase 1 预计算
    float precompute_ms  = 0;        // 当前代码使用的总 Precompute 时间（含与 D2H 并行）
    float scan_offset_ms = 0;
    float resize_sync_ms = 0;
    float scan_phase1_ms = 0;        // Phase 1 扫描（无排序）
    float d2h_p1_ms = 0;             // Phase 1 D2H
    float cpu_sort_ms = 0;           // CPU 排序（从外部记录）
    float precompute_p2_ms = 0;      // Phase 2 预计算（与 CPU 并行）
    float h2d_threshold_ms = 0;      // 阈值 H2D
    float scan_phase2_ms = 0;        // Phase 2 扫描
    float compact_ms = 0;            // [New]
    float sort_ms = 0;
    float gather_ms = 0;
    float total_ms = 0;
    
    long long total_candidates = 0; // 实际有效候选者
    long long total_max_candidates = 0; // 理论最大候选者(用于对比优化效果)
};
#endif // DEFS_H_