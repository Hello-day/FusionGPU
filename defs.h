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
#define K_CLUSTERS  4096 
#define DEFAULT_BATCH_SIZE 128
#define DEFAULT_TOP_M 20     
#define DEFAULT_RERANK_M 10000 

// --- 搜索参数默认值 (可通过参数调优工具获得推荐值) ---
#define DEFAULT_P1_LISTS 14             // Phase 1 密集搜索的聚类数量
#define DEFAULT_LIMIT_K 34            // 每个聚类内保留的候选数量
#define DEFAULT_THRESHOLD_COEFF 1.0f    // Phase 2 阈值放宽系数 

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
};

// 细粒度耗时统计事件
struct BatchLogicEvents {
    cudaEvent_t evt_start;
    cudaEvent_t evt_prune_count;     // [Fused] Prune + Count 融合
    cudaEvent_t evt_precompute;
    cudaEvent_t evt_scan_offset;
    cudaEvent_t evt_resize_sync;
    
    cudaEvent_t evt_scan_phase1_end; // Phase 1 结束
    cudaEvent_t evt_scan_cand;       // Phase 2 结束
    
    cudaEvent_t evt_compact;         // [New] Compaction 结束
    cudaEvent_t evt_sort;
    cudaEvent_t evt_gather;
};

// 耗时统计结果
struct GPUStageTimings {
    float prune_count_ms = 0;        // [Fused] Prune + Count 融合
    float precompute_ms = 0;
    float scan_offset_ms = 0;
    float resize_sync_ms = 0;
    float scan_phase1_ms = 0;
    float scan_phase2_ms = 0;
    float compact_ms = 0;            // [New]
    float sort_ms = 0;
    float gather_ms = 0;
    float total_ms = 0;
    
    long long total_candidates = 0; // 实际有效候选者
    long long total_max_candidates = 0; // 理论最大候选者(用于对比优化效果)
};
#endif // DEFS_H_