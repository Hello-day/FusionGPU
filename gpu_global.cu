#include "defs.h"
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cfloat>
#include <algorithm>

// ==========================================
// 1. 基础 Kernels
// ==========================================

__global__ void prune_candidates_kernel(
    uint32_t* __restrict__ ids,   
    const float* __restrict__ top_m_dists,  
    int top_m,
    float prune_ratio
) {
    int q_idx = blockIdx.x; 
    float best_dist = top_m_dists[q_idx * top_m + 0];
    float threshold = best_dist * prune_ratio;

    for (int m_idx = threadIdx.x; m_idx < top_m; m_idx += blockDim.x) {
        int idx = q_idx * top_m + m_idx;
        if (top_m_dists[idx] > threshold) {
            ids[idx] = 0xFFFFFFFF; 
        }
    }
}

// 融合内核: Prune + Count + Prefix Sum (per-query)
__global__ void prune_and_count_fused_kernel(
    uint32_t* __restrict__ ids,
    const float* __restrict__ top_m_dists,
    const int* __restrict__ cluster_offsets,
    int* __restrict__ out_counts,
    int top_m,
    float prune_ratio
) {
    int q_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    // 步骤1: Prune - 标记超过阈值的候选
    float best_dist = top_m_dists[q_idx * top_m + 0];
    float threshold = best_dist * prune_ratio;
    
    for (int m_idx = tid; m_idx < top_m; m_idx += blockDim.x) {
        int idx = q_idx * top_m + m_idx;
        if (top_m_dists[idx] > threshold) {
            ids[idx] = 0xFFFFFFFF;
        }
    }
    __syncthreads();  // 确保所有线程完成剪枝
    
    // 步骤2: Count - 统计有效聚类的向量总数
    __shared__ int s_count;
    if (tid == 0) s_count = 0;
    __syncthreads();
    
    // 每个线程处理部分候选，累加到共享内存
    int local_count = 0;
    for (int m_idx = tid; m_idx < top_m; m_idx += blockDim.x) {
        uint32_t raw_id = ids[q_idx * top_m + m_idx];
        if (raw_id != 0xFFFFFFFF) {
            int c_id = (int)raw_id;
            local_count += cluster_offsets[c_id + 1] - cluster_offsets[c_id];
        }
    }
    
    // 使用共享内存原子操作减少冲突
    atomicAdd(&s_count, local_count);
    __syncthreads();
    
    // 步骤3: 写回全局内存
    if (tid == 0) {
        out_counts[q_idx] = s_count;
    }
}

__global__ void batch_residual_precompute_kernel(
    const float* __restrict__ queries, const float* __restrict__ centroids,    
    const uint32_t* __restrict__ top_m_ids, const float* __restrict__ codebook,     
    float* __restrict__ out_tables, int top_m
) {
    int q_idx = blockIdx.x;
    int m_idx = blockIdx.y; 
    uint32_t raw_id = top_m_ids[q_idx * top_m + m_idx];
    if (raw_id == 0xFFFFFFFF) return; 

    int c_id = (int)raw_id;
    int tid = threadIdx.x;
    
    // 优化1: 使用共享内存缓存残差向量，避免重复计算
    __shared__ float s_residual[DIM];
    
    // 协作加载并计算残差 (query - centroid)
    for (int i = tid; i < DIM; i += blockDim.x) {
        s_residual[i] = queries[q_idx * DIM + i] - centroids[c_id * DIM + i];
    }
    __syncthreads();
    
    // 优化2: 使用网格跨步循环处理所有PQ码字
    for (int k_idx = tid; k_idx < PQ_K; k_idx += blockDim.x) {
        
        // 优化3: 循环展开 - 让编译器优化PQ_M循环
        #pragma unroll 4
        for (int sub = 0; sub < PQ_M; ++sub) {
            int dim_offset = sub * PQ_SUB_DIM;
            const float* c_sub = codebook + (sub * PQ_K + k_idx) * PQ_SUB_DIM;
            
            float dist = 0.0f;
            
            // 优化4: 根据PQ_SUB_DIM的值选择不同的优化路径
            #if PQ_SUB_DIM == 4
                // 向量化加载 - 一次处理4个float
                float4 residual_vec = *reinterpret_cast<const float4*>(&s_residual[dim_offset]);
                float4 codebook_vec = *reinterpret_cast<const float4*>(c_sub);
                
                float4 diff_vec;
                diff_vec.x = residual_vec.x - codebook_vec.x;
                diff_vec.y = residual_vec.y - codebook_vec.y;
                diff_vec.z = residual_vec.z - codebook_vec.z;
                diff_vec.w = residual_vec.w - codebook_vec.w;
                
                // 使用FMA指令优化
                dist = fmaf(diff_vec.x, diff_vec.x, 
                       fmaf(diff_vec.y, diff_vec.y,
                       fmaf(diff_vec.z, diff_vec.z, diff_vec.w * diff_vec.w)));
            #elif PQ_SUB_DIM == 8
                // 使用两个float4
                float4 residual_vec1 = *reinterpret_cast<const float4*>(&s_residual[dim_offset]);
                float4 residual_vec2 = *reinterpret_cast<const float4*>(&s_residual[dim_offset + 4]);
                float4 codebook_vec1 = *reinterpret_cast<const float4*>(c_sub);
                float4 codebook_vec2 = *reinterpret_cast<const float4*>(c_sub + 4);
                
                float4 diff1, diff2;
                diff1.x = residual_vec1.x - codebook_vec1.x;
                diff1.y = residual_vec1.y - codebook_vec1.y;
                diff1.z = residual_vec1.z - codebook_vec1.z;
                diff1.w = residual_vec1.w - codebook_vec1.w;
                
                diff2.x = residual_vec2.x - codebook_vec2.x;
                diff2.y = residual_vec2.y - codebook_vec2.y;
                diff2.z = residual_vec2.z - codebook_vec2.z;
                diff2.w = residual_vec2.w - codebook_vec2.w;
                
                dist = fmaf(diff1.x, diff1.x, fmaf(diff1.y, diff1.y, fmaf(diff1.z, diff1.z, diff1.w * diff1.w)));
                dist = fmaf(diff2.x, diff2.x, fmaf(diff2.y, diff2.y, fmaf(diff2.z, diff2.z, fmaf(diff2.w, diff2.w, dist))));
            #else
                // 通用路径 - 支持任意PQ_SUB_DIM
                // 优化5: 循环展开内层循环
                #pragma unroll
                for (int d = 0; d < PQ_SUB_DIM; ++d) {
                    float diff = s_residual[dim_offset + d] - c_sub[d];
                    dist = fmaf(diff, diff, dist);  // 使用FMA: dist += diff * diff
                }
            #endif
            
            // 优化6: 合并写入 - 确保内存访问合并
            out_tables[(size_t)q_idx * (top_m * PQ_M * PQ_K) + 
                      (size_t)m_idx * (PQ_M * PQ_K) + 
                      sub * PQ_K + k_idx] = dist;
        }
    }
}

__global__ void ivf_count_kernel(
    const uint32_t* top_m_ids, const int* cluster_offsets, int* out_counts, int top_m
) {
    int q_idx = blockIdx.x;
    for (int m_idx = threadIdx.x; m_idx < top_m; m_idx += blockDim.x) {
        uint32_t raw_id = top_m_ids[q_idx * top_m + m_idx];
        if (raw_id == 0xFFFFFFFF) continue;
        int c_id = (int)raw_id;
        atomicAdd(&out_counts[q_idx], cluster_offsets[c_id + 1] - cluster_offsets[c_id]);
    }
}

// ==========================================
// 2. Scan & Two-Phase Kernels
// ==========================================

// ==========================================
// 通用向量化 PQ 距离计算 (支持任意 PQ_M)
// ==========================================
// 
// 加载策略：
//   - 每 16 个 codes 使用 uint4 加载 (16 bytes)
//   - 每 4 个 codes 使用 uchar4 处理
//   - 剩余部分逐个处理
//
// 编译期常量决定分支，无运行时开销

// 辅助宏：处理 4 个连续 codes (一个 uchar4)
#define PROCESS_UCHAR4(codes, base_idx, lut, dist) do { \
    dist += lut[(base_idx + 0) * PQ_K + codes.x]; \
    dist += lut[(base_idx + 1) * PQ_K + codes.y]; \
    dist += lut[(base_idx + 2) * PQ_K + codes.z]; \
    dist += lut[(base_idx + 3) * PQ_K + codes.w]; \
} while(0)

// 辅助宏：处理 16 个连续 codes (一个 uint4 = 4 个 uchar4)
#define PROCESS_UINT4(u4_val, base_idx, lut, dist) do { \
    const uchar4* _c = reinterpret_cast<const uchar4*>(&u4_val); \
    PROCESS_UCHAR4(_c[0], base_idx + 0, lut, dist); \
    PROCESS_UCHAR4(_c[1], base_idx + 4, lut, dist); \
    PROCESS_UCHAR4(_c[2], base_idx + 8, lut, dist); \
    PROCESS_UCHAR4(_c[3], base_idx + 12, lut, dist); \
} while(0)

__device__ __forceinline__ float compute_pq_dist_vectorized(
    const uint8_t* __restrict__ pq_codes_ptr,
    const float* __restrict__ lut
) {
    float dist = 0.0f;
    
    // 编译期常量
    constexpr int NUM_UINT4 = PQ_M / 16;        // 完整的 uint4 块数
    constexpr int REMAINING_AFTER_UINT4 = PQ_M % 16;
    constexpr int NUM_UCHAR4 = REMAINING_AFTER_UINT4 / 4;  // 剩余的 uchar4 块数
    constexpr int REMAINING_BYTES = REMAINING_AFTER_UINT4 % 4;  // 最终剩余字节
    
    const uint4* ptr_u4 = reinterpret_cast<const uint4*>(pq_codes_ptr);
    
    // 阶段 1：处理完整的 uint4 块 (每块 16 bytes)
    #pragma unroll
    for (int i = 0; i < NUM_UINT4; ++i) {
        uint4 codes = ptr_u4[i];
        PROCESS_UINT4(codes, i * 16, lut, dist);
    }
    
    // 阶段 2：处理剩余的 uchar4 块 (每块 4 bytes)
    if constexpr (NUM_UCHAR4 > 0) {
        const uchar4* ptr_uc4 = reinterpret_cast<const uchar4*>(pq_codes_ptr + NUM_UINT4 * 16);
        #pragma unroll
        for (int i = 0; i < NUM_UCHAR4; ++i) {
            uchar4 codes = ptr_uc4[i];
            PROCESS_UCHAR4(codes, NUM_UINT4 * 16 + i * 4, lut, dist);
        }
    }
    
    // 阶段 3：处理最后剩余的字节 (0-3 bytes)
    if constexpr (REMAINING_BYTES > 0) {
        constexpr int START_IDX = NUM_UINT4 * 16 + NUM_UCHAR4 * 4;
        #pragma unroll
        for (int i = 0; i < REMAINING_BYTES; ++i) {
            dist += lut[(START_IDX + i) * PQ_K + pq_codes_ptr[START_IDX + i]];
        }
    }
    
    return dist;
}

#undef PROCESS_UCHAR4
#undef PROCESS_UINT4

__global__ void ivf_scan_phase1_kernel(
    const int* cluster_offsets, const uint32_t* top_m_ids, 
    const uint8_t* all_pq_codes, const int* all_vec_ids,
    const float* global_tables, 
    const int* query_base_offsets, int* query_atomic_counters,
    int* out_ids, float* out_dists, 
    int top_m, int p1_lists, int limit_k, float* d_list_cutoffs
) 
{
    int q_idx = blockIdx.x; 
    int m_idx = blockIdx.y; 
    if (m_idx >= p1_lists) return;
    if (threadIdx.x == 0) d_list_cutoffs[q_idx * p1_lists + m_idx] = 0.0f;  // 初始化为0，后面取最大值

    uint32_t raw_id = top_m_ids[q_idx * top_m + m_idx];
    if (raw_id == 0xFFFFFFFF) return;

    int c_id = (int)raw_id;
    int start = cluster_offsets[c_id]; 
    int len = cluster_offsets[c_id + 1] - start;
    if (len == 0) return;

    const int ITEMS_PER_THREAD = 16;
    const int BLOCK_THREADS = 256; 
    int write_count = (len < limit_k) ? len : limit_k;
    if (len > BLOCK_THREADS * ITEMS_PER_THREAD) len = BLOCK_THREADS * ITEMS_PER_THREAD; 

    __shared__ int write_start;
    if (threadIdx.x == 0) write_start = atomicAdd(&query_atomic_counters[q_idx], write_count);
    __syncthreads();

    int global_base = query_base_offsets[q_idx] + write_start;
    const float* my_table = global_tables + (size_t)q_idx * (top_m * PQ_M * PQ_K) + (size_t)m_idx * (PQ_M * PQ_K);

    typedef cub::BlockRadixSort<float, BLOCK_THREADS, ITEMS_PER_THREAD, int> BlockSort;
    __shared__ typename BlockSort::TempStorage sort_storage;
    float thread_keys[ITEMS_PER_THREAD];
    int   thread_vals[ITEMS_PER_THREAD];

    #pragma unroll
    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int local_idx = threadIdx.x * ITEMS_PER_THREAD + i;
        if (local_idx < len) {
            int ivf_idx = start + local_idx;
            // 使用向量化加载计算 PQ 距离
            float dist = compute_pq_dist_vectorized(
                all_pq_codes + (size_t)ivf_idx * PQ_M,
                my_table
            );
            thread_keys[i] = dist; 
            thread_vals[i] = all_vec_ids[ivf_idx];
        } else {
            thread_keys[i] = FLT_MAX; 
            thread_vals[i] = -1;
        }
    }
    __syncthreads(); 
    BlockSort(sort_storage).Sort(thread_keys, thread_vals);
    __syncthreads(); 

    for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
        int rank = threadIdx.x * ITEMS_PER_THREAD + i;
        if (rank < write_count) {
            out_ids[global_base + rank] = thread_vals[i];
            out_dists[global_base + rank] = thread_keys[i];
        }
        // 记录第 limit_k 名的距离 (用于计算阈值)
        if (rank == write_count - 1) d_list_cutoffs[q_idx * p1_lists + m_idx] = thread_keys[i];
    }
}

// 计算阈值: 取 P1_LISTS 中最小的 cutoff * threshold_coeff
__global__ void compute_threshold_kernel(const float* d_list_cutoffs, float* d_thresholds, int p1_lists, float threshold_coeff) {
    int q_idx = blockIdx.x;
    if (threadIdx.x != 0) return;
    
    float min_dist = FLT_MAX;
    for (int i = 0; i < p1_lists; ++i) {
        float val = d_list_cutoffs[q_idx * p1_lists + i];
        if (val < min_dist) {
            min_dist = val;
        }
    }
    d_thresholds[q_idx] = min_dist * threshold_coeff;
}

__global__ void ivf_scan_phase2_kernel(
    const int* cluster_offsets, const uint32_t* top_m_ids, 
    const uint8_t* all_pq_codes, const int* all_vec_ids,
    const float* global_tables, 
    const int* query_base_offsets, int* query_atomic_counters,
    int* out_ids, float* out_dists, 
    int top_m, const float* d_thresholds
) 
{
    int q_idx = blockIdx.x; 
    int m_idx = 8 + blockIdx.y; 
    if (m_idx >= top_m) return;
    float threshold = d_thresholds[q_idx];
    
    uint32_t raw_id = top_m_ids[q_idx * top_m + m_idx];
    if (raw_id == 0xFFFFFFFF) return;

    int c_id = (int)raw_id;
    int start = cluster_offsets[c_id]; 
    int len = cluster_offsets[c_id + 1] - start;
    if (len == 0) return;

    const float* my_table = global_tables + (size_t)q_idx * (top_m * PQ_M * PQ_K) + (size_t)m_idx * (PQ_M * PQ_K);
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        int ivf_idx = start + i; 
        // 使用向量化加载计算 PQ 距离
        float dist = compute_pq_dist_vectorized(
            all_pq_codes + (size_t)ivf_idx * PQ_M,
            my_table
        );
        
        if (dist <= threshold) {
            int global_pos = atomicAdd(&query_atomic_counters[q_idx], 1);
            int global_base = query_base_offsets[q_idx];
            out_ids[global_base + global_pos] = all_vec_ids[ivf_idx];
            out_dists[global_base + global_pos] = dist;
        }
    }
}

// ==========================================
// [New] Compaction Kernel
// ==========================================
// 将稀疏数据 (src) 搬运到紧凑连续的 buffer (dst)
__global__ void compact_candidates_kernel(
    const int* __restrict__ src_ids, const float* __restrict__ src_dists,
    const int* __restrict__ src_offsets,
    const int* __restrict__ valid_counts, // d_atomic
    const int* __restrict__ dst_offsets,
    int* __restrict__ dst_ids, float* __restrict__ dst_dists,
    int batch_size
) {
    int q_idx = blockIdx.x;
    if (q_idx >= batch_size) return;

    int count = valid_counts[q_idx];
    int src_base = src_offsets[q_idx];
    int dst_base = dst_offsets[q_idx];

    // 每个 Block 处理一个 Query 的搬运
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        dst_ids[dst_base + i] = src_ids[src_base + i];
        dst_dists[dst_base + i] = src_dists[src_base + i];
    }
}

// Gather Kernel
__global__ void gather_top_m_kernel(
    const int* __restrict__ sorted_ids, const float* __restrict__ sorted_dists,
    const int* __restrict__ offsets, const int* __restrict__ counts,
    int* __restrict__ out_top_ids, float* __restrict__ out_top_dists, int* __restrict__ out_real_counts, 
    int rerank_m
) {
    int q_idx = blockIdx.x; int lane = threadIdx.x;
    if (lane >= rerank_m) return;
    int count = counts[q_idx];
    int actual_k = (count < rerank_m) ? count : rerank_m;
    if (lane == 0) out_real_counts[q_idx] = actual_k;
    
    if (lane < actual_k) {
        int src_idx = offsets[q_idx] + lane;
        out_top_ids[q_idx * rerank_m + lane]   = sorted_ids[src_idx];
        out_top_dists[q_idx * rerank_m + lane] = sorted_dists[src_idx];
    } else {
        out_top_ids[q_idx * rerank_m + lane] = -1;
        out_top_dists[q_idx * rerank_m + lane] = FLT_MAX;
    }
}

// ==========================================
// 3. Rerank Logic
// ==========================================
__global__ void init_rerank_state_kernel(int* d_final_ids, float* d_final_dists, int* d_stable_cnt, bool* d_finished, int batch_size, int final_k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batch_size) {
        d_stable_cnt[idx] = 0; d_finished[idx] = false;
        for (int k = 0; k < final_k; ++k) { d_final_ids[idx * final_k + k] = -1; d_final_dists[idx * final_k + k] = FLT_MAX; }
    }
}

__global__ void heuristic_rerank_kernel(const int* cid, const int* ccnt, const float* q, const float* v, int* tid, float* td, int* stab, bool* fin, int dim, int rm, int off, int mb, int k, int tv, float eps, int b) {
    int q_idx = blockIdx.x;
    if (fin[q_idx]) return;
    int total = ccnt[q_idx];
    if (off >= total) { if (threadIdx.x == 0) fin[q_idx] = true; return; }

    extern __shared__ char smem[];
    float* s_q = (float*)smem;
    size_t align = (dim * sizeof(float) + 15)/16*16;
    int cap = k + mb;
    float* s_d = (float*)(smem + align);
    int* s_i = (int*)&s_d[cap];

    for(int i=threadIdx.x; i<dim; i+=blockDim.x) s_q[i] = q[q_idx*dim+i];
    for(int i=threadIdx.x; i<k; i+=blockDim.x) { s_d[i] = td[q_idx*k+i]; s_i[i] = tid[q_idx*k+i]; }
    __syncthreads();

    int eff = 0; if(off < total) eff = min(mb, total - off);
    for(int i=threadIdx.x; i<mb; i+=blockDim.x) {
        int widx = k+i; float dist = FLT_MAX; int vid = -1;
        if(i < eff) {
            vid = cid[q_idx*rm + off + i];
            if(vid >= 0 && vid < tv) {
                const float* vec = v + (size_t)vid*dim;
                float local_d = 0.0f;
                for(int j=0; j<dim; ++j) { float diff = s_q[j] - vec[j]; local_d += diff*diff; }
                dist = local_d;
            }
        }
        s_d[widx] = dist; s_i[widx] = vid;
    }
    __syncthreads();

    typedef cub::BlockRadixSort<float, 256, 1, int> BlockSort;
    __shared__ typename BlockSort::TempStorage temp;
    float key[1] = {FLT_MAX}; int val[1] = {-1};
    if(threadIdx.x < cap) { key[0] = s_d[threadIdx.x]; val[0] = s_i[threadIdx.x]; }
    __syncthreads();
    BlockSort(temp).Sort(key, val);
    if(threadIdx.x < k) { s_d[threadIdx.x] = key[0]; s_i[threadIdx.x] = val[0]; }
    __syncthreads();

    if(threadIdx.x == 0) {
        int chg = 0;
        for(int i=0; i<k; ++i) {
            int nid = s_i[i]; if(nid == -1) continue; bool f = false;
            for(int j=0; j<k; ++j) if(tid[q_idx*k+j] == nid) { f = true; break; }
            if(!f) chg++;
        }
        if((float)chg/k <= eps) stab[q_idx]++; else stab[q_idx] = 0;
        if(stab[q_idx] >= b) fin[q_idx] = true;
    }
    __syncthreads();
    if(threadIdx.x < k) { tid[q_idx*k+threadIdx.x] = s_i[threadIdx.x]; td[q_idx*k+threadIdx.x] = s_d[threadIdx.x]; }
}

void run_gpu_rerank(float* dq, int* tid, int* tcnt, float* dv, int* fid, float* fd, int* st, bool* fin, int bs, int rm, int k, int tv, int mb, float eps, int b, cudaStream_t s) {
    init_rerank_state_kernel<<<(bs+255)/256, 256, 0, s>>>(fid, fd, st, fin, bs, k);
    int cap = k + mb; size_t smem = DIM*4 + 16 + cap*8;
    cudaFuncSetAttribute(heuristic_rerank_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 96*1024);
    for(int off=0; off<rm; off+=mb) 
        heuristic_rerank_kernel<<<bs, 256, smem, s>>>(tid, tcnt, dq, dv, fid, fd, st, fin, DIM, rm, off, mb, k, tv, eps, b);
}

// ==========================================
// 4. Logic Wrapper
// ==========================================

void run_gpu_batch_logic(
    const IVFIndexGPU& idx, float* d_queries, 
    uint32_t* d_cagra_top_m, 
    float* d_cagra_dists,    
    float* d_global_tables,
    int*& d_flat_ids, float*& d_flat_dists, int*& d_flat_ids_alt, float*& d_flat_dists_alt, 
    size_t& current_pool_cap, int* d_counts, int* d_offsets, int* d_atomic,
    int* d_top_ids, float* d_top_dists, int* d_top_counts,
    int batch_size, int top_m, int rerank_m, int final_k, 
    float prune_ratio,
    float* d_list_cutoffs, 
    float* d_thresholds, 
    int* d_compact_offsets,
    cudaStream_t stream_main,
    BatchLogicEvents* events,
    int p1_lists = DEFAULT_P1_LISTS,              // Phase 1 密集搜索的聚类数量
    int limit_k = DEFAULT_LIMIT_K,                // 每个聚类内保留的候选数量
    float threshold_coeff = DEFAULT_THRESHOLD_COEFF  // Phase 2 阈值放宽系数
) {
    if (events) cudaEventRecord(events->evt_start, stream_main);

    // 0+2. Prune + Count (Fused) - 融合内核减少kernel launch开销
    cudaMemsetAsync(d_counts, 0, batch_size * sizeof(int), stream_main);
    prune_and_count_fused_kernel<<<batch_size, 256, 0, stream_main>>>(
        d_cagra_top_m, d_cagra_dists, idx.d_cluster_offsets, d_counts, top_m, prune_ratio);
    if (events) cudaEventRecord(events->evt_prune_count, stream_main);

    // 3. Scan Offsets (提前执行，为Resize Check准备数据)
    void *d_temp_scan = NULL; size_t temp_scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_scan, temp_scan_bytes, d_counts, d_offsets, batch_size, stream_main);
    cudaMalloc(&d_temp_scan, temp_scan_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_scan, temp_scan_bytes, d_counts, d_offsets, batch_size, stream_main);
    cudaFree(d_temp_scan);
    if (events) cudaEventRecord(events->evt_scan_offset, stream_main);

    // 4. Resize Check (异步启动，与Precompute并行)
    // 使用pinned memory实现零拷贝
    static int* h_total_pinned = nullptr;
    if (!h_total_pinned) {
        cudaMallocHost(&h_total_pinned, 2 * sizeof(int));  // [last_count, last_offset]
    }
    
    // 异步拷贝到CPU（不阻塞）
    cudaMemcpyAsync(&h_total_pinned[0], d_counts + batch_size - 1, sizeof(int), cudaMemcpyDeviceToHost, stream_main);
    cudaMemcpyAsync(&h_total_pinned[1], d_offsets + batch_size - 1, sizeof(int), cudaMemcpyDeviceToHost, stream_main);
    
    // 1. Precompute (与CPU的Resize Check并行执行)
    batch_residual_precompute_kernel<<<dim3(batch_size, top_m), 256, 0, stream_main>>>(
        d_queries, idx.d_centroids, d_cagra_top_m, idx.d_pq_codebook, d_global_tables, top_m);
    if (events) cudaEventRecord(events->evt_precompute, stream_main);
    
    // 现在CPU可以在Precompute执行期间处理Resize Check
    // 同步等待D2H完成（此时Precompute正在GPU上执行）
    cudaStreamSynchronize(stream_main);
    
    // CPU计算total（简单加法，几乎零开销）
    int total_items = h_total_pinned[1] + h_total_pinned[0];  // offset + count
    
    // 写回GPU（用于CUB Sort）
    cudaMemcpyAsync(d_offsets + batch_size, &total_items, sizeof(int), cudaMemcpyHostToDevice, stream_main);
    
    // 检查是否需要resize
    size_t needed = total_items;
    if (needed > current_pool_cap) {
        current_pool_cap = needed * 1.5;
        if(d_flat_ids) cudaFree(d_flat_ids); if(d_flat_dists) cudaFree(d_flat_dists);
        if(d_flat_ids_alt) cudaFree(d_flat_ids_alt); if(d_flat_dists_alt) cudaFree(d_flat_dists_alt);
        cudaMalloc(&d_flat_ids, current_pool_cap * sizeof(int));
        cudaMalloc(&d_flat_dists, current_pool_cap * sizeof(float));
        cudaMalloc(&d_flat_ids_alt, current_pool_cap * sizeof(int));
        cudaMalloc(&d_flat_dists_alt, current_pool_cap * sizeof(float));
    }
    
    // 记录resize完成事件（包含同步开销）
    if (events) cudaEventRecord(events->evt_resize_sync, stream_main);

    // 5. Scan Candidates (Into d_flat_ids - Sparse)
    cudaMemsetAsync(d_flat_dists, 0x7F, needed * sizeof(float), stream_main);
    cudaMemsetAsync(d_atomic, 0, batch_size * sizeof(int), stream_main);
    
    // 使用传入的参数，确保不超过 top_m
    int actual_p1_lists = (top_m < p1_lists) ? top_m : p1_lists;
    
    ivf_scan_phase1_kernel<<<dim3(batch_size, actual_p1_lists), 256, 0, stream_main>>>(
        idx.d_cluster_offsets, d_cagra_top_m, idx.d_all_pq_codes, idx.d_all_vector_ids, 
        d_global_tables, d_offsets, d_atomic, d_flat_ids, d_flat_dists, 
        top_m, actual_p1_lists, limit_k, d_list_cutoffs
    );
    compute_threshold_kernel<<<batch_size, 1, 0, stream_main>>>(d_list_cutoffs, d_thresholds, actual_p1_lists, threshold_coeff);
    if (events) cudaEventRecord(events->evt_scan_phase1_end, stream_main);

    if (top_m > actual_p1_lists) {
        ivf_scan_phase2_kernel<<<dim3(batch_size, top_m - actual_p1_lists), 128, 0, stream_main>>>(
            idx.d_cluster_offsets, d_cagra_top_m, idx.d_all_pq_codes, idx.d_all_vector_ids, 
            d_global_tables, d_offsets, d_atomic, d_flat_ids, d_flat_dists, 
            top_m, d_thresholds
        );
    }
    if (events) cudaEventRecord(events->evt_scan_cand, stream_main);

    // =========================================================
    // 6. [Optimization] Compaction
    // =========================================================
    // A. 计算紧凑 Offsets (Scan d_atomic -> d_compact_offsets)
    // 注意：Scan 需要 `batch_size + 1` 大小的 offset 数组
    d_temp_scan = NULL; temp_scan_bytes = 0;
    cub::DeviceScan::ExclusiveSum(d_temp_scan, temp_scan_bytes, d_atomic, d_compact_offsets, batch_size, stream_main);
    cudaMalloc(&d_temp_scan, temp_scan_bytes);
    cub::DeviceScan::ExclusiveSum(d_temp_scan, temp_scan_bytes, d_atomic, d_compact_offsets, batch_size, stream_main);
    cudaFree(d_temp_scan);

    // B. 获取 Compact Total Size (Host-Device Sync needed for CUB Sort API)
    // 性能权衡：这是一个极小的 D2H，延时约 10us，远小于排序优化带来的 ms 级提升
    int compact_last_cnt, compact_last_off;
    cudaMemcpyAsync(&compact_last_cnt, d_atomic + batch_size - 1, sizeof(int), cudaMemcpyDeviceToHost, stream_main);
    cudaMemcpyAsync(&compact_last_off, d_compact_offsets + batch_size - 1, sizeof(int), cudaMemcpyDeviceToHost, stream_main);
    cudaStreamSynchronize(stream_main);
    int compact_total_items = compact_last_off + compact_last_cnt;
    cudaMemcpyAsync(d_compact_offsets + batch_size, &compact_total_items, sizeof(int), cudaMemcpyHostToDevice, stream_main);

    // C. 执行 Compaction 搬运
    // 从 d_flat_ids (Sparse) -> d_flat_ids_alt (Dense)
    // 读取使用 d_offsets, 写入使用 d_compact_offsets
    compact_candidates_kernel<<<batch_size, 256, 0, stream_main>>>(
        d_flat_ids, d_flat_dists,
        d_offsets, d_atomic,
        d_compact_offsets,
        d_flat_ids_alt, d_flat_dists_alt,
        batch_size
    );

    if (events) cudaEventRecord(events->evt_compact, stream_main);

    // 7. Sort (Compact)
    // 现在我们在 Dense Buffer 上排序，使用 d_compact_offsets
    // 输入: d_flat_ids_alt (作为初始 keys/values)
    // CUB DoubleBuffer 会自动管理 swap
    cub::DoubleBuffer<float> d_keys(d_flat_dists_alt, d_flat_dists);
    cub::DoubleBuffer<int>   d_values(d_flat_ids_alt, d_flat_ids);
    
    void *d_temp_sort = NULL; size_t temp_sort_bytes = 0;
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_sort, temp_sort_bytes, d_keys, d_values,
        compact_total_items, batch_size, d_compact_offsets, d_compact_offsets + 1, 0, sizeof(float)*8, stream_main);
    cudaMalloc(&d_temp_sort, temp_sort_bytes);
    cub::DeviceSegmentedRadixSort::SortPairs(d_temp_sort, temp_sort_bytes, d_keys, d_values,
        compact_total_items, batch_size, d_compact_offsets, d_compact_offsets + 1, 0, sizeof(float)*8, stream_main);
    cudaFree(d_temp_sort);
    
    if (events) cudaEventRecord(events->evt_sort, stream_main);

    // 8. Gather
    // 使用 d_atomic (Count) 和 d_compact_offsets
    gather_top_m_kernel<<<batch_size, 256, 0, stream_main>>>(
        d_values.Current(), d_keys.Current(), 
        d_compact_offsets, d_atomic, 
        d_top_ids, d_top_dists, d_top_counts, rerank_m
    );
    
    if (events) cudaEventRecord(events->evt_gather, stream_main);
}