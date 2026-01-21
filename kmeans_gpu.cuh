#ifndef KMEANS_GPU_CUH
#define KMEANS_GPU_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <random>
#include <cfloat>
#include <algorithm>

// ============== 高性能 GPU K-Means (cuBLAS 加速 + 分批处理) ==============

// 分批大小：每批处理的点数 (根据显存调整)
#define KMEANS_BATCH_SIZE 500000  // 50万点 × 2048 × 4 = 4GB

// CUDA 错误检查宏
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        std::cerr << "cuBLAS Error: " << status \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// -------------- Kernel: 计算向量范数平方 --------------
__global__ void compute_norms_kernel(
    const float* __restrict__ data,
    float* __restrict__ norms,
    int n, int dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    const float* vec = data + idx * dim;
    float sum = 0.0f;
    for (int d = 0; d < dim; ++d) {
        sum += vec[d] * vec[d];
    }
    norms[idx] = sum;
}

// -------------- Kernel: 从距离矩阵找最近质心 (带偏移) --------------
__global__ void find_nearest_centroid_batched_kernel(
    const float* __restrict__ dist_matrix,  // [batch_size, k]
    const float* __restrict__ data_norms,   // [batch_size] (已偏移)
    const float* __restrict__ cent_norms,   // [k]
    int* __restrict__ assignments,          // [batch_size] (已偏移)
    int batch_size, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;
    
    float data_norm = data_norms[idx];
    const float* row = dist_matrix + idx * k;
    
    float min_dist = FLT_MAX;
    int best = 0;
    
    for (int c = 0; c < k; ++c) {
        float dist = data_norm + cent_norms[c] + row[c];
        if (dist < min_dist) {
            min_dist = dist;
            best = c;
        }
    }
    assignments[idx] = best;
}

// -------------- Kernel: 累加质心 --------------
__global__ void accumulate_centroids_kernel(
    const float* __restrict__ data,
    const int* __restrict__ assignments,
    float* __restrict__ new_centroids,
    int* __restrict__ counts,
    int n, int dim, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int cluster = assignments[idx];
    const float* point = data + idx * dim;
    
    atomicAdd(&counts[cluster], 1);
    for (int d = 0; d < dim; ++d) {
        atomicAdd(&new_centroids[cluster * dim + d], point[d]);
    }
}

// -------------- Kernel: 计算新质心 --------------
__global__ void compute_centroids_kernel(
    float* __restrict__ centroids,
    const int* __restrict__ counts,
    int dim, int k
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    
    int count = counts[c];
    if (count > 0) {
        float inv_count = 1.0f / count;
        for (int d = 0; d < dim; ++d) {
            centroids[c * dim + d] *= inv_count;
        }
    }
}

// ============== GPU 内存管理器 (分批处理版) ==============

struct KMeansGPUContext {
    float* d_data = nullptr;           // [n, dim]
    float* d_centroids = nullptr;      // [k, dim]
    int* d_assignments = nullptr;      // [n]
    int* d_counts = nullptr;           // [k]
    float* d_data_norms = nullptr;     // [n]
    float* d_cent_norms = nullptr;     // [k]
    float* d_dist_matrix = nullptr;    // [batch_size, k] - 只分配一个批次
    
    cublasHandle_t cublas_handle = nullptr;
    
    size_t allocated_n = 0;
    size_t allocated_k = 0;
    size_t allocated_dim = 0;
    size_t allocated_batch = 0;
    
    void init_cublas() {
        if (!cublas_handle) {
            CUBLAS_CHECK(cublasCreate(&cublas_handle));
        }
    }
    
    void ensure_capacity(int n, int k, int dim, int batch_size) {
        init_cublas();
        
        bool need_realloc = (n > (int)allocated_n) || (k > (int)allocated_k) || 
                           (dim > (int)allocated_dim) || (batch_size > (int)allocated_batch);
        
        if (need_realloc) {
            free_all_buffers();
            
            size_t data_bytes = (size_t)n * dim * sizeof(float);
            size_t centroid_bytes = (size_t)k * dim * sizeof(float);
            size_t dist_matrix_bytes = (size_t)batch_size * k * sizeof(float);
            
            std::cout << "    Allocating GPU memory: data=" << (data_bytes/1024/1024) << "MB, "
                      << "dist_matrix=" << (dist_matrix_bytes/1024/1024) << "MB" << std::endl;
            
            CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
            CUDA_CHECK(cudaMalloc(&d_centroids, centroid_bytes));
            CUDA_CHECK(cudaMalloc(&d_assignments, n * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_counts, k * sizeof(int)));
            CUDA_CHECK(cudaMalloc(&d_data_norms, n * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_cent_norms, k * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_dist_matrix, dist_matrix_bytes));
            
            allocated_n = n;
            allocated_k = k;
            allocated_dim = dim;
            allocated_batch = batch_size;
        }
    }
    
    void free_all_buffers() {
        if (d_data) { cudaFree(d_data); d_data = nullptr; }
        if (d_centroids) { cudaFree(d_centroids); d_centroids = nullptr; }
        if (d_assignments) { cudaFree(d_assignments); d_assignments = nullptr; }
        if (d_counts) { cudaFree(d_counts); d_counts = nullptr; }
        if (d_data_norms) { cudaFree(d_data_norms); d_data_norms = nullptr; }
        if (d_cent_norms) { cudaFree(d_cent_norms); d_cent_norms = nullptr; }
        if (d_dist_matrix) { cudaFree(d_dist_matrix); d_dist_matrix = nullptr; }
    }
    
    void free_all() {
        free_all_buffers();
        if (cublas_handle) { cublasDestroy(cublas_handle); cublas_handle = nullptr; }
        allocated_n = allocated_k = allocated_dim = allocated_batch = 0;
    }
    
    ~KMeansGPUContext() { free_all(); }
};

static KMeansGPUContext g_kmeans_ctx;

/**
 * 分批计算 assignments
 */
inline void batch_assign_clusters(
    KMeansGPUContext& ctx,
    int n, int k, int dim, int batch_size
) {
    int block_size = 256;
    float alpha = -2.0f, beta = 0.0f;
    
    for (int start = 0; start < n; start += batch_size) {
        int curr_batch = std::min(batch_size, n - start);
        int grid_batch = (curr_batch + block_size - 1) / block_size;
        
        // cuBLAS SGEMM: dist_matrix[batch, k] = -2 * data[batch, dim] @ centroids[k, dim]^T
        CUBLAS_CHECK(cublasSgemm(
            ctx.cublas_handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            k, curr_batch, dim,
            &alpha,
            ctx.d_centroids, dim,
            ctx.d_data + start * dim, dim,
            &beta,
            ctx.d_dist_matrix, k
        ));
        
        // 找最近质心
        find_nearest_centroid_batched_kernel<<<grid_batch, block_size>>>(
            ctx.d_dist_matrix,
            ctx.d_data_norms + start,
            ctx.d_cent_norms,
            ctx.d_assignments + start,
            curr_batch, k
        );
    }
}

/**
 * 高性能 GPU K-Means (cuBLAS 加速 + 分批处理)
 */
inline void run_kmeans_gpu(
    const float* h_data,
    int n, int dim, int k,
    std::vector<float>& h_centroids,
    int max_iter = 20
) {
    // 动态计算批大小 (确保距离矩阵不超过 2GB)
    size_t max_dist_matrix_bytes = 2ULL * 1024 * 1024 * 1024;  // 2GB
    int batch_size = std::min(n, (int)(max_dist_matrix_bytes / (k * sizeof(float))));
    batch_size = std::max(batch_size, 10000);  // 至少 10000
    
    std::cout << "  [CUDA] Running GPU K-Means with cuBLAS (n=" << n << ", k=" << k 
              << ", dim=" << dim << ", batch=" << batch_size << ")..." << std::endl;
    
    // 1. 初始化
    g_kmeans_ctx.ensure_capacity(n, k, dim, batch_size);
    
    size_t data_bytes = (size_t)n * dim * sizeof(float);
    size_t centroid_bytes = (size_t)k * dim * sizeof(float);
    
    std::cout << "    Copying data to GPU (" << (data_bytes / 1024 / 1024) << " MB)..." << std::endl;
    CUDA_CHECK(cudaMemcpy(g_kmeans_ctx.d_data, h_data, data_bytes, cudaMemcpyHostToDevice));
    
    // 2. 初始化质心
    std::cout << "    Initializing centroids..." << std::endl;
    std::vector<float> init_centroids(k * dim);
    std::mt19937 rng(42);
    int stride = std::max(1, n / k);
    for (int i = 0; i < k; ++i) {
        int idx = ((i * stride) + (rng() % stride)) % n;
        for (int d = 0; d < dim; ++d) {
            init_centroids[i * dim + d] = h_data[idx * dim + d];
        }
    }
    CUDA_CHECK(cudaMemcpy(g_kmeans_ctx.d_centroids, init_centroids.data(), centroid_bytes, cudaMemcpyHostToDevice));
    
    // 3. 预计算数据范数
    int block_size = 256;
    int grid_n = (n + block_size - 1) / block_size;
    int grid_k = (k + block_size - 1) / block_size;
    
    compute_norms_kernel<<<grid_n, block_size>>>(
        g_kmeans_ctx.d_data, g_kmeans_ctx.d_data_norms, n, dim
    );
    
    std::cout << "    Starting iterations..." << std::endl;
    
    // 4. 迭代
    for (int iter = 0; iter < max_iter; ++iter) {
        // 4.1 计算质心范数
        compute_norms_kernel<<<grid_k, block_size>>>(
            g_kmeans_ctx.d_centroids, g_kmeans_ctx.d_cent_norms, k, dim
        );
        
        // 4.2 分批计算 assignments
        batch_assign_clusters(g_kmeans_ctx, n, k, dim, batch_size);
        CUDA_CHECK(cudaGetLastError());
        
        // 4.3 清零累加器
        CUDA_CHECK(cudaMemset(g_kmeans_ctx.d_centroids, 0, centroid_bytes));
        CUDA_CHECK(cudaMemset(g_kmeans_ctx.d_counts, 0, k * sizeof(int)));
        
        // 4.4 累加
        accumulate_centroids_kernel<<<grid_n, block_size>>>(
            g_kmeans_ctx.d_data,
            g_kmeans_ctx.d_assignments,
            g_kmeans_ctx.d_centroids,
            g_kmeans_ctx.d_counts,
            n, dim, k
        );
        
        // 4.5 计算新质心
        compute_centroids_kernel<<<grid_k, block_size>>>(
            g_kmeans_ctx.d_centroids,
            g_kmeans_ctx.d_counts,
            dim, k
        );
        
        if ((iter + 1) % 5 == 0) {
            CUDA_CHECK(cudaDeviceSynchronize());
            std::cout << "    Iteration " << (iter + 1) << "/" << max_iter << std::endl;
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    
    h_centroids.resize(k * dim);
    CUDA_CHECK(cudaMemcpy(h_centroids.data(), g_kmeans_ctx.d_centroids, centroid_bytes, cudaMemcpyDeviceToHost));
    
    std::cout << "  [CUDA] K-Means complete." << std::endl;
}

/**
 * GPU 初始分配
 */
inline void gpu_assign_initial_clusters(
    const float* h_data,           
    const float* h_centroids,      
    int n, int dim, int k,
    std::vector<std::vector<int>>& buckets  
) {
    size_t max_dist_matrix_bytes = 2ULL * 1024 * 1024 * 1024;
    int batch_size = std::min(n, (int)(max_dist_matrix_bytes / (k * sizeof(float))));
    batch_size = std::max(batch_size, 10000);
    
    g_kmeans_ctx.ensure_capacity(n, k, dim, batch_size);
    
    CUDA_CHECK(cudaMemcpy(g_kmeans_ctx.d_data, h_data, (size_t)n * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_kmeans_ctx.d_centroids, h_centroids, (size_t)k * dim * sizeof(float), cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_n = (n + block_size - 1) / block_size;
    int grid_k = (k + block_size - 1) / block_size;
    
    compute_norms_kernel<<<grid_n, block_size>>>(g_kmeans_ctx.d_data, g_kmeans_ctx.d_data_norms, n, dim);
    compute_norms_kernel<<<grid_k, block_size>>>(g_kmeans_ctx.d_centroids, g_kmeans_ctx.d_cent_norms, k, dim);
    
    batch_assign_clusters(g_kmeans_ctx, n, k, dim, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    std::vector<int> h_assignments(n);
    CUDA_CHECK(cudaMemcpy(h_assignments.data(), g_kmeans_ctx.d_assignments, n * sizeof(int), cudaMemcpyDeviceToHost));
    
    buckets.clear();
    buckets.resize(k);
    for (int i = 0; i < n; ++i) {
        buckets[h_assignments[i]].push_back(i);
    }
}

/**
 * GPU 分配并返回 assignments 数组
 */
inline void gpu_assign_to_centroids(
    const float* h_data,           
    const float* h_centroids,      
    int n, int dim, int k,
    std::vector<int>& assignments
) {
    size_t max_dist_matrix_bytes = 2ULL * 1024 * 1024 * 1024;
    int batch_size = std::min(n, (int)(max_dist_matrix_bytes / (k * sizeof(float))));
    batch_size = std::max(batch_size, 10000);
    
    g_kmeans_ctx.ensure_capacity(n, k, dim, batch_size);
    
    CUDA_CHECK(cudaMemcpy(g_kmeans_ctx.d_data, h_data, (size_t)n * dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_kmeans_ctx.d_centroids, h_centroids, (size_t)k * dim * sizeof(float), cudaMemcpyHostToDevice));
    
    int block_size = 256;
    int grid_n = (n + block_size - 1) / block_size;
    int grid_k = (k + block_size - 1) / block_size;
    
    compute_norms_kernel<<<grid_n, block_size>>>(g_kmeans_ctx.d_data, g_kmeans_ctx.d_data_norms, n, dim);
    compute_norms_kernel<<<grid_k, block_size>>>(g_kmeans_ctx.d_centroids, g_kmeans_ctx.d_cent_norms, k, dim);
    
    batch_assign_clusters(g_kmeans_ctx, n, k, dim, batch_size);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    assignments.resize(n);
    CUDA_CHECK(cudaMemcpy(assignments.data(), g_kmeans_ctx.d_assignments, n * sizeof(int), cudaMemcpyDeviceToHost));
}

inline void cleanup_kmeans_gpu() {
    g_kmeans_ctx.free_all();
}

// ============== GPU PQ Encoding (高性能 PQ 编码) ==============

// Kernel: 计算残差并编码 (每个线程处理一个向量的一个子空间)
__global__ void encode_residuals_kernel(
    const float* __restrict__ data,           // [n, dim]
    const float* __restrict__ centroids,      // [n_clusters, dim]
    const float* __restrict__ codebook,       // [PQ_M, PQ_K, sub_dim]
    const int* __restrict__ cluster_ids,      // [n]
    uint8_t* __restrict__ codes,              // [n, PQ_M]
    int n, int dim, int pq_m, int pq_k, int sub_dim
) {
    int vec_idx = blockIdx.x;
    int sub_idx = threadIdx.x;  // 子空间索引
    
    if (vec_idx >= n || sub_idx >= pq_m) return;
    
    int cid = cluster_ids[vec_idx];
    
    // 计算该子空间的起始位置
    int data_offset = vec_idx * dim + sub_idx * sub_dim;
    int cent_offset = cid * dim + sub_idx * sub_dim;
    int codebook_offset = sub_idx * pq_k * sub_dim;
    
    // 找到最近的码本条目
    float min_dist = FLT_MAX;
    uint8_t best_k = 0;
    
    for (int k = 0; k < pq_k; ++k) {
        float dist = 0.0f;
        const float* center = codebook + codebook_offset + k * sub_dim;
        
        #pragma unroll
        for (int d = 0; d < sub_dim; ++d) {
            float residual = data[data_offset + d] - centroids[cent_offset + d];
            float diff = residual - center[d];
            dist += diff * diff;
        }
        
        if (dist < min_dist) {
            min_dist = dist;
            best_k = (uint8_t)k;
        }
    }
    
    codes[vec_idx * pq_m + sub_idx] = best_k;
}

/**
 * GPU 加速的残差 PQ 编码
 */
inline void encode_residuals_to_pq_gpu(
    const float* h_data,                      // [n, dim] host 数据
    int n, int dim,
    const std::vector<int>& h_cluster_ids,    // [n] 每个向量的聚类 ID
    const std::vector<float>& h_centroids,    // [n_clusters, dim] 聚类中心
    const std::vector<float>& h_codebook,     // [PQ_M, PQ_K, sub_dim] PQ 码本
    uint8_t* h_codes,                         // [n, PQ_M] 输出编码
    int pq_m, int pq_k, int sub_dim
) {
    int n_clusters = h_centroids.size() / dim;
    
    std::cout << "  [CUDA] Encoding " << n << " vectors (GPU)..." << std::endl;
    
    // 分配 GPU 内存
    float* d_data;
    float* d_centroids;
    float* d_codebook;
    int* d_cluster_ids;
    uint8_t* d_codes;
    
    size_t data_bytes = (size_t)n * dim * sizeof(float);
    size_t centroid_bytes = h_centroids.size() * sizeof(float);
    size_t codebook_bytes = h_codebook.size() * sizeof(float);
    size_t ids_bytes = n * sizeof(int);
    size_t codes_bytes = (size_t)n * pq_m * sizeof(uint8_t);
    
    CUDA_CHECK(cudaMalloc(&d_data, data_bytes));
    CUDA_CHECK(cudaMalloc(&d_centroids, centroid_bytes));
    CUDA_CHECK(cudaMalloc(&d_codebook, codebook_bytes));
    CUDA_CHECK(cudaMalloc(&d_cluster_ids, ids_bytes));
    CUDA_CHECK(cudaMalloc(&d_codes, codes_bytes));
    
    // 拷贝数据到 GPU
    CUDA_CHECK(cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), centroid_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_codebook, h_codebook.data(), codebook_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cluster_ids, h_cluster_ids.data(), ids_bytes, cudaMemcpyHostToDevice));
    
    // 启动 kernel: 每个 block 处理一个向量，每个线程处理一个子空间
    // Grid: n 个 blocks, Block: pq_m 个线程
    encode_residuals_kernel<<<n, pq_m>>>(
        d_data, d_centroids, d_codebook, d_cluster_ids, d_codes,
        n, dim, pq_m, pq_k, sub_dim
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 拷贝结果回 CPU
    CUDA_CHECK(cudaMemcpy(h_codes, d_codes, codes_bytes, cudaMemcpyDeviceToHost));
    
    // 清理
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_codebook);
    cudaFree(d_cluster_ids);
    cudaFree(d_codes);
    
    std::cout << "  [CUDA] Encoding complete." << std::endl;
}

/**
 * GPU PQ Codebook Training
 */
template<int PQ_M_T, int PQ_K_T, int PQ_SUB_DIM_T>
inline void train_pq_codebooks_gpu(
    const float* flat_data,
    int n, int full_dim,
    std::vector<float>& codebook
) {
    codebook.resize((size_t)PQ_M_T * PQ_K_T * PQ_SUB_DIM_T);
    
    std::cout << "  [CUDA] Training PQ Codebooks (M=" << PQ_M_T 
              << ", K=" << PQ_K_T << ", sub_dim=" << PQ_SUB_DIM_T << ")..." << std::endl;
    
    for (int m = 0; m < PQ_M_T; ++m) {
        std::vector<float> sub_vecs((size_t)n * PQ_SUB_DIM_T);
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < PQ_SUB_DIM_T; ++d) {
                sub_vecs[(size_t)i * PQ_SUB_DIM_T + d] = flat_data[(size_t)i * full_dim + m * PQ_SUB_DIM_T + d];
            }
        }
        
        std::vector<float> sub_centroids;
        run_kmeans_gpu(sub_vecs.data(), n, PQ_SUB_DIM_T, PQ_K_T, sub_centroids, 15);
        
        size_t offset = (size_t)m * PQ_K_T * PQ_SUB_DIM_T;
        std::copy(sub_centroids.begin(), sub_centroids.end(), codebook.begin() + offset);
    }
    
    std::cout << "  [CUDA] PQ Codebook training complete." << std::endl;
}

#endif // KMEANS_GPU_CUH
