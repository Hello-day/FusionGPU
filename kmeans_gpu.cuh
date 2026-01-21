#ifndef KMEANS_GPU_CUH
#define KMEANS_GPU_CUH

#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <random>
#include <cfloat>

// ============== 纯 CUDA K-Means 实现 (不依赖 RAFT) ==============

// Kernel: 计算每个点到所有质心的距离，找到最近的质心
__global__ void assign_clusters_kernel(
    const float* __restrict__ data,      // [n, dim]
    const float* __restrict__ centroids, // [k, dim]
    int* __restrict__ assignments,       // [n]
    int n, int dim, int k
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    const float* point = data + idx * dim;
    float min_dist = FLT_MAX;
    int best_cluster = 0;
    
    for (int c = 0; c < k; ++c) {
        const float* centroid = centroids + c * dim;
        float dist = 0.0f;
        for (int d = 0; d < dim; ++d) {
            float diff = point[d] - centroid[d];
            dist += diff * diff;
        }
        if (dist < min_dist) {
            min_dist = dist;
            best_cluster = c;
        }
    }
    assignments[idx] = best_cluster;
}

// Kernel: 累加每个簇的点和计数
__global__ void accumulate_centroids_kernel(
    const float* __restrict__ data,      // [n, dim]
    const int* __restrict__ assignments, // [n]
    float* __restrict__ new_centroids,   // [k, dim] - 累加器
    int* __restrict__ counts,            // [k]
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

// Kernel: 计算新质心 (除以计数)
__global__ void compute_centroids_kernel(
    float* __restrict__ centroids,       // [k, dim]
    const int* __restrict__ counts,      // [k]
    int dim, int k
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= k) return;
    
    int count = counts[c];
    if (count > 0) {
        for (int d = 0; d < dim; ++d) {
            centroids[c * dim + d] /= count;
        }
    }
}

/**
 * GPU K-Means (纯 CUDA 实现，无 RAFT 依赖)
 */
inline void run_kmeans_gpu(
    const float* h_data,    // host 数据 [n, dim]
    int n, int dim, int k,
    std::vector<float>& h_centroids,  // 输出质心
    int max_iter = 20
) {
    // 1. 分配 GPU 内存
    float *d_data, *d_centroids;
    int *d_assignments, *d_counts;
    
    size_t data_bytes = (size_t)n * dim * sizeof(float);
    size_t centroid_bytes = (size_t)k * dim * sizeof(float);
    
    cudaMalloc(&d_data, data_bytes);
    cudaMalloc(&d_centroids, centroid_bytes);
    cudaMalloc(&d_assignments, n * sizeof(int));
    cudaMalloc(&d_counts, k * sizeof(int));
    
    // 2. 拷贝数据到 GPU
    cudaMemcpy(d_data, h_data, data_bytes, cudaMemcpyHostToDevice);
    
    // 3. 随机初始化质心 (K-Means++)
    std::vector<float> init_centroids(k * dim);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(0, n - 1);
    
    // 简单随机选择 k 个点作为初始质心
    std::vector<bool> chosen(n, false);
    for (int i = 0; i < k; ++i) {
        int idx;
        do { idx = dist(rng); } while (chosen[idx]);
        chosen[idx] = true;
        for (int d = 0; d < dim; ++d) {
            init_centroids[i * dim + d] = h_data[idx * dim + d];
        }
    }
    cudaMemcpy(d_centroids, init_centroids.data(), centroid_bytes, cudaMemcpyHostToDevice);
    
    // 4. 迭代
    int block_size = 256;
    int grid_size_n = (n + block_size - 1) / block_size;
    int grid_size_k = (k + block_size - 1) / block_size;
    
    std::cout << "  [CUDA] Running GPU K-Means (n=" << n << ", k=" << k << ", dim=" << dim << ")..." << std::endl;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        // 4.1 分配: 每个点找最近质心
        assign_clusters_kernel<<<grid_size_n, block_size>>>(
            d_data, d_centroids, d_assignments, n, dim, k
        );
        
        // 4.2 清零累加器
        cudaMemset(d_centroids, 0, centroid_bytes);
        cudaMemset(d_counts, 0, k * sizeof(int));
        
        // 4.3 累加
        accumulate_centroids_kernel<<<grid_size_n, block_size>>>(
            d_data, d_assignments, d_centroids, d_counts, n, dim, k
        );
        
        // 4.4 计算新质心
        compute_centroids_kernel<<<grid_size_k, block_size>>>(
            d_centroids, d_counts, dim, k
        );
        
        if ((iter + 1) % 5 == 0) {
            std::cout << "    Iteration " << (iter + 1) << "/" << max_iter << std::endl;
        }
    }
    
    // 5. 拷贝结果回 host
    h_centroids.resize(k * dim);
    cudaMemcpy(h_centroids.data(), d_centroids, centroid_bytes, cudaMemcpyDeviceToHost);
    
    // 6. 清理
    cudaFree(d_data);
    cudaFree(d_centroids);
    cudaFree(d_assignments);
    cudaFree(d_counts);
    
    std::cout << "  [CUDA] K-Means complete." << std::endl;
}

/**
 * GPU PQ Codebook Training
 */
template<int PQ_M_T, int PQ_K_T, int PQ_SUB_DIM_T>
inline void train_pq_codebooks_gpu(
    const float* flat_data,  // [n, DIM]
    int n, int full_dim,
    std::vector<float>& codebook  // output: [PQ_M, PQ_K, PQ_SUB_DIM]
) {
    codebook.resize((size_t)PQ_M_T * PQ_K_T * PQ_SUB_DIM_T);
    
    std::cout << "  [CUDA] Training PQ Codebooks (M=" << PQ_M_T 
              << ", K=" << PQ_K_T << ", sub_dim=" << PQ_SUB_DIM_T << ")..." << std::endl;
    
    for (int m = 0; m < PQ_M_T; ++m) {
        std::cout << "    Training subspace " << m << "/" << PQ_M_T << "..." << std::endl;
        
        // 提取子向量
        std::vector<float> sub_vecs((size_t)n * PQ_SUB_DIM_T);
        for (int i = 0; i < n; ++i) {
            for (int d = 0; d < PQ_SUB_DIM_T; ++d) {
                sub_vecs[(size_t)i * PQ_SUB_DIM_T + d] = flat_data[(size_t)i * full_dim + m * PQ_SUB_DIM_T + d];
            }
        }
        
        // GPU K-Means
        std::vector<float> sub_centroids;
        run_kmeans_gpu(sub_vecs.data(), n, PQ_SUB_DIM_T, PQ_K_T, sub_centroids, 15);
        
        // 拷贝到全局 codebook
        size_t offset = (size_t)m * PQ_K_T * PQ_SUB_DIM_T;
        std::copy(sub_centroids.begin(), sub_centroids.end(), codebook.begin() + offset);
    }
    
    std::cout << "  [CUDA] PQ Codebook training complete." << std::endl;
}

#endif // KMEANS_GPU_CUH