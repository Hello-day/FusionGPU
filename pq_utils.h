#ifndef PQ_UTILS_H_
#define PQ_UTILS_H_

#include "defs.h"
#include <vector>
#include <cmath>
#include <limits>
#include <random>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <cstring>
#include <iostream>

// PQ 子空间训练用的快速 K-Means (单线程版本，用于外层并行)
// 特点: 随机初始化 + 少量迭代，适合 PQ 场景
inline void run_kmeans_pq_fast(const float* data, int n, int dim, int k, 
                               std::vector<float>& centroids,
                               int seed, int max_iter = 8) {
    std::mt19937 rng(seed);
    centroids.resize(k * dim);
    
    // 随机初始化: 直接随机选 k 个样本作为初始中心
    std::vector<int> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng);
    
    for (int c = 0; c < k; ++c) {
        std::memcpy(centroids.data() + c * dim, 
                   data + indices[c] * dim, dim * sizeof(float));
    }

    // K-Means 迭代 (单线程)
    std::vector<int> assign(n);
    std::vector<double> new_centroids(k * dim);
    std::vector<int> counts(k);
    
    for (int iter = 0; iter < max_iter; ++iter) { 
        // E-step: 分配
        for (int i = 0; i < n; ++i) {
            float min_d = std::numeric_limits<float>::max();
            int best_c = 0;
            for (int c = 0; c < k; ++c) {
                float d = 0;
                for (int j = 0; j < dim; ++j) {
                    float diff = data[i * dim + j] - centroids[c * dim + j];
                    d += diff * diff;
                }
                if (d < min_d) { min_d = d; best_c = c; }
            }
            assign[i] = best_c;
        }

        // M-step: 更新中心
        std::fill(new_centroids.begin(), new_centroids.end(), 0.0);
        std::fill(counts.begin(), counts.end(), 0);
        
        for (int i = 0; i < n; ++i) {
            int c = assign[i];
            counts[c]++;
            for (int j = 0; j < dim; ++j) 
                new_centroids[c * dim + j] += data[i * dim + j];
        }

        for (int c = 0; c < k; ++c) {
            if (counts[c] > 0) {
                for (int j = 0; j < dim; ++j) 
                    centroids[c * dim + j] = (float)(new_centroids[c * dim + j] / counts[c]);
            }
        }
    }
}

// 训练残差 PQ 码本 (高度优化版)
// sample_size: 用于训练的采样数量 (默认 50000)
// max_iter: K-Means 最大迭代次数 (默认 8)
inline void train_residual_pq_codebooks(
    const point_t<float>* points, 
    int n, 
    const std::vector<int>& vec_cluster_ids, 
    const std::vector<float>& clusters, 
    std::vector<float>& codebook,
    int sample_size = 50000,
    int max_iter = 8
) {
    const float* flat_data = (const float*)points;
    
    std::cout << "  Computing residuals & sampling..." << std::flush;
    
    // 1. 采样 + 计算残差 (合并操作)
    int actual_sample = std::min(sample_size, n);
    std::vector<int> sample_indices(n);
    std::iota(sample_indices.begin(), sample_indices.end(), 0);
    
    std::mt19937 rng(12345);
    std::shuffle(sample_indices.begin(), sample_indices.end(), rng);
    sample_indices.resize(actual_sample);
    
    // 采样后的残差
    std::vector<float> sampled_residuals(actual_sample * DIM);
    #pragma omp parallel for
    for(int i = 0; i < actual_sample; ++i) {
        int orig_idx = sample_indices[i];
        int cid = vec_cluster_ids[orig_idx];
        for(int j = 0; j < DIM; ++j) {
            sampled_residuals[i * DIM + j] = flat_data[orig_idx * DIM + j] - clusters[cid * DIM + j];
        }
    }
    std::cout << " done (" << actual_sample << " samples)" << std::endl;

    // 2. 并行训练 32 个子空间的 PQ 码本
    codebook.resize(PQ_M * PQ_K * PQ_SUB_DIM);
    
    std::cout << "  Training " << PQ_M << " subspaces in parallel (iter=" << max_iter << ")..." << std::flush;
    
    // 预先为每个子空间提取数据
    std::vector<std::vector<float>> sub_vecs_all(PQ_M);
    for (int m = 0; m < PQ_M; ++m) {
        sub_vecs_all[m].resize(actual_sample * PQ_SUB_DIM);
        for (int i = 0; i < actual_sample; ++i) {
            for (int d = 0; d < PQ_SUB_DIM; ++d) {
                sub_vecs_all[m][i * PQ_SUB_DIM + d] = sampled_residuals[i * DIM + m * PQ_SUB_DIM + d];
            }
        }
    }
    
    // 并行训练所有子空间 (每个子空间用单线程 K-Means)
    #pragma omp parallel for schedule(dynamic)
    for (int m = 0; m < PQ_M; ++m) {
        std::vector<float> sub_centroids;
        run_kmeans_pq_fast(sub_vecs_all[m].data(), actual_sample, PQ_SUB_DIM, PQ_K, 
                          sub_centroids, 1234 + m, max_iter);
        
        size_t offset = m * PQ_K * PQ_SUB_DIM;
        std::copy(sub_centroids.begin(), sub_centroids.end(), codebook.begin() + offset);
    }
    std::cout << " done" << std::endl;
}

// 编码残差
inline void encode_residuals_to_pq(
    const point_t<float>* points, 
    int n, 
    const std::vector<int>& vec_cluster_ids,
    const std::vector<float>& clusters,
    const std::vector<float>& codebook, 
    uint8_t* codes
) {
    const float* flat_data = (const float*)points;
    
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        int cid = vec_cluster_ids[i];
        
        for (int m = 0; m < PQ_M; ++m) {
            float min_d = std::numeric_limits<float>::max();
            uint8_t best_k = 0;
            
            int sub_start_idx = i * DIM + m * PQ_SUB_DIM;
            int cluster_start_idx = cid * DIM + m * PQ_SUB_DIM;

            for (int k = 0; k < PQ_K; ++k) {
                const float* center = codebook.data() + (m * PQ_K + k) * PQ_SUB_DIM;
                float d = 0;
                for(int j=0; j<PQ_SUB_DIM; ++j) {
                    float residual = flat_data[sub_start_idx + j] - clusters[cluster_start_idx + j];
                    float diff = residual - center[j];
                    d += diff * diff;
                }
                if (d < min_d) { min_d = d; best_k = k; }
            }
            codes[i * PQ_M + m] = best_k;
        }
    }
}

#endif // PQ_UTILS_H_