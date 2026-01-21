#include "defs.h"
#include "utils.h"
#include "pq_utils.h"
#include "kmeans_gpu.cuh"    
#include "cagra_adapter.cuh" 
#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <numeric>  // std::accumulate
#include <cmath>    // std::ceil, std::max
#include <deque>    // std::deque
#include <cuda_runtime.h>

// 定义任务结构体，用于递归队列
struct BucketTask {
    std::vector<int> indices;    // 该 bucket 包含的向量 ID
    std::vector<float> centroid; // 该 bucket 对应的中心点坐标
};

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./build_global <base.fvecs>" << std::endl;
        return 1;
    }

    // ---------------------------------------------------------
    // 1. 数据加载
    // ---------------------------------------------------------
    float* data_ptr;
    int n = read_vecs<float>(argv[1], data_ptr, DIM);
    point_t<float>* points = (point_t<float>*)data_ptr;
    std::cout << "Loaded " << n << " vectors." << std::endl;

    // ---------------------------------------------------------
    // 2. 初始 K-Means 训练
    // ---------------------------------------------------------
    std::cout << "Training " << K_CLUSTERS << " initial centroids (GPU)..." << std::endl;
    std::vector<float> initial_centroids;
    // initial_centroids 会被 resize 为 K_CLUSTERS * DIM
    run_kmeans_gpu(data_ptr, n, DIM, K_CLUSTERS, initial_centroids, 20);

    // *注意*：此时不要构建 CAGRA 索引，因为中心点即将发生剧烈变化

    // ---------------------------------------------------------
    // 3. 初始分配 (Initial Assignment)
    // ---------------------------------------------------------
    std::cout << "Assigning vectors to initial clusters..." << std::endl;
    std::vector<std::vector<int>> initial_buckets(K_CLUSTERS);
    
    #pragma omp parallel for
    for(int i=0; i<n; ++i) {
        float min_d = std::numeric_limits<float>::max();
        int best_c = 0;
        
        // 暴力搜索最近中心点 (此时 K 很小，CPU 尚可接受，或者也可以用 GPU search)
        for(int c=0; c<K_CLUSTERS; ++c) {
            float d = 0; 
            for(int j=0; j<DIM; ++j) {
                float diff = points[i].coordinates[j] - initial_centroids[c*DIM+j];
                d += diff*diff;
            }
            if(d < min_d) { min_d = d; best_c = c; }
        }
        
        #pragma omp critical
        initial_buckets[best_c].push_back(i);
    }

    // ---------------------------------------------------------
    // 4. 递归重平衡 (Recursive Rebalancing)
    // ---------------------------------------------------------
    std::cout << "--- Advanced Recursive Rebalancing (Queue-based) ---" << std::endl;

    // A. 计算基准统计
    long total_vecs_count = 0;
    for(const auto& b : initial_buckets) total_vecs_count += b.size();
    
    // 设定目标平均值
    double avg_size = (double)total_vecs_count / K_CLUSTERS;
    
    // 设定阈值：
    // 1. 大小超过平均值 2.0 倍的必须分裂
    // 2. 最小分裂粒度设为 32，防止 K-Means 对极少数据报错
    size_t split_threshold = (size_t)(avg_size * 2);
    size_t min_splittable_size = 32; 

    std::cout << "Target Avg Size: " << avg_size << ", Split Threshold: " << split_threshold << std::endl;

    // B. 初始化任务队列
    std::deque<BucketTask> queue;
    for(int c=0; c<K_CLUSTERS; ++c) {
        if(initial_buckets[c].empty()) continue; // 丢弃初始空桶
        
        std::vector<float> cent(DIM);
        for(int d=0; d<DIM; ++d) cent[d] = initial_centroids[c*DIM+d];
        queue.push_back({initial_buckets[c], cent});
    }

    // 准备接收最终结果
    std::vector<std::vector<int>> final_buckets;
    std::vector<float> final_centroids;
    int split_ops = 0; // 统计分裂操作次数

    // C. 处理队列
    while(!queue.empty()) {
        BucketTask task = queue.front();
        queue.pop_front();

        size_t curr_size = task.indices.size();

        // [终止条件]：如果小于阈值 OR 数据量太少 -> 接受为最终结果
        if (curr_size <= split_threshold || curr_size < min_splittable_size) {
            final_buckets.push_back(task.indices);
            final_centroids.insert(final_centroids.end(), task.centroid.begin(), task.centroid.end());
            continue;
        }

        // [执行分裂]：发现过载列表
        split_ops++;
        
        // 动态计算 sub_k：例如 1000个点，平均200，则尝试分成 5 份
        int sub_k = std::ceil((double)curr_size / avg_size);
        if (sub_k < 2) sub_k = 2; // 至少分2份
        // 保护：防止 sub_k 过大导致每个中心点分不到数据
        if (sub_k > (int)curr_size / 2) sub_k = (int)curr_size / 2; 

        // 提取数据
        std::vector<float> subset_data(curr_size * DIM);
        for(size_t i=0; i < curr_size; ++i) {
            const float* src = points[task.indices[i]].coordinates; 
            std::copy(src, src + DIM, subset_data.begin() + i*DIM);
        }

        // 运行局部 K-Means
        std::vector<float> sub_centroids_flat;
        // iter=15 保证局部快速收敛
        run_kmeans_gpu(subset_data.data(), curr_size, DIM, sub_k, sub_centroids_flat, 15);

        // 分配到新的 sub_k 个子桶
        std::vector<std::vector<int>> child_buckets(sub_k);
        for(size_t i=0; i < curr_size; ++i) {
            int original_vec_idx = task.indices[i];
            const float* vec = points[original_vec_idx].coordinates;
            
            float min_local_d = std::numeric_limits<float>::max();
            int best_sub_c = 0;

            for(int k=0; k<sub_k; ++k) {
                float d = 0;
                for(int d_idx=0; d_idx<DIM; ++d_idx) {
                    float diff = vec[d_idx] - sub_centroids_flat[k*DIM + d_idx];
                    d += diff * diff;
                }
                if(d < min_local_d) {
                    min_local_d = d;
                    best_sub_c = k;
                }
            }
            child_buckets[best_sub_c].push_back(original_vec_idx);
        }

        // [关键] 将所有非空子桶重新放入队列进行检查
        // 如果分裂后的某个子桶依然很大（极端偏斜），下次循环会再次被分裂
        for(int k=0; k<sub_k; ++k) {
            if (child_buckets[k].empty()) continue;

            std::vector<float> child_cent(DIM);
            for(int d=0; d<DIM; ++d) child_cent[d] = sub_centroids_flat[k*DIM + d];
            
            queue.push_back({child_buckets[k], child_cent});
        }
    }

    // D. 更新全局状态
    int final_n_clusters = final_buckets.size();
    
    // 更新 vector ID 映射 (Training PQ 需要)
    std::vector<int> vec_cluster_ids(n, -1);
    for(int c = 0; c < final_n_clusters; ++c) {
        for(int vec_idx : final_buckets[c]) {
            vec_cluster_ids[vec_idx] = c;
        }
    }

    std::cout << "Rebalancing Complete." << std::endl;
    std::cout << "Split Ops: " << split_ops << std::endl;
    std::cout << "Cluster Count: " << K_CLUSTERS << " -> " << final_n_clusters << std::endl;

    // ---------------------------------------------------------
    // 5. 构建 CAGRA 索引 (Build CAGRA)
    // ---------------------------------------------------------
    // 此时 centroids 已经稳定，且包含了所有分裂出来的中心
    std::cout << "Building CAGRA Index for " << final_n_clusters << " centroids..." << std::endl;
    float* d_centroids;
    cudaMalloc(&d_centroids, final_centroids.size() * sizeof(float));
    cudaMemcpy(d_centroids, final_centroids.data(), final_centroids.size() * sizeof(float), cudaMemcpyHostToDevice);
    
    // 使用新的聚类数量
    build_cagra_index(d_centroids, final_n_clusters, DIM, "../res/cagra_centroids.index");
    cudaFree(d_centroids);
    std::cout << "CAGRA Index built and saved." << std::endl;

    // ---------------------------------------------------------
    // 6. 统计与导出分布 (Statistics)
    // ---------------------------------------------------------
    std::cout << "Exporting distribution stats..." << std::endl;
    std::ofstream csv_file("cluster_distribution.csv");
    csv_file << "cluster_id,count\n"; 
    for(int c = 0; c < final_n_clusters; ++c) {
        csv_file << c << "," << final_buckets[c].size() << "\n";
    }
    csv_file.close();

    // ---------------------------------------------------------
    // 7. 训练 Residual PQ
    // ---------------------------------------------------------
    std::cout << "Training Residual Global PQ..." << std::endl;
    std::vector<float> global_codebook;
    // 使用更新后的 buckets, IDs, centroids
    train_residual_pq_codebooks(points, n, vec_cluster_ids, final_centroids, global_codebook);
    
    std::ofstream out_pq("../res/global_pq_codebook.bin", std::ios::binary);
    out_pq.write((char*)global_codebook.data(), global_codebook.size() * sizeof(float));
    out_pq.close();

    // ---------------------------------------------------------
    // 8. 编码残差 (Encoding)
    // ---------------------------------------------------------
    std::cout << "Encoding residuals..." << std::endl;
    std::vector<uint8_t> temp_codes((size_t)n * PQ_M);
    encode_residuals_to_pq(points, n, vec_cluster_ids, final_centroids, global_codebook, temp_codes.data());

    // ---------------------------------------------------------
    // 9. 构建与保存倒排列表 (Building & Saving IVF)
    // ---------------------------------------------------------
    std::cout << "Building Inverted Lists..." << std::endl;
    std::vector<int> offsets(final_n_clusters + 1, 0);
    std::vector<int> all_ids;
    std::vector<uint8_t> all_codes;
    all_ids.reserve(n);
    all_codes.reserve((size_t)n * PQ_M);

    for (int c = 0; c < final_n_clusters; ++c) {
        offsets[c] = all_ids.size();
        for (int vec_id : final_buckets[c]) {
            all_ids.push_back(vec_id);
            for(int m=0; m<PQ_M; ++m) 
                all_codes.push_back(temp_codes[(size_t)vec_id * PQ_M + m]);
        }
    }
    offsets[final_n_clusters] = all_ids.size();

    std::ofstream out_idx("../res/ivf_data.bin", std::ios::binary);
    int total_vecs = all_ids.size();
    
    // 写入文件头
    out_idx.write((char*)&total_vecs, sizeof(int));
    out_idx.write((char*)&final_n_clusters, sizeof(int)); // *重要*：写入新的聚类数
    out_idx.write((char*)offsets.data(), offsets.size() * sizeof(int));
    out_idx.write((char*)all_ids.data(), all_ids.size() * sizeof(int));
    out_idx.write((char*)all_codes.data(), all_codes.size() * sizeof(uint8_t));
    out_idx.write((char*)final_centroids.data(), final_centroids.size() * sizeof(float));
    out_idx.close();

    // 清理资源
    delete[] data_ptr;
    std::cout << "Build complete." << std::endl;
    return 0;
}