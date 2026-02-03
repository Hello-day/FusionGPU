#include "defs.h"
#include "utils.h"
#include "cagra_adapter.cuh"
#include <cuda_runtime.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <atomic>
#include <memory>
#include <optional>
#include <unordered_set>
#include <string>
#include <numeric>

extern void run_gpu_batch_logic(
    const IVFIndexGPU& idx, float* dq, uint32_t* dc, 
    float* dd, float* dt, 
    int*& did, float*& ddist, int*& did_alt, float*& ddist_alt, 
    size_t& cap, int* cnt, int* off, int* atm,
    int* top_id, float* top_dist, int* top_cnt,
    int bs, int tm, int rm, int final_k, float pr,
    float* d_list_cutoffs, float* d_thresholds, 
    int* d_compact_offsets,
    cudaStream_t s, BatchLogicEvents* events,
    int p1_lists, int limit_k, float threshold_coeff
);

extern void run_gpu_rerank(float* d_queries, int* d_top_ids, int* d_top_counts, float* d_base_vecs, int* d_final_ids, float* d_final_dists, int* d_stable_cnt, bool* d_finished, int batch_size, int rerank_m, int final_k, int total_vecs, int mini_batch_size, float epsilon, int beta, cudaStream_t stream);

struct PipelineTimings {
    std::atomic<double> h2d_ms{0};
    std::atomic<double> cagra_ms{0};
    std::atomic<double> gpu_batch_ms{0};
    std::atomic<double> gpu_rerank_ms{0};
    std::atomic<double> d2h_ms{0};
    std::atomic<double> e2e_ms{0};           
    std::atomic<double> first_batch_e2e{0};  
    std::atomic<double> cpu_wait_ms{0};
    std::atomic<int> batch_count{0};
};

class SearchPipeline {
    struct BatchCtx {
        int id;
        cudaStream_t stream;
        cudaEvent_t cpu_wait_evt;
        cudaEvent_t evt_h2d_start, evt_h2d_end, evt_cagra_start, evt_cagra_end;
        cudaEvent_t evt_gpu_batch_start, evt_gpu_batch_end, evt_rerank_start, evt_rerank_end;
        cudaEvent_t evt_d2h_start, evt_d2h_end;
        
        BatchLogicEvents logic_events;
        std::mutex state_mtx;
        std::condition_variable state_cv;
        bool is_busy = false;

        float* d_queries = nullptr;       
        uint32_t* d_cagra_res = nullptr;
        float* d_cagra_dists = nullptr; 
        float* d_global_tables = nullptr; 
        int* d_counts = nullptr;          
        int* d_offsets = nullptr;         
        int* d_atomic = nullptr;          
        int* d_flat_ids = nullptr;        
        float* d_flat_dists = nullptr;
        int* d_flat_ids_alt = nullptr;    
        float* d_flat_dists_alt = nullptr;
        size_t gpu_pool_cap = 0;    
        int *d_top_ids = nullptr, *d_top_counts = nullptr;
        float *d_top_dists = nullptr;
        int *d_final_ids = nullptr;
        float *d_final_dists = nullptr;
        int *h_final_ids = nullptr;
        int* d_stable_cnt = nullptr;
        bool* d_finished = nullptr;
        
        float* d_list_cutoffs = nullptr; 
        float* d_thresholds = nullptr;
        int* d_compact_offsets = nullptr;

        int* h_atomic_reader = nullptr; 
        int* h_total_max_reader = nullptr; 

        int current_batch_size = 0;
        bool is_serial_batch = false;  

        BatchCtx() {
            cudaEventCreate(&logic_events.evt_start);
            cudaEventCreate(&logic_events.evt_prune_count);
            cudaEventCreate(&logic_events.evt_precompute);
            cudaEventCreate(&logic_events.evt_scan_offset);
            cudaEventCreate(&logic_events.evt_resize_sync);
            cudaEventCreate(&logic_events.evt_scan_phase1_end);
            cudaEventCreate(&logic_events.evt_scan_cand);
            cudaEventCreate(&logic_events.evt_compact);
            cudaEventCreate(&logic_events.evt_sort);
            cudaEventCreate(&logic_events.evt_gather);
        }

        ~BatchCtx() {
            cudaEventDestroy(logic_events.evt_start);
            cudaEventDestroy(logic_events.evt_prune_count);
            cudaEventDestroy(logic_events.evt_precompute);
            cudaEventDestroy(logic_events.evt_scan_offset);
            cudaEventDestroy(logic_events.evt_resize_sync);
            cudaEventDestroy(logic_events.evt_scan_phase1_end);
            cudaEventDestroy(logic_events.evt_scan_cand);
            cudaEventDestroy(logic_events.evt_compact);
            cudaEventDestroy(logic_events.evt_sort);
            cudaEventDestroy(logic_events.evt_gather);
        }
    };

    int batch_size, top_m, rerank_m, final_k;
    int mini_batch_size; float epsilon; int beta;
    float prune_ratio;
    int p1_lists;           // Phase 1 密集搜索的聚类数量
    int limit_k;            // 每个聚类内保留的候选数量
    float threshold_coeff;  // Phase 2 阈值放宽系数
    IVFIndexGPU ivf_index;
    float* d_base_vecs = nullptr;
    int total_base_vecs;
    raft::device_resources raft_handle;
    std::optional<cuvs::neighbors::cagra::index<float, uint32_t>> cagra_idx_opt;
    
    std::vector<BatchCtx*> buffers;
    int num_streams;          
    int next_buffer_idx = 0;
    cudaStream_t init_stream;  // 用于初始化阶段的异步拷贝  

    std::thread worker_thread;
    std::queue<BatchCtx*> work_queue;
    std::mutex mtx;
    std::condition_variable cv;
    bool running = true;
    PipelineTimings pipeline_timings;
    
    struct ExtendedStats : public GPUStageTimings {
        size_t total_max_candidates = 0; 
    } global_gpu_accum_ext;

    std::vector<std::vector<int>> final_results;
    int submit_count = 0;  

public:
    SearchPipeline(int b_size, int tm, int rm, IVFIndexGPU index, const float* raw_vecs, const char* cagra_path, 
                   int total_queries, int total_base, int mb, float eps, int b, float pr, int n_streams = 2, int k_val = 10,
                   int p1 = DEFAULT_P1_LISTS, int lk = DEFAULT_LIMIT_K, float tc = DEFAULT_THRESHOLD_COEFF) 
        : batch_size(b_size), top_m(tm), rerank_m(rm), mini_batch_size(mb), epsilon(eps), beta(b), prune_ratio(pr),
          ivf_index(index), total_base_vecs(total_base), num_streams(n_streams), final_k(k_val),
          p1_lists(p1), limit_k(lk), threshold_coeff(tc)
    {
        cudaStreamCreate(&init_stream);
        cagra_idx_opt.emplace(load_cagra_index(raft_handle, cagra_path));
        final_results.resize(total_queries);
        cudaMalloc(&d_base_vecs, (size_t)total_base * DIM * sizeof(float));
        // 使用异步拷贝，初始化阶段完成后同步
        cudaMemcpyAsync(d_base_vecs, raw_vecs, (size_t)total_base * DIM * sizeof(float), cudaMemcpyHostToDevice, init_stream);
        // 注意：ivf_index已在外部加载，这里不需要重新加载
        size_t init_cap = (size_t)batch_size * 2000;

        for(int i=0; i<num_streams; ++i) {
            BatchCtx* ctx = new BatchCtx();
            ctx->id = i;
            cudaStreamCreate(&ctx->stream);
            cudaEventCreate(&ctx->cpu_wait_evt);
            cudaEventCreate(&ctx->evt_h2d_start); cudaEventCreate(&ctx->evt_h2d_end);
            cudaEventCreate(&ctx->evt_cagra_start); cudaEventCreate(&ctx->evt_cagra_end);
            cudaEventCreate(&ctx->evt_gpu_batch_start); cudaEventCreate(&ctx->evt_gpu_batch_end);
            cudaEventCreate(&ctx->evt_rerank_start); cudaEventCreate(&ctx->evt_rerank_end);
            cudaEventCreate(&ctx->evt_d2h_start); cudaEventCreate(&ctx->evt_d2h_end);
            
            cudaMalloc(&ctx->d_queries, batch_size * DIM * sizeof(float));
            cudaMalloc(&ctx->d_cagra_res, batch_size * top_m * sizeof(uint32_t));
            cudaMalloc(&ctx->d_cagra_dists, batch_size * top_m * sizeof(float)); 

            size_t table_size = (size_t)batch_size * top_m * PQ_M * PQ_K * sizeof(float);
            cudaMalloc(&ctx->d_global_tables, table_size);
            
            cudaMalloc(&ctx->d_counts, batch_size * sizeof(int));
            cudaMalloc(&ctx->d_offsets, (batch_size + 1) * sizeof(int));
            cudaMalloc(&ctx->d_atomic, batch_size * sizeof(int));
            
            ctx->gpu_pool_cap = init_cap;
            cudaMalloc(&ctx->d_flat_ids, init_cap * sizeof(int));
            cudaMalloc(&ctx->d_flat_dists, init_cap * sizeof(float));
            cudaMalloc(&ctx->d_flat_ids_alt, init_cap * sizeof(int));
            cudaMalloc(&ctx->d_flat_dists_alt, init_cap * sizeof(float));
            cudaMalloc(&ctx->d_top_ids, batch_size * rerank_m * sizeof(int));
            cudaMalloc(&ctx->d_top_dists, batch_size * rerank_m * sizeof(float));
            cudaMalloc(&ctx->d_top_counts, batch_size * sizeof(int));
            cudaMalloc(&ctx->d_final_ids, batch_size * final_k * sizeof(int));
            cudaMalloc(&ctx->d_final_dists, batch_size * final_k * sizeof(float));
            cudaMallocHost(&ctx->h_final_ids, batch_size * final_k * sizeof(int));
            
            cudaMallocHost(&ctx->h_atomic_reader, batch_size * sizeof(int));
            cudaMallocHost(&ctx->h_total_max_reader, sizeof(int));
            
            cudaMalloc(&ctx->d_stable_cnt, batch_size * sizeof(int));
            cudaMalloc(&ctx->d_finished, batch_size * sizeof(bool));
            
            // 使用 p1_lists 参数分配内存
            cudaMalloc(&ctx->d_list_cutoffs, batch_size * p1_lists * sizeof(float));
            cudaMalloc(&ctx->d_thresholds, batch_size * sizeof(float));
            cudaMalloc(&ctx->d_compact_offsets, (batch_size + 1) * sizeof(int));
            
            buffers.push_back(ctx); 
        }
        // 等待初始化流完成
        cudaStreamSynchronize(init_stream);
        worker_thread = std::thread(&SearchPipeline::result_collection_loop, this);
    }

    ~SearchPipeline() {
        { std::lock_guard<std::mutex> lk(mtx); running = false; }
        cv.notify_all();
        if(worker_thread.joinable()) worker_thread.join();
        raft::resource::sync_stream(raft_handle);
        cudaStreamDestroy(init_stream);
        if (d_base_vecs) cudaFree(d_base_vecs);
        
        for(BatchCtx* ctx : buffers) {
            cudaStreamDestroy(ctx->stream); cudaEventDestroy(ctx->cpu_wait_evt);
            cudaFree(ctx->d_queries); cudaFree(ctx->d_cagra_res); cudaFree(ctx->d_cagra_dists);
            cudaFree(ctx->d_global_tables); cudaFree(ctx->d_counts); cudaFree(ctx->d_offsets);
            cudaFree(ctx->d_atomic); cudaFree(ctx->d_flat_ids); cudaFree(ctx->d_flat_dists);
            cudaFree(ctx->d_flat_ids_alt); cudaFree(ctx->d_flat_dists_alt);
            cudaFree(ctx->d_top_ids); cudaFree(ctx->d_top_dists); cudaFree(ctx->d_top_counts);
            cudaFree(ctx->d_final_ids); cudaFree(ctx->d_final_dists); cudaFreeHost(ctx->h_final_ids);
            
            cudaFreeHost(ctx->h_atomic_reader);
            cudaFreeHost(ctx->h_total_max_reader);
            
            cudaFree(ctx->d_stable_cnt); cudaFree(ctx->d_finished);
            cudaFree(ctx->d_list_cutoffs); cudaFree(ctx->d_thresholds);
            cudaFree(ctx->d_compact_offsets);
            delete ctx;
        }
    }
    
    void reset_pipeline_timings() {
        pipeline_timings.h2d_ms.store(0.0);
        pipeline_timings.cagra_ms.store(0.0);
        pipeline_timings.gpu_batch_ms.store(0.0);
        pipeline_timings.gpu_rerank_ms.store(0.0);
        pipeline_timings.d2h_ms.store(0.0);
        pipeline_timings.e2e_ms.store(0.0);
        pipeline_timings.first_batch_e2e.store(0.0);
        pipeline_timings.cpu_wait_ms.store(0.0);
        pipeline_timings.batch_count.store(0);
        global_gpu_accum_ext = ExtendedStats(); 
    }

    void print_timing_stats(double total_time_ms) {
        int batch_cnt = pipeline_timings.batch_count.load();
        if (batch_cnt == 0) return;
        double avg_h2d = pipeline_timings.h2d_ms.load() / batch_cnt;
        double avg_cagra = pipeline_timings.cagra_ms.load() / batch_cnt;
        double avg_gpu_batch = pipeline_timings.gpu_batch_ms.load() / batch_cnt;
        double avg_rerank = pipeline_timings.gpu_rerank_ms.load() / batch_cnt;
        double avg_d2h = pipeline_timings.d2h_ms.load() / batch_cnt;
        double total_sum = avg_h2d + avg_cagra + avg_gpu_batch + avg_rerank + avg_d2h;
        double avg_e2e = pipeline_timings.e2e_ms.load() / batch_cnt;
        double avg_cpu_wait = pipeline_timings.cpu_wait_ms.load() / batch_cnt;
        
        std::cout << "\n============= Pipeline Timing Breakdown =============\n";
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "  H2D:          " << std::setw(8) << avg_h2d << " ms\n";
        std::cout << "  CAGRA:        " << std::setw(8) << avg_cagra << " ms\n";
        std::cout << "  Coarse (PQ):  " << std::setw(8) << avg_gpu_batch << " ms\n";
        std::cout << "  Rerank (GPU): " << std::setw(8) << avg_rerank << " ms\n";
        std::cout << "  D2H:          " << std::setw(8) << avg_d2h << " ms\n";
        std::cout << "  -----------------------------------------\n";
        std::cout << "  Sum of stages:" << std::setw(8) << total_sum << " ms\n";
        std::cout << "  E2E/Batch(avg):" << std::setw(7) << avg_e2e << " ms\n";
        std::cout << "  CPU wait/Batch:" << std::setw(7) << avg_cpu_wait << " ms\n";
        std::cout << "  batch count:  " << batch_cnt << "\n";
        std::cout << "=====================================================\n";
        
        std::cout << "\n============= GPU Batch Logic Fine-grained Breakdown (Avg/Batch) =============\n";
        std::cout << "  0. Prune+Count (Fused):     " << std::setw(8) << global_gpu_accum_ext.prune_count_ms / batch_cnt << " ms\n";
        std::cout << "  1. Scan Offsets (Prefix):   " << std::setw(8) << global_gpu_accum_ext.scan_offset_ms / batch_cnt << " ms\n";
        std::cout << "  2. Precompute (Residual):   " << std::setw(8) << global_gpu_accum_ext.precompute_ms / batch_cnt << " ms  [*]\n";
        std::cout << "  3. Resize Check & Sync:     " << std::setw(8) << global_gpu_accum_ext.resize_sync_ms / batch_cnt << " ms\n";
        std::cout << "     [*] Precompute与D2H传输并行执行\n";
        std::cout << "  4a. Scan Phase 1 (Top-8):   " << std::setw(8) << global_gpu_accum_ext.scan_phase1_ms / batch_cnt << " ms\n";
        std::cout << "  4b. Scan Phase 2 (Filter):  " << std::setw(8) << global_gpu_accum_ext.scan_phase2_ms / batch_cnt << " ms\n";
        std::cout << "  5. Compaction (Scan+Move):  " << std::setw(8) << global_gpu_accum_ext.compact_ms / batch_cnt << " ms\n";
        std::cout << "  6. Sort (Compact Radix):    " << std::setw(8) << global_gpu_accum_ext.sort_ms / batch_cnt << " ms\n";
        std::cout << "  7. Gather (Top-K):          " << std::setw(8) << global_gpu_accum_ext.gather_ms / batch_cnt << " ms\n";
        std::cout << "----------------------------------------------------------------\n";
        std::cout << "  Total GPU Batch Kernel:     " << std::setw(8) << global_gpu_accum_ext.total_ms / batch_cnt << " ms\n";
        
        size_t avg_max_cand = global_gpu_accum_ext.total_max_candidates / batch_cnt;
        size_t avg_real_cand = global_gpu_accum_ext.total_candidates / batch_cnt;
        
        std::cout << "  [Allocated] Max Potential Candidates: " << avg_max_cand << "\n";
        std::cout << "  [Actual]    Filtered Real Candidates: " << avg_real_cand << " (Sorted Size)\n";
        double reduction_rate = 100.0 * (1.0 - (double)avg_real_cand / (double)avg_max_cand);
        std::cout << "  >> Pruning Reduction Rate: " << std::setprecision(1) << reduction_rate << " %\n";
        std::cout << "============================================================================\n";
    }
    const std::vector<std::vector<int>>& get_results() const { return final_results; }

    void submit_batch(const std::vector<float>& queries, int query_offset) {
        BatchCtx* c = buffers[next_buffer_idx];
        
        {
            std::unique_lock<std::mutex> lk(c->state_mtx);
            c->state_cv.wait(lk, [&]{ return !c->is_busy; });
            c->is_busy = true; 
        }
        
        // 快速检查事件状态（非阻塞），如果未就绪则等待（实际等待时间由工作线程统计）
        // 注意：cudaEventQuery 几乎不阻塞，真正的等待在 condition_variable 中
        cudaError_t event_status = cudaEventQuery(c->cpu_wait_evt);
        if (event_status == cudaErrorNotReady) {
            // 事件未就绪，说明前一个批次还在处理，等待时间已在 condition_variable 中统计
            // 这里不需要额外统计，因为 cudaEventQuery 本身不阻塞
        }
        
        c->is_serial_batch = (submit_count == 0);
        submit_count++;
        c->current_batch_size = queries.size() / DIM;
        c->id = query_offset;

        // H2D
        cudaEventRecord(c->evt_h2d_start, c->stream);
        cudaMemcpyAsync(c->d_queries, queries.data(), queries.size() * sizeof(float), cudaMemcpyHostToDevice, c->stream);
        cudaEventRecord(c->evt_h2d_end, c->stream);
        
        // CAGRA
        cudaEventRecord(c->evt_cagra_start, c->stream);
        raft::resource::set_cuda_stream(raft_handle, c->stream);
        search_cagra_index(raft_handle, *cagra_idx_opt, c->d_queries, c->current_batch_size, DIM, top_m, c->d_cagra_res, c->d_cagra_dists);
        cudaEventRecord(c->evt_cagra_end, c->stream);
        
        // GPU Batch
        cudaEventRecord(c->evt_gpu_batch_start, c->stream);
        run_gpu_batch_logic(
            ivf_index, c->d_queries, c->d_cagra_res, c->d_cagra_dists, c->d_global_tables, 
            c->d_flat_ids, c->d_flat_dists, c->d_flat_ids_alt, c->d_flat_dists_alt, 
            c->gpu_pool_cap, c->d_counts, c->d_offsets, c->d_atomic,
            c->d_top_ids, c->d_top_dists, c->d_top_counts, 
            c->current_batch_size, top_m, rerank_m, final_k, prune_ratio,
            c->d_list_cutoffs, c->d_thresholds, c->d_compact_offsets,
            c->stream, &c->logic_events,
            p1_lists, limit_k, threshold_coeff
        );
        cudaEventRecord(c->evt_gpu_batch_end, c->stream);

        // Rerank
        cudaEventRecord(c->evt_rerank_start, c->stream);
        run_gpu_rerank(c->d_queries, c->d_top_ids, c->d_top_counts, d_base_vecs, c->d_final_ids, c->d_final_dists,
            c->d_stable_cnt, c->d_finished, c->current_batch_size, rerank_m, final_k, total_base_vecs, mini_batch_size, epsilon, beta, c->stream);
        cudaEventRecord(c->evt_rerank_end, c->stream);
        
        // D2H: 合并拷贝最终结果和统计信息，减少 D2H 传输次数
        cudaEventRecord(c->evt_d2h_start, c->stream);
        cudaMemcpyAsync(c->h_final_ids, c->d_final_ids, c->current_batch_size * final_k * sizeof(int), cudaMemcpyDeviceToHost, c->stream);
        // Stats Copy: 与最终结果一起拷贝，利用 PCIe 带宽
        cudaMemcpyAsync(c->h_atomic_reader, c->d_atomic, c->current_batch_size * sizeof(int), cudaMemcpyDeviceToHost, c->stream);
        cudaMemcpyAsync(c->h_total_max_reader, c->d_offsets + c->current_batch_size, sizeof(int), cudaMemcpyDeviceToHost, c->stream);
        cudaEventRecord(c->evt_d2h_end, c->stream);
        
        cudaEventRecord(c->cpu_wait_evt, c->stream);
        { std::lock_guard<std::mutex> lk(mtx); work_queue.push(c); }
        cv.notify_one();
        
        next_buffer_idx = (next_buffer_idx + 1) % num_streams;
    }
    
    void wait_all() {
        // 优化：直接同步所有流，比轮询更高效
        // cudaStreamSynchronize 内部已优化，会先快速检查再阻塞
        for(BatchCtx* ctx : buffers) {
            cudaStreamSynchronize(ctx->stream);
        }
        // 确保工作队列也处理完毕
        {
            std::unique_lock<std::mutex> lk(mtx);
            cv.wait(lk, [&]{ return work_queue.empty(); });
        }
    }

    void result_collection_loop() {
        cudaSetDevice(0);
        while (true) {
            BatchCtx* c = nullptr;
            {
                std::unique_lock<std::mutex> lk(mtx);
                cv.wait(lk, [&]{ return !work_queue.empty() || !running; });
                if (!running && work_queue.empty()) return;
                c = work_queue.front();
                work_queue.pop();
            }
            // 优化：直接使用 cudaEventSynchronize，它内部已优化（会先快速轮询再阻塞）
            // 比手动轮询更高效，减少不必要的 sleep 和上下文切换
            cudaEventSynchronize(c->cpu_wait_evt);
            
            float h2d, cagra, batch, rerank, d2h, e2e;
            cudaError_t err;
            err = cudaEventElapsedTime(&h2d, c->evt_h2d_start, c->evt_h2d_end); if(err) h2d=0;
            err = cudaEventElapsedTime(&cagra, c->evt_cagra_start, c->evt_cagra_end); if(err) cagra=0;
            err = cudaEventElapsedTime(&batch, c->evt_gpu_batch_start, c->evt_gpu_batch_end); if(err) batch=0;
            err = cudaEventElapsedTime(&rerank, c->evt_rerank_start, c->evt_rerank_end); if(err) rerank=0;
            err = cudaEventElapsedTime(&d2h, c->evt_d2h_start, c->evt_d2h_end); if(err) d2h=0;
            err = cudaEventElapsedTime(&e2e, c->evt_h2d_start, c->evt_d2h_end); if(err) e2e=0;
            
            // 优化：使用 compare-and-swap 循环减少锁竞争（比 load+store 更高效）
            auto add_atomic = [](std::atomic<double>& atom, double val) {
                double expected = atom.load();
                while (!atom.compare_exchange_weak(expected, expected + val, std::memory_order_relaxed)) {}
            };
            add_atomic(pipeline_timings.h2d_ms, h2d);
            add_atomic(pipeline_timings.cagra_ms, cagra);
            add_atomic(pipeline_timings.gpu_batch_ms, batch);
            add_atomic(pipeline_timings.gpu_rerank_ms, rerank);
            add_atomic(pipeline_timings.d2h_ms, d2h);
            add_atomic(pipeline_timings.e2e_ms, e2e);
            
            if (pipeline_timings.batch_count.load() == 0) pipeline_timings.first_batch_e2e.store(e2e);
            pipeline_timings.batch_count++;
            
            // Detailed Stats
            // 注意：由于Pipeline重排，Precompute与Resize Check的D2H是并行的
            float t_prune_count=0, t_scan_off=0, t_precompute_overlap=0, t_resize=0, t_scan_p1=0, t_scan_p2=0, t_compact=0, t_sort=0, t_gather=0, t_total=0;
            cudaEventElapsedTime(&t_prune_count, c->logic_events.evt_start, c->logic_events.evt_prune_count);
            cudaEventElapsedTime(&t_scan_off, c->logic_events.evt_prune_count, c->logic_events.evt_scan_offset);
            cudaEventElapsedTime(&t_precompute_overlap, c->logic_events.evt_scan_offset, c->logic_events.evt_precompute);
            cudaEventElapsedTime(&t_resize, c->logic_events.evt_precompute, c->logic_events.evt_resize_sync);
            cudaEventElapsedTime(&t_scan_p1, c->logic_events.evt_resize_sync, c->logic_events.evt_scan_phase1_end);
            cudaEventElapsedTime(&t_scan_p2, c->logic_events.evt_scan_phase1_end, c->logic_events.evt_scan_cand);
            cudaEventElapsedTime(&t_compact, c->logic_events.evt_scan_cand, c->logic_events.evt_compact);
            cudaEventElapsedTime(&t_sort, c->logic_events.evt_compact, c->logic_events.evt_sort);
            cudaEventElapsedTime(&t_gather, c->logic_events.evt_sort, c->logic_events.evt_gather);
            cudaEventElapsedTime(&t_total, c->logic_events.evt_start, c->logic_events.evt_gather);
            
            global_gpu_accum_ext.prune_count_ms += t_prune_count;
            global_gpu_accum_ext.scan_offset_ms += t_scan_off;
            global_gpu_accum_ext.precompute_ms += t_precompute_overlap;  // 包含Precompute + D2H并行时间
            global_gpu_accum_ext.resize_sync_ms += t_resize;  // 同步 + Resize时间
            global_gpu_accum_ext.scan_phase1_ms += t_scan_p1;
            global_gpu_accum_ext.scan_phase2_ms += t_scan_p2;
            global_gpu_accum_ext.compact_ms += t_compact;
            global_gpu_accum_ext.sort_ms += t_sort;
            global_gpu_accum_ext.gather_ms += t_gather;
            global_gpu_accum_ext.total_ms += t_total;

            long long real_count = 0;
            for(int i=0; i<c->current_batch_size; ++i) real_count += c->h_atomic_reader[i];
            global_gpu_accum_ext.total_candidates += real_count;

            if (c->h_total_max_reader) {
                global_gpu_accum_ext.total_max_candidates += *c->h_total_max_reader;
            }

            // 优化：使用 reserve + 直接赋值，避免多次 reallocation
            for(int i=0; i < c->current_batch_size; ++i) {
                final_results[c->id + i].resize(final_k);
                std::memcpy(final_results[c->id + i].data(), 
                           c->h_final_ids + i * final_k, 
                           final_k * sizeof(int));
            }

            {
                std::lock_guard<std::mutex> lk(c->state_mtx);
                c->is_busy = false;
            }
            c->state_cv.notify_all();
        }
    }
};

void load_ivf_index_to_gpu(const std::string& res_dir, IVFIndexGPU& idx, cudaStream_t stream = nullptr) {
    std::vector<float> codebook(PQ_M * PQ_K * PQ_SUB_DIM);
    read_binary_vector(res_dir + "/global_pq_codebook.bin", codebook);
    cudaMalloc(&idx.d_pq_codebook, codebook.size() * sizeof(float));
    if (stream) {
        cudaMemcpyAsync(idx.d_pq_codebook, codebook.data(), codebook.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemcpy(idx.d_pq_codebook, codebook.data(), codebook.size() * sizeof(float), cudaMemcpyHostToDevice);
    }

    std::ifstream ifs(res_dir + "/ivf_data.bin", std::ios::binary);
    if(!ifs) { std::cerr << "Error opening ivf_data.bin" << std::endl; exit(1); }
    int total_vecs, n_clusters;
    ifs.read((char*)&total_vecs, sizeof(int));
    ifs.read((char*)&n_clusters, sizeof(int));
    
    std::vector<int> offsets(n_clusters + 1); std::vector<int> ids(total_vecs); std::vector<uint8_t> codes((size_t)total_vecs * PQ_M);
    ifs.read((char*)offsets.data(), offsets.size() * sizeof(int));
    ifs.read((char*)ids.data(), ids.size() * sizeof(int));
    ifs.read((char*)codes.data(), codes.size() * sizeof(uint8_t));
    
    std::vector<float> centroids(n_clusters * DIM);
    ifs.read((char*)centroids.data(), centroids.size() * sizeof(float));
    
    cudaMalloc(&idx.d_cluster_offsets, offsets.size() * sizeof(int));
    if (stream) {
        cudaMemcpyAsync(idx.d_cluster_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemcpy(idx.d_cluster_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&idx.d_all_vector_ids, ids.size() * sizeof(int));
    if (stream) {
        cudaMemcpyAsync(idx.d_all_vector_ids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemcpy(idx.d_all_vector_ids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&idx.d_all_pq_codes, codes.size() * sizeof(uint8_t));
    if (stream) {
        cudaMemcpyAsync(idx.d_all_pq_codes, codes.data(), codes.size() * sizeof(uint8_t), cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemcpy(idx.d_all_pq_codes, codes.data(), codes.size() * sizeof(uint8_t), cudaMemcpyHostToDevice);
    }
    cudaMalloc(&idx.d_centroids, centroids.size() * sizeof(float));
    if (stream) {
        cudaMemcpyAsync(idx.d_centroids, centroids.data(), centroids.size() * sizeof(float), cudaMemcpyHostToDevice, stream);
    } else {
        cudaMemcpy(idx.d_centroids, centroids.data(), centroids.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
}

float compute_recall(const std::vector<std::vector<int>>& results, const int* groundtruth, int nq, int gt_k, int recall_k) {
    int total_hits = 0;
    for (int i = 0; i < nq; ++i) {
        // 构建该 query 的 GT 集合
        std::unordered_set<int> gt_set;
        for (int j = 0; j < gt_k; ++j) {
            gt_set.insert(groundtruth[i * gt_k + j]);
        }

        // 为了防止返回结果中存在重复 ID 被重复计数，
        // 这里对预测结果进行去重，只对每个唯一 ID 统计一次命中。
        std::unordered_set<int> seen_pred_ids;
        int hits = 0;
        int check_k = std::min(recall_k, (int)results[i].size());
        for (int j = 0; j < check_k; ++j) {
            int pred_id = results[i][j];
            // 只在第一次出现该 ID 时参与统计
            if (seen_pred_ids.insert(pred_id).second) {
                if (gt_set.count(pred_id)) {
                    hits++;
                }
            }
        }
        total_hits += hits;
    }
    return (float)total_hits / (nq * recall_k);
}

int main(int argc, char** argv) {
    cudaSetDevice(0);
    std::string base_path, query_path, gt_file;
    int batch_size = DEFAULT_BATCH_SIZE; int top_m = DEFAULT_TOP_M; int rerank_m = DEFAULT_RERANK_M;     
    int mini_batch_size = DEFAULT_RERANK_MINI_BATCH; float epsilon = DEFAULT_RERANK_EPSILON; int beta = DEFAULT_RERANK_BETA;
    float prune_ratio = 100.0f; 
    int n_streams = 2;
    int final_k = 10;
    int p1_lists = DEFAULT_P1_LISTS;
    int limit_k = DEFAULT_LIMIT_K;
    float threshold_coeff = DEFAULT_THRESHOLD_COEFF;
    
    for(int i=1; i<argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-b" || arg == "--batch_size") { if(i+1 < argc) batch_size = std::atoi(argv[++i]); }
        else if (arg == "-m" || arg == "--top_m") { if(i+1 < argc) top_m = std::atoi(argv[++i]); }
        else if (arg == "-r" || arg == "--rerank_m") { if(i+1 < argc) rerank_m = std::atoi(argv[++i]); }
        else if (arg == "-mb" || arg == "--mini_batch") { if(i+1 < argc) mini_batch_size = std::atoi(argv[++i]); }
        else if (arg == "-e" || arg == "--epsilon") { if(i+1 < argc) epsilon = std::atof(argv[++i]); }
        else if (arg == "--beta") { if(i+1 < argc) beta = std::atoi(argv[++i]); }
        else if (arg == "-p" || arg == "--prune") { if(i+1 < argc) prune_ratio = std::atof(argv[++i]); } 
        else if (arg == "-s" || arg == "--streams") { if(i+1 < argc) n_streams = std::atoi(argv[++i]); }
        else if (arg == "-k" || arg == "--topk") { if(i+1 < argc) final_k = std::atoi(argv[++i]); }
        else if (arg == "--p1_lists") { if(i+1 < argc) p1_lists = std::atoi(argv[++i]); }
        else if (arg == "--limit_k") { if(i+1 < argc) limit_k = std::atoi(argv[++i]); }
        else if (arg == "--threshold_coeff") { if(i+1 < argc) threshold_coeff = std::atof(argv[++i]); }
        else if (arg[0] != '-') {
            if (base_path.empty()) base_path = arg; else if (query_path.empty()) query_path = arg; else if (gt_file.empty()) gt_file = arg;
        }
    }

    if (base_path.empty() || query_path.empty()) {
        std::cerr << "Usage: ./search_pipeline <base> <query> [gt] [Options]\n"; return 1;
    }

    float* raw; int n = read_vecs<float>(base_path, raw, DIM);
    // 使用异步流加载索引
    cudaStream_t init_load_stream;
    cudaStreamCreate(&init_load_stream);
    IVFIndexGPU idx_gpu; 
    load_ivf_index_to_gpu("../res", idx_gpu, init_load_stream);
    cudaStreamSynchronize(init_load_stream);  // 初始化阶段需要等待完成
    cudaStreamDestroy(init_load_stream);
    float* q_ptr; int nq = read_vecs<float>(query_path, q_ptr, DIM);
    std::vector<float> all_q(q_ptr, q_ptr + (size_t)nq * DIM);
    
    // 初始化 Pipeline，传入所有参数
    SearchPipeline pipeline(batch_size, top_m, rerank_m, idx_gpu, raw, "../res/cagra_centroids.index", 
                           nq, n, mini_batch_size, epsilon, beta, prune_ratio, n_streams, final_k,
                           p1_lists, limit_k, threshold_coeff);

    // 预热
    {
        std::vector<float> warmup_q(all_q.begin(), all_q.begin() + std::min(nq, batch_size) * DIM);
        for(int i=0; i<n_streams; ++i){
            pipeline.submit_batch(warmup_q, 0);
        }
        pipeline.wait_all(); // 确保彻底完成
    }
    
    pipeline.reset_pipeline_timings();
    
    auto t_start = std::chrono::high_resolution_clock::now();
    int batches = (nq + batch_size - 1) / batch_size;
    for(int i=0; i<batches; ++i) {
        int start = i * batch_size;
        int sz = std::min(batch_size, nq - start);
        // 优化：避免不必要的 vector 拷贝，直接传递数据指针范围
        // 注意：submit_batch 内部会异步拷贝，所以这里可以安全地传递临时 vector
        std::vector<float> bq(sz * DIM);
        std::memcpy(bq.data(), all_q.data() + start * DIM, sz * DIM * sizeof(float));
        pipeline.submit_batch(bq, start);
    }
    pipeline.wait_all();
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    
    std::cout << "\nTotal time: " << elapsed_ms << " ms, QPS: " << nq / (elapsed_ms / 1000.0) << std::endl;
    pipeline.print_timing_stats(elapsed_ms);
    
    if (!gt_file.empty()) {
        int* gt_data = nullptr; int gt_k = 0; read_ivecs(gt_file, gt_data, gt_k);
        float r1 = compute_recall(pipeline.get_results(), gt_data, nq, gt_k, 1);
        float r10 = compute_recall(pipeline.get_results(), gt_data, nq, gt_k, 10);
        std::cout << "Recall@1: " << r1*100 << "%, Recall@10: " << r10*100 << "%" << std::endl;
        delete[] gt_data;
    }
    delete[] raw; delete[] q_ptr;
    std::quick_exit(0);
}