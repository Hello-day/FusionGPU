#ifndef UTILITY_ELBOW_CUH
#define UTILITY_ELBOW_CUH

#include <vector>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

// 效用函数法结果
struct UtilityResult {
    int best_p1_lists;              // 最佳 P1_LISTS
    float best_score;               // 最佳效用得分
    float coverage_at_best;         // 最佳点的覆盖率
    float cost_at_best;             // 最佳点的成本
    std::vector<float> scores;      // 每个位置的效用得分
    std::vector<float> costs;       // 每个位置的成本
    std::vector<float> coverages;   // 每个位置的覆盖率
};

// 成本配置参数
struct CostConfig {
    float lambda;           // 惩罚系数 (0.1~1.0)
    float filter_rate;      // Phase 2 过滤率 (0.3~0.7)
    float phase1_weight;    // Phase 1 权重 (0.5~0.7)
    float phase2_weight;    // Phase 2 权重 (0.3~0.5)
    int limit_k;            // 每个列表保留的候选数
    
    // 默认参数
    CostConfig() 
        : lambda(0.3f)
        , filter_rate(0.5f)
        , phase1_weight(0.6f)
        , phase2_weight(0.4f)
        , limit_k(100)
    {}
};

/**
 * 效用函数法确定最佳 P1_LISTS
 * 
 * 核心思想：Score(i) = Coverage(i) - λ × Cost(i) - Penalty(i)
 * 
 * 其中：
 * - Coverage(i)：覆盖率收益
 * - Cost(i)：计算成本
 * - Penalty(i)：尾部增长慢的惩罚
 * 
 * 硬性要求：覆盖率必须 >= 80%
 * 
 * @param hits_per_list 每个列表位置的命中数
 * @param total_hits 总命中数
 * @param config 成本配置参数
 * @return UtilityResult 包含最佳 P1_LISTS 和详细得分
 */
UtilityResult find_optimal_p1_lists_utility(
    const std::vector<int>& hits_per_list,
    int total_hits,
    const CostConfig& config = CostConfig()
) {
    UtilityResult result;
    int n = hits_per_list.size();
    
    result.scores.resize(n);
    result.costs.resize(n);
    result.coverages.resize(n);
    
    float best_score = -1e9f;
    int best_i = -1;
    
    const float MIN_COVERAGE_HARD = 0.80f;  // 硬性要求：覆盖率 >= 80%
    
    int cumsum = 0;
    for (int i = 0; i < n; ++i) {
        cumsum += hits_per_list[i];
        float coverage = (float)cumsum / total_hits;
        result.coverages[i] = coverage;
        
        // ===== 硬性过滤：覆盖率必须 >= 80% =====
        if (coverage < MIN_COVERAGE_HARD) {
            result.scores[i] = -1e9f;  // 标记为无效
            result.costs[i] = 0.0f;
            continue;
        }
        
        // ===== 成本估算 =====
        
        // Phase 1 成本：与列表数线性相关
        // 包括：PQ 距离计算 + 排序
        float cost_p1 = (float)(i + 1) / n;
        
        // Phase 2 成本：与候选数相关
        // 候选数 = 列表数 × LIMIT_K × 过滤率
        // 过滤率：cutoff 越大，过滤越少，Phase 2 成本越高
        float candidates = (i + 1) * config.limit_k * config.filter_rate;
        float max_candidates = n * config.limit_k * config.filter_rate;
        float cost_p2 = candidates / max_candidates;
        
        // 总成本（加权）
        float total_cost = config.phase1_weight * cost_p1 + 
                          config.phase2_weight * cost_p2;
        result.costs[i] = total_cost;
        
        // ===== 尾部增长惩罚 =====
        // 惩罚那些绝对增长慢的点（即使覆盖率已经很高）
        // 如果当前点的增长 < 前一个点的增长的50%，说明增长变慢了
        float tail_penalty = 0.0f;
        if (i > 0) {
            float current_growth = (float)hits_per_list[i] / total_hits;
            float prev_growth = (float)hits_per_list[i-1] / total_hits;
            
            // 如果增长速度下降超过50%，施加惩罚
            if (current_growth < prev_growth * 0.5f) {
                // 惩罚强度随着覆盖率增加而增加
                // 在80%-95%之间，惩罚从0增加到最大
                float coverage_excess = (coverage - MIN_COVERAGE_HARD) / (0.95f - MIN_COVERAGE_HARD);
                coverage_excess = std::min(1.0f, std::max(0.0f, coverage_excess));
                
                // 增长下降的幅度
                float growth_decline = (prev_growth - current_growth) / prev_growth;
                
                tail_penalty = coverage_excess * growth_decline * 0.2f;  // 最大惩罚 0.2
            }
        }
        
        // ===== 效用得分 =====
        // Score = Coverage - λ × Cost - Penalty
        float score = coverage - config.lambda * total_cost - tail_penalty;
        result.scores[i] = score;
        
        // 更新最佳点
        if (score > best_score) {
            best_score = score;
            best_i = i;
        }
    }
    
    // 如果没有找到满足条件的点，返回覆盖率最高的点
    if (best_i < 0) {
        best_i = n - 1;
        best_score = result.scores[n-1];
    }
    
    result.best_p1_lists = best_i + 1;
    result.best_score = best_score;
    result.coverage_at_best = result.coverages[best_i];
    result.cost_at_best = result.costs[best_i];
    
    return result;
}

/**
 * 打印效用函数法的详细结果
 */
void print_utility_result(const UtilityResult& result, const CostConfig& config) {
    std::cout << "\n  ========== 效用函数法结果 ==========" << std::endl;
    std::cout << "  最佳 P1_LISTS: " << result.best_p1_lists << std::endl;
    std::cout << "  最佳效用得分: " << std::fixed << std::setprecision(4) << result.best_score << std::endl;
    std::cout << "  覆盖率: " << std::fixed << std::setprecision(2) << result.coverage_at_best * 100 << "%" << std::endl;
    std::cout << "  成本: " << std::fixed << std::setprecision(4) << result.cost_at_best << std::endl;
    
    std::cout << "\n  参数配置:" << std::endl;
    std::cout << "  - λ (惩罚系数): " << config.lambda << std::endl;
    std::cout << "  - 过滤率: " << config.filter_rate << std::endl;
    std::cout << "  - Phase 1 权重: " << config.phase1_weight << std::endl;
    std::cout << "  - Phase 2 权重: " << config.phase2_weight << std::endl;
    std::cout << "  - LIMIT_K: " << config.limit_k << std::endl;
    
    // 打印详细得分表格（前20个）
    std::cout << "\n  详细得分 (前20个列表):" << std::endl;
    std::cout << "  " << std::string(90, '-') << std::endl;
    std::cout << "  List  Coverage   Cost_P1   Cost_P2   TotalCost   Score      Status" << std::endl;
    std::cout << "  " << std::string(90, '-') << std::endl;
    
    int n = std::min(20, (int)result.scores.size());
    for (int i = 0; i < n; ++i) {
        // 重新计算各部分成本用于显示
        float cost_p1 = (float)(i + 1) / result.scores.size();
        float candidates = (i + 1) * config.limit_k * config.filter_rate;
        float max_candidates = result.scores.size() * config.limit_k * config.filter_rate;
        float cost_p2 = candidates / max_candidates;
        
        std::string status = "";
        if (i + 1 == result.best_p1_lists) {
            status = "🌟 最佳";
        }
        
        std::cout << "  " << std::setw(4) << (i + 1) << "  "
                  << std::fixed << std::setprecision(2) << std::setw(7) << result.coverages[i] * 100 << "%  "
                  << std::fixed << std::setprecision(4) << std::setw(8) << cost_p1 << "  "
                  << std::fixed << std::setprecision(4) << std::setw(8) << cost_p2 << "  "
                  << std::fixed << std::setprecision(4) << std::setw(10) << result.costs[i] << "  "
                  << std::fixed << std::setprecision(4) << std::setw(9) << result.scores[i] << "  "
                  << status << std::endl;
    }
    std::cout << "  " << std::string(90, '=') << std::endl;
}

/**
 * 对比不同 λ 值的结果
 */
void compare_lambda_values(
    const std::vector<int>& hits_per_list,
    int total_hits,
    const std::vector<float>& lambda_values = {0.1f, 0.3f, 0.5f, 1.0f}
) {
    std::cout << "\n  ========== λ 参数敏感性分析 ==========" << std::endl;
    std::cout << "  " << std::string(80, '-') << std::endl;
    std::cout << "  λ值    最佳P1_LISTS  覆盖率    成本      得分      含义" << std::endl;
    std::cout << "  " << std::string(80, '-') << std::endl;
    
    for (float lambda : lambda_values) {
        CostConfig config;
        config.lambda = lambda;
        
        UtilityResult result = find_optimal_p1_lists_utility(hits_per_list, total_hits, config);
        
        std::string meaning;
        if (lambda <= 0.2f) meaning = "重召回率";
        else if (lambda <= 0.4f) meaning = "平衡";
        else if (lambda <= 0.7f) meaning = "重效率";
        else meaning = "极重效率";
        
        std::cout << "  " << std::fixed << std::setprecision(1) << std::setw(4) << lambda << "   "
                  << std::setw(12) << result.best_p1_lists << "  "
                  << std::fixed << std::setprecision(1) << std::setw(7) << result.coverage_at_best * 100 << "%  "
                  << std::fixed << std::setprecision(4) << std::setw(8) << result.cost_at_best << "  "
                  << std::fixed << std::setprecision(4) << std::setw(8) << result.best_score << "  "
                  << meaning << std::endl;
    }
    std::cout << "  " << std::string(80, '=') << std::endl;
}

#endif // UTILITY_ELBOW_CUH
