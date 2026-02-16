#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

#include <torch/torch.h>
#include <torch/script.h>

#include <vector>
#include <utility>
#include <string>
#include <cstring>
#include <stdexcept>

class ScenarioEnv;
struct EnvState;

namespace py = pybind11;

// ============================================================================
// 优化数据结构
// ============================================================================

/**
 * RingBuffer: 环形缓冲区，避免shift_append_obs_seq中的内存拷贝
 * 用于管理观测序列，支持O(1)时间复杂度的追加操作
 */
class RingBuffer {
public:
    RingBuffer() : seq_len_(0), obs_dim_(0), head_(0) {}
    
    RingBuffer(int seq_len, int obs_dim) 
        : seq_len_(seq_len), obs_dim_(obs_dim), head_(0) {
        data_.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(obs_dim), 0.0f);
    }
    
    void init(int seq_len, int obs_dim) {
        seq_len_ = seq_len;
        obs_dim_ = obs_dim;
        head_ = 0;
        data_.resize(static_cast<size_t>(seq_len) * static_cast<size_t>(obs_dim), 0.0f);
    }
    
    void reset() {
        head_ = 0;
        std::fill(data_.begin(), data_.end(), 0.0f);
    }
    
    // 用单个观测填充整个序列（用于初始化）
    void fill_with(const float* obs) {
        for (int t = 0; t < seq_len_; ++t) {
            std::memcpy(data_.data() + static_cast<size_t>(t) * static_cast<size_t>(obs_dim_),
                       obs, sizeof(float) * static_cast<size_t>(obs_dim_));
        }
        head_ = 0;
    }
    
    void fill_with(const std::vector<float>& obs) {
        if (static_cast<int>(obs.size()) != obs_dim_) {
            throw std::runtime_error("RingBuffer::fill_with: obs size mismatch");
        }
        fill_with(obs.data());
    }
    
    // O(1)追加新观测，覆盖最旧的
    void append(const float* obs_new) {
        std::memcpy(data_.data() + static_cast<size_t>(head_) * static_cast<size_t>(obs_dim_),
                   obs_new, sizeof(float) * static_cast<size_t>(obs_dim_));
        head_ = (head_ + 1) % seq_len_;
    }
    
    void append(const std::vector<float>& obs_new) {
        if (static_cast<int>(obs_new.size()) != obs_dim_) {
            throw std::runtime_error("RingBuffer::append: obs size mismatch");
        }
        append(obs_new.data());
    }
    
    // 输出为扁平化序列（按时间顺序：最旧到最新）
    void to_flat(std::vector<float>& out) const {
        out.resize(static_cast<size_t>(seq_len_) * static_cast<size_t>(obs_dim_));
        for (int i = 0; i < seq_len_; ++i) {
            int idx = (head_ + i) % seq_len_;
            std::memcpy(out.data() + static_cast<size_t>(i) * static_cast<size_t>(obs_dim_),
                       data_.data() + static_cast<size_t>(idx) * static_cast<size_t>(obs_dim_),
                       sizeof(float) * static_cast<size_t>(obs_dim_));
        }
    }
    
    std::vector<float> to_flat() const {
        std::vector<float> out;
        to_flat(out);
        return out;
    }
    
    // 从扁平化序列初始化
    void from_flat(const std::vector<float>& flat) {
        if (static_cast<int>(flat.size()) != seq_len_ * obs_dim_) {
            throw std::runtime_error("RingBuffer::from_flat: size mismatch");
        }
        data_ = flat;
        head_ = 0;
    }
    
    int seq_len() const { return seq_len_; }
    int obs_dim() const { return obs_dim_; }
    bool empty() const { return data_.empty(); }
    
private:
    int seq_len_;
    int obs_dim_;
    int head_;  // 指向下一个要写入的位置（也是最旧数据的位置）
    std::vector<float> data_;
};

/**
 * 预分配的工作缓冲区，避免rollout中频繁的内存分配
 */
 struct RolloutWorkspace {
    // 其他智能体的观测序列缓冲区
    std::vector<RingBuffer> other_obs_bufs;
    // 其他智能体的LSTM隐藏状态
    std::vector<std::vector<float>> other_h;
    std::vector<std::vector<float>> other_c;
    // 临时向量缓冲区
    std::vector<float> temp_obs_flat;
    std::vector<float> throttles;
    std::vector<float> steerings;
    
    // ========== 预分配 Tensor 缓冲区 ==========
    // 批量推理用的 tensor（避免每次 clone + to(device)）
    torch::Tensor obs_batch_tensor;      // (max_batch, seq_len, obs_dim)
    torch::Tensor h_batch_tensor;        // (1, max_batch, lstm_hidden_dim)
    torch::Tensor c_batch_tensor;        // (1, max_batch, lstm_hidden_dim)
    torch::Tensor action_batch_tensor;   // (max_batch, 2)
    
    // 单样本推理用的 tensor
    torch::Tensor obs_single_tensor;     // (1, seq_len, obs_dim)
    torch::Tensor h_single_tensor;       // (1, 1, lstm_hidden_dim)
    torch::Tensor c_single_tensor;       // (1, 1, lstm_hidden_dim)
    torch::Tensor action_single_tensor;  // (1, 2)
    
    // 配置参数（用于检查是否需要重新分配）
    int max_batch_size = 0;
    int seq_len = 0;
    int obs_dim = 0;
    int lstm_hidden_dim = 0;
    
    // 是否已初始化
    bool initialized = false;
    bool tensors_allocated = false;
    
    void init(int n_agents, int seq_len_, int obs_dim_, int lstm_hidden_dim_) {
        other_obs_bufs.resize(static_cast<size_t>(n_agents));
        other_h.resize(static_cast<size_t>(n_agents));
        other_c.resize(static_cast<size_t>(n_agents));
        
        for (int j = 0; j < n_agents; ++j) {
            other_obs_bufs[static_cast<size_t>(j)].init(seq_len_, obs_dim_);
            other_h[static_cast<size_t>(j)].resize(static_cast<size_t>(lstm_hidden_dim_), 0.0f);
            other_c[static_cast<size_t>(j)].resize(static_cast<size_t>(lstm_hidden_dim_), 0.0f);
        }
        
        temp_obs_flat.resize(static_cast<size_t>(seq_len_) * static_cast<size_t>(obs_dim_));
        throttles.resize(static_cast<size_t>(n_agents));
        steerings.resize(static_cast<size_t>(n_agents));
        
        // 保存配置
        max_batch_size = n_agents;
        seq_len = seq_len_;
        obs_dim = obs_dim_;
        lstm_hidden_dim = lstm_hidden_dim_;
        
        initialized = true;
    }
    
    void reset(int n_agents) {
        for (int j = 0; j < n_agents; ++j) {
            other_obs_bufs[static_cast<size_t>(j)].reset();
            std::fill(other_h[static_cast<size_t>(j)].begin(), 
                     other_h[static_cast<size_t>(j)].end(), 0.0f);
            std::fill(other_c[static_cast<size_t>(j)].begin(), 
                     other_c[static_cast<size_t>(j)].end(), 0.0f);
        }
    }
    
    // 确保 tensor 已分配（懒初始化，首次使用时分配）
    void ensure_tensors_allocated() {
        if (tensors_allocated) return;
        if (!initialized) {
            throw std::runtime_error("RolloutWorkspace::ensure_tensors_allocated called before init");
        }
        
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);
        
        // 批量推理 tensor
        obs_batch_tensor = torch::zeros({max_batch_size, seq_len, obs_dim}, opts);
        h_batch_tensor = torch::zeros({1, max_batch_size, lstm_hidden_dim}, opts);
        c_batch_tensor = torch::zeros({1, max_batch_size, lstm_hidden_dim}, opts);
        action_batch_tensor = torch::zeros({max_batch_size, 2}, opts);
        
        // 单样本推理 tensor
        obs_single_tensor = torch::zeros({1, seq_len, obs_dim}, opts);
        h_single_tensor = torch::zeros({1, 1, lstm_hidden_dim}, opts);
        c_single_tensor = torch::zeros({1, 1, lstm_hidden_dim}, opts);
        action_single_tensor = torch::zeros({1, 2}, opts);
        
        tensors_allocated = true;
    }
    
    // 获取 obs batch tensor 的可写 slice（避免 clone）
    // 返回 (batch_size, seq_len, obs_dim) 的 tensor
    torch::Tensor get_obs_batch_slice(int batch_size) {
        ensure_tensors_allocated();
        if (batch_size > max_batch_size) {
            // 需要扩展（罕见情况）
            auto opts = torch::TensorOptions().dtype(torch::kFloat32);
            obs_batch_tensor = torch::zeros({batch_size, seq_len, obs_dim}, opts);
            h_batch_tensor = torch::zeros({1, batch_size, lstm_hidden_dim}, opts);
            c_batch_tensor = torch::zeros({1, batch_size, lstm_hidden_dim}, opts);
            action_batch_tensor = torch::zeros({batch_size, 2}, opts);
            max_batch_size = batch_size;
        }
        return obs_batch_tensor.slice(0, 0, batch_size);
    }
    
    torch::Tensor get_h_batch_slice(int batch_size) {
        ensure_tensors_allocated();
        return h_batch_tensor.slice(1, 0, batch_size);
    }
    
    torch::Tensor get_c_batch_slice(int batch_size) {
        ensure_tensors_allocated();
        return c_batch_tensor.slice(1, 0, batch_size);
    }
    
    torch::Tensor get_action_batch_slice(int batch_size) {
        ensure_tensors_allocated();
        return action_batch_tensor.slice(0, 0, batch_size);
    }
    
    // 填充 obs batch tensor 的某一行
    void fill_obs_batch_row(int row_idx, const std::vector<float>& obs_seq_flat_data) {
        ensure_tensors_allocated();
        if ((int)obs_seq_flat_data.size() != seq_len * obs_dim) {
            throw std::runtime_error("fill_obs_batch_row: size mismatch");
        }
        auto accessor = obs_batch_tensor.accessor<float, 3>();
        const float* src = obs_seq_flat_data.data();
        for (int t = 0; t < seq_len; ++t) {
            for (int d = 0; d < obs_dim; ++d) {
                accessor[row_idx][t][d] = src[t * obs_dim + d];
            }
        }
    }
    
    // 填充 h/c batch tensor 的某一列（batch 维度）
    void fill_hc_batch_col(int col_idx, const std::vector<float>& h_data, const std::vector<float>& c_data) {
        ensure_tensors_allocated();
        auto h_acc = h_batch_tensor.accessor<float, 3>();
        auto c_acc = c_batch_tensor.accessor<float, 3>();
        for (int i = 0; i < lstm_hidden_dim; ++i) {
            h_acc[0][col_idx][i] = (i < (int)h_data.size()) ? h_data[i] : 0.0f;
            c_acc[0][col_idx][i] = (i < (int)c_data.size()) ? c_data[i] : 0.0f;
        }
    }
    
    // 填充单样本 tensor
    void fill_single_obs(const std::vector<float>& obs_seq_flat_data) {
        ensure_tensors_allocated();
        auto accessor = obs_single_tensor.accessor<float, 3>();
        const float* src = obs_seq_flat_data.data();
        for (int t = 0; t < seq_len; ++t) {
            for (int d = 0; d < obs_dim; ++d) {
                accessor[0][t][d] = src[t * obs_dim + d];
            }
        }
    }
    
    void fill_single_hc(const std::vector<float>& h_data, const std::vector<float>& c_data) {
        ensure_tensors_allocated();
        auto h_acc = h_single_tensor.accessor<float, 3>();
        auto c_acc = c_single_tensor.accessor<float, 3>();
        for (int i = 0; i < lstm_hidden_dim; ++i) {
            h_acc[0][0][i] = (i < (int)h_data.size()) ? h_data[i] : 0.0f;
            c_acc[0][0][i] = (i < (int)c_data.size()) ? c_data[i] : 0.0f;
        }
    }
    
    void fill_single_action(float throttle, float steering) {
        ensure_tensors_allocated();
        auto acc = action_single_tensor.accessor<float, 2>();
        acc[0][0] = throttle;
        acc[0][1] = steering;
    }
};

// 线程局部的工作空间，避免多线程竞争
#ifdef _MSC_VER
    #define MCTS_THREAD_LOCAL __declspec(thread)
#else
    #define MCTS_THREAD_LOCAL thread_local
#endif

// ============================================================================
// MCTS搜索函数声明
// ============================================================================

std::pair<std::vector<float>, py::dict> mcts_search_lstm_torchscript(
    ScenarioEnv& env,
    const EnvState& root_state,
    const std::vector<float>& root_obs_seq, // flattened (T*obs_dim)
    int seq_len,
    int obs_dim,
    const std::string& model_path,
    const std::vector<float>& root_h,
    const std::vector<float>& root_c,
    int lstm_hidden_dim,
    int agent_index,
    int num_simulations,
    int num_action_samples,
    int rollout_depth,
    float c_puct,
    float temperature,
    float gamma,
    float dirichlet_alpha,
    float dirichlet_eps,
    unsigned int seed
);

std::pair<std::vector<float>, py::dict> mcts_search_seq(
    ScenarioEnv& env,
    const EnvState& root_state,
    const std::vector<float>& root_obs_seq, // flattened (T*obs_dim)
    int seq_len,
    int obs_dim,
    const py::function& infer_fn,
    int agent_index,
    int num_simulations,
    int num_action_samples,
    int rollout_depth,
    float c_puct,
    float temperature,
    float gamma,
    float dirichlet_alpha,
    float dirichlet_eps,
    unsigned int seed
);

std::pair<std::vector<float>, py::dict> mcts_search(
    ScenarioEnv& env,
    const EnvState& root_state,
    const std::vector<float>& root_obs,
    const py::function& infer_fn,
    int agent_index,
    int num_simulations,
    int num_action_samples,
    int rollout_depth,
    float c_puct,
    float temperature,
    float gamma,
    float dirichlet_alpha,
    float dirichlet_eps,
    unsigned int seed
);

// (LSTM) MCTS API (2-step).
// infer_policy_value signature (Python):
//   (obs_batch: List[List[float]], h_batch: np.ndarray(B,H) or (B,1,H), c_batch: np.ndarray(B,H) or (B,1,H))
//     -> (mean: np.ndarray(B,2), std: np.ndarray(B,2), value: np.ndarray(B) or (B,1))
// infer_next_hidden signature (Python):
//   (obs_batch: List[List[float]], h_batch: np.ndarray(B,H) or (B,1,H), c_batch: np.ndarray(B,H) or (B,1,H), action_batch: np.ndarray(B,2))
//     -> (h_next: np.ndarray(B,H) or (B,1,H), c_next: np.ndarray(B,H) or (B,1,H))
std::pair<std::vector<float>, py::dict> mcts_search_lstm(
    ScenarioEnv& env,
    const EnvState& root_state,
    const std::vector<float>& root_obs,
    const py::function& infer_policy_value,
    const py::function& infer_next_hidden,
    const std::vector<float>& root_h,
    const std::vector<float>& root_c,
    int lstm_hidden_dim,
    int agent_index,
    int num_simulations,
    int num_action_samples,
    int rollout_depth,
    float c_puct,
    float temperature,
    float gamma,
    float dirichlet_alpha,
    float dirichlet_eps,
    unsigned int seed
);