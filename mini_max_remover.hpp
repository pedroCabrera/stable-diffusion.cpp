#ifndef __MINI_MAX_REMOVER_HPP__
#define __MINI_MAX_REMOVER_HPP__

#include <map>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common.hpp"
#include "ggml_extend.hpp"
#include "rope.hpp"
#include "vae.hpp"
#include "wan.hpp"

namespace MINIMAX {

    constexpr int MINIMAX_GRAPH_SIZE = 10240;
    constexpr float DEFAULT_EPS = 1e-6f;

    struct TensorAliasRegistry {
        static std::unordered_map<std::string, std::string>& aliases() {
            static std::unordered_map<std::string, std::string> map;
            return map;
        }

        static void register_alias(const std::string& alias, const std::string& original) {
            if (alias == original) {
                return;
            }
            aliases()[alias] = original;
        }

        static bool resolve_alias(const std::string& alias, std::string& original) {
            const auto& map = aliases();
            auto it         = map.find(alias);
            if (it == map.end()) {
                return false;
            }
            original = it->second;
            return true;
        }
    };

    // ================================== FP32LayerNorm ==================================
    class FP32LayerNorm : public UnaryBlock {
    protected:
        int64_t dim;
        float eps;
        bool elementwise_affine;

        void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {
            if (elementwise_affine) {
                params["weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
                params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            }
        }

    public:
        FP32LayerNorm(int64_t dim, float eps = DEFAULT_EPS, bool elementwise_affine = true)
            : dim(dim), eps(eps), elementwise_affine(elementwise_affine) {}

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* x) {
            // Convert to FP32 for normalization
            struct ggml_tensor* x_f32 = ggml_cast(ctx, x, GGML_TYPE_F32);

            // Apply layer normalization
            x_f32 = ggml_norm(ctx, x_f32, eps);

            if (elementwise_affine && params.find("weight") != params.end()) {
                x_f32 = ggml_mul(ctx, x_f32, params["weight"]);
                x_f32 = ggml_add(ctx, x_f32, params["bias"]);
            }

            // Convert back to original type
            return ggml_cast(ctx, x_f32, x->type);
        }

        void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            if (elementwise_affine) {
                tensors[prefix + "weight"] = params["weight"];
                tensors[prefix + "bias"] = params["bias"];
            }
        }
    };

    // ================================== 3D Rotary Position Embedding ==================================
    class RotaryPosEmbed3D : public GGMLBlock {
    protected:
        int attention_head_dim;
        std::tuple<int, int, int> patch_size;
        int max_seq_len;
        float theta;
        int h_dim, w_dim, t_dim;

        void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {
            // Calculate dimensions for t, h, w as in Python
            h_dim = w_dim = 2 * (attention_head_dim / 6);
            t_dim = attention_head_dim - h_dim - w_dim;

            // Pre-computed frequency tensors for each dimension
            params["freqs_t"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, t_dim / 2, max_seq_len);
            params["freqs_h"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, h_dim / 2, max_seq_len);
            params["freqs_w"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, w_dim / 2, max_seq_len);

            // Initialize frequency tensors with proper values
            float* freqs_t_data = (float*)params["freqs_t"]->data;
            float* freqs_h_data = (float*)params["freqs_h"]->data;
            float* freqs_w_data = (float*)params["freqs_w"]->data;

            for (int i = 0; i < max_seq_len; i++) {
                for (int j = 0; j < t_dim / 2; j++) {
                    freqs_t_data[i * (t_dim / 2) + j] = (float)i * pow(theta, -2.0f * j / t_dim);
                }
                for (int j = 0; j < h_dim / 2; j++) {
                    freqs_h_data[i * (h_dim / 2) + j] = (float)i * pow(theta, -2.0f * j / h_dim);
                }
                for (int j = 0; j < w_dim / 2; j++) {
                    freqs_w_data[i * (w_dim / 2) + j] = (float)i * pow(theta, -2.0f * j / w_dim);
                }
            }
        }

    public:
        RotaryPosEmbed3D(int attention_head_dim, std::tuple<int, int, int> patch_size, int max_seq_len = 1024, float theta = 10000.0f)
            : attention_head_dim(attention_head_dim), patch_size(patch_size), max_seq_len(max_seq_len), theta(theta) {}

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* hidden_states) {
            // hidden_states: [batch, in_channels, frames, height, width]
            int batch_size = hidden_states->ne[4];
            int num_channels = hidden_states->ne[2];
            int num_frames = hidden_states->ne[1];
            int height = hidden_states->ne[0];

            int p_t = std::get<0>(patch_size), p_h = std::get<1>(patch_size), p_w = std::get<2>(patch_size);
            int ppf = num_frames / p_t;
            int pph = height / p_h;
            int ppw = num_channels / p_w;  // Note: this might need adjustment based on actual tensor layout

            // Get appropriate slices of frequency tensors
            struct ggml_tensor* freqs_t_slice = ggml_view_1d(ctx, params["freqs_t"], ppf, 0);
            struct ggml_tensor* freqs_h_slice = ggml_view_1d(ctx, params["freqs_h"], pph, 0);
            struct ggml_tensor* freqs_w_slice = ggml_view_1d(ctx, params["freqs_w"], ppw, 0);

            // Expand frequencies to match spatial dimensions
            freqs_t_slice = ggml_repeat(ctx, ggml_reshape_4d(ctx, freqs_t_slice, 1, 1, ppf, freqs_t_slice->ne[0]), ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 1, pph, ppw, 1));
            freqs_h_slice = ggml_repeat(ctx, ggml_reshape_4d(ctx, freqs_h_slice, 1, pph, 1, freqs_h_slice->ne[0]), ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ppf, 1, ppw, 1));
            freqs_w_slice = ggml_repeat(ctx, ggml_reshape_4d(ctx, freqs_w_slice, 1, 1, ppw, freqs_w_slice->ne[0]), ggml_new_tensor_4d(ctx, GGML_TYPE_F32, ppf, pph, 1, 1));

            // Concatenate frequencies
            struct ggml_tensor* freqs = ggml_concat(ctx, freqs_t_slice, freqs_h_slice, 3);
            freqs = ggml_concat(ctx, freqs, freqs_w_slice, 3);

            // Reshape to final frequency tensor [1, 1, seq_len, attention_head_dim]
            int seq_len = ppf * pph * ppw;
            freqs = ggml_reshape_4d(ctx, freqs, 1, 1, seq_len, attention_head_dim);

            return freqs;
        }

        void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            tensors[prefix + "freqs_t"] = params["freqs_t"];
            tensors[prefix + "freqs_h"] = params["freqs_h"];
            tensors[prefix + "freqs_w"] = params["freqs_w"];
        }
    };

    // ================================== MiniMax Attention Processor ==================================
    class MinimaxAttnProcessor : public GGMLBlock {
    protected:
        int64_t dim;
        int64_t heads;
        std::string qk_norm;
        float eps;

        void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {
            // QKV projection matrices
            params["to_q"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dim, dim);
            params["to_k"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dim, dim);
            params["to_v"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dim, dim);
            params["to_out_0"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dim, dim);
            params["to_out_1"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);

            if (qk_norm == "rms_norm_across_heads") {
                // QK normalization (RMS norm without affine)
                params["norm_q_weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
                params["norm_k_weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            }
        }

    public:
        MinimaxAttnProcessor(int64_t dim, int64_t heads, std::string qk_norm = "rms_norm_across_heads", float eps = DEFAULT_EPS)
            : dim(dim), heads(heads), qk_norm(qk_norm), eps(eps) {}

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                   struct ggml_tensor* hidden_states,
                                   struct ggml_tensor* rotary_emb = nullptr) {
            int batch_size = hidden_states->ne[2];
            int seq_len = hidden_states->ne[1];

            // QKV projections
            struct ggml_tensor* query = ggml_mul_mat(ctx, params["to_q"], hidden_states);
            struct ggml_tensor* key = ggml_mul_mat(ctx, params["to_k"], hidden_states);
            struct ggml_tensor* value = ggml_mul_mat(ctx, params["to_v"], hidden_states);

            // Apply QK normalization if specified
            if (qk_norm == "rms_norm_across_heads") {
                query = ggml_rms_norm(ctx, query, eps);
                query = ggml_mul(ctx, query, params["norm_q_weight"]);

                key = ggml_rms_norm(ctx, key, eps);
                key = ggml_mul(ctx, key, params["norm_k_weight"]);
            }

            // Reshape for multi-head attention: [batch, seq_len, dim] -> [batch, seq_len, heads, head_dim]
            int head_dim = dim / heads;
            query = ggml_reshape_4d(ctx, query, head_dim, heads, seq_len, batch_size);
            key = ggml_reshape_4d(ctx, key, head_dim, heads, seq_len, batch_size);
            value = ggml_reshape_4d(ctx, value, head_dim, heads, seq_len, batch_size);

            // Permute to [head_dim, seq_len, heads, batch] for attention
            query = ggml_permute(ctx, query, 0, 2, 1, 3);
            key = ggml_permute(ctx, key, 0, 2, 1, 3);
            value = ggml_permute(ctx, value, 0, 2, 1, 3);

            // Apply rotary embeddings if provided
            if (rotary_emb != nullptr) {
                query = apply_rotary_emb(ctx, query, rotary_emb);
                key = apply_rotary_emb(ctx, key, rotary_emb);
            }

            // Scaled dot-product attention
            struct ggml_tensor* attn_weights = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, key)), query);
            attn_weights = ggml_scale(ctx, attn_weights, 1.0f / sqrtf((float)head_dim));
            attn_weights = ggml_soft_max(ctx, attn_weights);

            struct ggml_tensor* attn_output = ggml_mul_mat(ctx, value, attn_weights);

            // Reshape back: [head_dim, seq_len, heads, batch] -> [batch, seq_len, dim]
            attn_output = ggml_permute(ctx, attn_output, 0, 2, 1, 3);
            attn_output = ggml_reshape_3d(ctx, attn_output, dim, seq_len, batch_size);

            // Output projection
            attn_output = ggml_mul_mat(ctx, params["to_out_0"], attn_output);
            attn_output = ggml_add(ctx, attn_output, params["to_out_1"]);

            return attn_output;
        }

        void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            tensors[prefix + "to_q.weight"] = params["to_q"];
            tensors[prefix + "to_k.weight"] = params["to_k"];
            tensors[prefix + "to_v.weight"] = params["to_v"];
            tensors[prefix + "to_out.0.weight"] = params["to_out_0"];
            tensors[prefix + "to_out.1.weight"] = params["to_out_1"];

            if (qk_norm == "rms_norm_across_heads") {
                tensors[prefix + "norm_q.weight"] = params["norm_q_weight"];
                tensors[prefix + "norm_k.weight"] = params["norm_k_weight"];
            }
        }

    private:
        struct ggml_tensor* apply_rotary_emb(struct ggml_context* ctx, struct ggml_tensor* hidden_states, struct ggml_tensor* freqs) {
            // Apply rotary embedding using complex number rotation as in Python
            // hidden_states: [head_dim, seq_len, heads, batch]
            // freqs: [1, 1, seq_len, head_dim]

            int head_dim = hidden_states->ne[0];
            int seq_len = hidden_states->ne[1];
            int heads = hidden_states->ne[2];
            int batch_size = hidden_states->ne[3];

            // Split into real and imaginary parts
            int half_dim = head_dim / 2;
            struct ggml_tensor* x_real = ggml_view_4d(ctx, hidden_states, half_dim, seq_len, heads, batch_size, 0, 0, 0, 0);
            struct ggml_tensor* x_imag = ggml_view_4d(ctx, hidden_states, half_dim, seq_len, heads, batch_size, half_dim * sizeof(float), 0, 0, 0);

            // Get cos and sin from frequencies (freqs contains pre-computed cos/sin values)
            struct ggml_tensor* cos_freqs = ggml_cos(ctx, freqs);
            struct ggml_tensor* sin_freqs = ggml_sin(ctx, freqs);

            // Apply rotation: x_rotated = x_real * cos - x_imag * sin, x_imag * cos + x_real * sin
            struct ggml_tensor* x_real_rot = ggml_sub(ctx,
                ggml_mul(ctx, x_real, cos_freqs),
                ggml_mul(ctx, x_imag, sin_freqs));
            struct ggml_tensor* x_imag_rot = ggml_add(ctx,
                ggml_mul(ctx, x_imag, cos_freqs),
                ggml_mul(ctx, x_real, sin_freqs));

            // Concatenate back
            return ggml_concat(ctx, x_real_rot, x_imag_rot, -1);
        }
    };
    class MinimaxTransformerBlock : public GGMLBlock {
    protected:
        int64_t dim;
        int64_t ffn_dim;
        int64_t num_heads;
        std::string qk_norm;
        bool cross_attn_norm;
        float eps;

        void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {
            // Create parameters that match the actual tensors in safetensors
            params["scale_shift_table"] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, 2, 1);

            // Attention parameters
            params["norm_q_weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            params["norm_k_weight"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            params["to_q"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dim, dim);
            params["to_q_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            params["to_k"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dim, dim);
            params["to_k_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            params["to_v"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dim, dim);
            params["to_v_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);
            params["to_out_0"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F16, dim, dim);
            params["to_out_1"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, dim);

            // Initialize scale_shift_table with random values
            float* data = (float*)params["scale_shift_table"]->data;
            for (int i = 0; i < dim * 2; i++) {
                data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f / sqrt(dim);
            }
        }

    public:
        MinimaxTransformerBlock(int64_t dim, int64_t ffn_dim, int64_t num_heads,
                               std::string qk_norm = "rms_norm_across_heads",
                               bool cross_attn_norm = false, float eps = DEFAULT_EPS)
            : dim(dim), ffn_dim(ffn_dim), num_heads(num_heads), qk_norm(qk_norm),
              cross_attn_norm(cross_attn_norm), eps(eps) {

            // Layer norms (elementwise_affine=False for MiniMax)
            blocks["norm1"] = std::shared_ptr<GGMLBlock>(new FP32LayerNorm(dim, eps, false));
            blocks["norm2"] = std::shared_ptr<GGMLBlock>(new FP32LayerNorm(dim, eps, false));

            // Self-attention with QK normalization
            blocks["attn1"] = std::shared_ptr<GGMLBlock>(new MinimaxAttnProcessor(dim, num_heads, qk_norm, eps));

            // Feed-forward network
            blocks["ffn"] = std::shared_ptr<GGMLBlock>(new FeedForward(dim, ffn_dim, 4));
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                   struct ggml_tensor* hidden_states,
                                   struct ggml_tensor* temb,
                                   struct ggml_tensor* rotary_emb = nullptr) {

            // Get modulation parameters: scale_shift_table + temb, then chunk into 2 parts
            struct ggml_tensor* modulation = ggml_add(ctx, params["scale_shift_table"], ggml_reshape_2d(ctx, temb, dim, 1));

            // Split into 2 components: shift, scale
            struct ggml_tensor* shift = ggml_view_2d(ctx, modulation, dim, 1, 0, 0);
            struct ggml_tensor* scale = ggml_view_2d(ctx, modulation, dim, 1, dim * sizeof(float), 0);

            // Self-attention block
            // QKV projections with bias
            struct ggml_tensor* query = ggml_mul_mat(ctx, params["to_q"], hidden_states);
            query = ggml_add(ctx, query, params["to_q_bias"]);

            struct ggml_tensor* key = ggml_mul_mat(ctx, params["to_k"], hidden_states);
            key = ggml_add(ctx, key, params["to_k_bias"]);

            struct ggml_tensor* value = ggml_mul_mat(ctx, params["to_v"], hidden_states);
            value = ggml_add(ctx, value, params["to_v_bias"]);

            // Apply QK normalization
            query = ggml_rms_norm(ctx, query, eps);
            query = ggml_mul(ctx, query, params["norm_q_weight"]);

            key = ggml_rms_norm(ctx, key, eps);
            key = ggml_mul(ctx, key, params["norm_k_weight"]);

            // Reshape for multi-head attention
            int head_dim = dim / num_heads;
            query = ggml_reshape_4d(ctx, query, head_dim, num_heads, hidden_states->ne[1], hidden_states->ne[2]);
            key = ggml_reshape_4d(ctx, key, head_dim, num_heads, hidden_states->ne[1], hidden_states->ne[2]);
            value = ggml_reshape_4d(ctx, value, head_dim, num_heads, hidden_states->ne[1], hidden_states->ne[2]);

            // Permute for attention
            query = ggml_permute(ctx, query, 0, 2, 1, 3);
            key = ggml_permute(ctx, key, 0, 2, 1, 3);
            value = ggml_permute(ctx, value, 0, 2, 1, 3);

            // Scaled dot-product attention
            struct ggml_tensor* attn_weights = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, key)), query);
            attn_weights = ggml_scale(ctx, attn_weights, 1.0f / sqrtf((float)head_dim));
            attn_weights = ggml_soft_max(ctx, attn_weights);

            struct ggml_tensor* attn_output = ggml_mul_mat(ctx, value, attn_weights);

            // Reshape back: [head_dim, seq_len, heads, batch] -> [batch, seq_len, dim]
            attn_output = ggml_permute(ctx, attn_output, 0, 2, 1, 3);
            attn_output = ggml_reshape_3d(ctx, attn_output, dim, hidden_states->ne[1], hidden_states->ne[2]);

            // Output projection
            attn_output = ggml_mul_mat(ctx, params["to_out_0"], attn_output);
            attn_output = ggml_add(ctx, attn_output, params["to_out_1"]);

            // Residual connection
            hidden_states = ggml_add(ctx, hidden_states, attn_output);

            // Feed-forward network
            struct ggml_tensor* ff_input = ggml_mul_mat(ctx, params["ffn_0_weight"], hidden_states);
            ff_input = ggml_add(ctx, ff_input, params["ffn_0_bias"]);
            ff_input = ggml_silu(ctx, ff_input);

            struct ggml_tensor* ff_output = ggml_mul_mat(ctx, params["ffn_2_weight"], ff_input);
            ff_output = ggml_add(ctx, ff_output, params["ffn_2_bias"]);

            // Apply modulation and residual
            struct ggml_tensor* modulated_ff = ggml_mul(ctx, ff_output, ggml_add(ctx, ggml_new_f32(ctx, 1.0f), scale));
            modulated_ff = ggml_add(ctx, modulated_ff, shift);
            hidden_states = ggml_add(ctx, hidden_states, modulated_ff);

            return hidden_states;
        }

        void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            tensors[prefix + "scale_shift_table"] = params["scale_shift_table"];

            // Map attention tensors directly to match safetensors structure
            tensors[prefix + "attn1.norm_q.weight"] = params["norm_q_weight"];
            tensors[prefix + "attn1.norm_k.weight"] = params["norm_k_weight"];
            tensors[prefix + "attn1.to_q.weight"] = params["to_q"];
            tensors[prefix + "attn1.to_q.bias"] = params["to_q_bias"];
            tensors[prefix + "attn1.to_k.weight"] = params["to_k"];
            tensors[prefix + "attn1.to_k.bias"] = params["to_k_bias"];
            tensors[prefix + "attn1.to_v.weight"] = params["to_v"];
            tensors[prefix + "attn1.to_v.bias"] = params["to_v_bias"];
            tensors[prefix + "attn1.to_out.0.weight"] = params["to_out_0"];
            tensors[prefix + "attn1.to_out.1.weight"] = params["to_out_1"];

            // Map FFN tensors
            tensors[prefix + "ffn.net.0.proj.weight"] = params["ffn_0_weight"];
            tensors[prefix + "ffn.net.0.proj.bias"] = params["ffn_0_bias"];
            tensors[prefix + "ffn.net.2.weight"] = params["ffn_2_weight"];
            tensors[prefix + "ffn.net.2.bias"] = params["ffn_2_bias"];
        }
    };

    // ================================== 3D Patch Embedding ==================================
    class PatchEmbed3D : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t inner_dim;
        std::tuple<int, int, int> patch_size;

        void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {
            // Conv3D weight: [out_channels, in_channels, kT, kH, kW]
            int64_t ne[5] = {std::get<2>(patch_size), std::get<1>(patch_size), std::get<0>(patch_size), in_channels, inner_dim};
            params["weight"] = ggml_new_tensor(ctx, GGML_TYPE_F16, 5, ne);
            params["bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, inner_dim);
        }

    public:
        PatchEmbed3D(int64_t in_channels, int64_t inner_dim, std::tuple<int, int, int> patch_size)
            : in_channels(in_channels), inner_dim(inner_dim), patch_size(patch_size) {}

        struct ggml_tensor* forward(struct ggml_context* ctx, struct ggml_tensor* hidden_states) {
            // hidden_states: [batch, in_channels, frames, height, width]
            // Apply 3D convolution
            struct ggml_tensor* conv_output = ggml_conv_3d(ctx, params["weight"], hidden_states,
                                                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
            conv_output = ggml_add(ctx, conv_output, params["bias"]);

            // Flatten spatial dimensions: [batch, inner_dim, frames, height, width] -> [batch, seq_len, inner_dim]
            int batch_size = conv_output->ne[4];
            int seq_len = conv_output->ne[3] * conv_output->ne[2] * conv_output->ne[1];

            conv_output = ggml_reshape_3d(ctx, conv_output, inner_dim, seq_len, batch_size);
            // Permute [inner_dim, seq_len, batch] -> [seq_len, inner_dim, batch]
            // Since ggml_permute is for 4D tensors, we reshape to 4D, permute, then back to 3D
            conv_output = ggml_reshape_4d(ctx, conv_output, inner_dim, seq_len, batch_size, 1);
            conv_output = ggml_permute(ctx, conv_output, 1, 0, 2, 3); // [seq_len, inner_dim, batch, 1]
            conv_output = ggml_reshape_3d(ctx, conv_output, seq_len, inner_dim, batch_size);

            return conv_output;
        }

        void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            tensors[prefix + "weight"] = params["weight"];
            tensors[prefix + "bias"] = params["bias"];
        }
    };

    // ================================== Timestep Embedding ==================================
    class TimestepEmbedding : public GGMLBlock {
    protected:
        int64_t in_channels;
        int64_t time_embed_dim;

        void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {
            // Sinusoidal embedding projection
            params["timesteps_proj_weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_channels, time_embed_dim);
            params["timesteps_proj_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, time_embed_dim);

            // Time embedder
            params["time_embedder_weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, time_embed_dim, time_embed_dim);
            params["time_embedder_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, time_embed_dim);

            // Time projection
            params["time_proj_weight"] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, time_embed_dim, time_embed_dim * 6);
            params["time_proj_bias"] = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, time_embed_dim * 6);
        }

    public:
        TimestepEmbedding(int64_t in_channels, int64_t time_embed_dim)
            : in_channels(in_channels), time_embed_dim(time_embed_dim) {}

        std::pair<struct ggml_tensor*, struct ggml_tensor*> forward(struct ggml_context* ctx, struct ggml_tensor* timestep) {
            // Timestep projection (sinusoidal embedding)
            struct ggml_tensor* t_emb = ggml_mul_mat(ctx, params["timesteps_proj_weight"], timestep);
            t_emb = ggml_add(ctx, t_emb, params["timesteps_proj_bias"]);

            // Time embedder
            t_emb = ggml_mul_mat(ctx, params["time_embedder_weight"], t_emb);
            t_emb = ggml_add(ctx, t_emb, params["time_embedder_bias"]);

            // SiLU activation
            t_emb = ggml_silu(ctx, t_emb);

            // Time projection
            struct ggml_tensor* timestep_proj = ggml_mul_mat(ctx, params["time_proj_weight"], t_emb);
            timestep_proj = ggml_add(ctx, timestep_proj, params["time_proj_bias"]);

            return {t_emb, timestep_proj};
        }

        void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            tensors[prefix + "timesteps_proj.weight"] = params["timesteps_proj_weight"];
            tensors[prefix + "timesteps_proj.bias"] = params["timesteps_proj_bias"];
            tensors[prefix + "time_embedder.weight"] = params["time_embedder_weight"];
            tensors[prefix + "time_embedder.bias"] = params["time_embedder_bias"];
            tensors[prefix + "time_proj.weight"] = params["time_proj_weight"];
            tensors[prefix + "time_proj.bias"] = params["time_proj_bias"];
        }
    };

    // ================================== 3D Patch Embedding ==================================
    class MinimaxTransformer;
    class MinimaxModel : public GGMLBlock {
    protected:
        std::tuple<int, int, int> patch_size;
        int64_t num_attention_heads;
        int64_t attention_head_dim;
        int64_t in_channels;
        int64_t out_channels;
        int64_t freq_dim;
        int64_t ffn_dim;
        int64_t num_layers;
        bool cross_attn_norm;
        std::string qk_norm;
        float eps;
        int rope_max_seq_len;
        int64_t inner_dim;

    public:
        MinimaxModel(std::tuple<int, int, int> patch_size = {1, 2, 2},
                     int64_t num_attention_heads = 40,
                     int64_t attention_head_dim = 128,
                     int64_t in_channels = 16,
                     int64_t out_channels = 16,
                     int64_t freq_dim = 256,
                     int64_t ffn_dim = 13824,
                     int64_t num_layers = 40,
                     bool cross_attn_norm = true,
                     std::string qk_norm = "rms_norm_across_heads",
                     float eps = DEFAULT_EPS,
                     int rope_max_seq_len = 1024)
            : patch_size(patch_size), num_attention_heads(num_attention_heads),
              attention_head_dim(attention_head_dim), in_channels(in_channels),
              out_channels(out_channels), freq_dim(freq_dim), ffn_dim(ffn_dim),
              num_layers(num_layers), cross_attn_norm(cross_attn_norm), qk_norm(qk_norm),
              eps(eps), rope_max_seq_len(rope_max_seq_len) {

            inner_dim = num_attention_heads * attention_head_dim;

            // No block-based initialization needed - using direct parameters
        }

        void init_params(struct ggml_context* ctx, const String2GGMLType& tensor_types = {}, const std::string prefix = "") {
            LOG_DEBUG("MinimaxModel::init_params started");
            ggml_type wtype = GGML_TYPE_F32;
            auto get_type = [&](const std::string& name) -> ggml_type {
                auto iter = tensor_types.find(name);
                return iter != tensor_types.end() ? iter->second : wtype;
            };

            // Helper to create tensor with appropriate GGML function based on dimensions
            auto create_tensor = [&](const std::string& name, const std::vector<int64_t>& shape) -> struct ggml_tensor* {
                ggml_type type = get_type(name);
                LOG_DEBUG("Creating tensor '%s' with %zu dims", name.c_str(), shape.size());
                if (shape.size() == 1) {
                    return ggml_new_tensor_1d(ctx, type, shape[0]);
                } else if (shape.size() == 2) {
                    return ggml_new_tensor_2d(ctx, type, shape[0], shape[1]);
                } else if (shape.size() == 3) {
                    return ggml_new_tensor_3d(ctx, type, shape[0], shape[1], shape[2]);
                } else if (shape.size() == 4) {
                    return ggml_new_tensor_4d(ctx, type, shape[0], shape[1], shape[2], shape[3]);
                } else {
                    // Fallback, but shouldn't happen
                    int64_t ne[4] = {1, 1, 1, 1};
                    for (size_t i = 0; i < shape.size() && i < 4; ++i) {
                        ne[i] = shape[i];
                    }
                    return ggml_new_tensor(ctx, type, 4, ne);
                }
            };

            // Patch embedding - Safetensors has [1536, 48, 1, 2, 2] (5D)
            // Model loader collapses to [73728, 1, 2, 2] then reverses to [2, 2, 1, 73728]
            int64_t patch_shape[4] = {2, 2, 1, 73728};
            params["patch_embedding.weight"] = ggml_new_tensor(ctx, get_type("patch_embedding.weight"), 4, patch_shape);
            params["patch_embedding.bias"] = create_tensor("patch_embedding.bias", {inner_dim});

            // Time embedder - GGML loads 2D as 4D with trailing 1s
            //  Time embedder - shapes after GGML reversal
            // Safetensors [1536, 256] → GGML [256, 1536]
            int64_t time_embed_dim = inner_dim;
            params["condition_embedder.time_embedder.linear_1.weight"] = create_tensor("condition_embedder.time_embedder.linear_1.weight", {freq_dim, time_embed_dim});
            params["condition_embedder.time_embedder.linear_1.bias"] = create_tensor("condition_embedder.time_embedder.linear_1.bias", {time_embed_dim});
            params["condition_embedder.time_embedder.linear_2.weight"] = create_tensor("condition_embedder.time_embedder.linear_2.weight", {time_embed_dim, time_embed_dim});
            params["condition_embedder.time_embedder.linear_2.bias"] = create_tensor("condition_embedder.time_embedder.linear_2.bias", {time_embed_dim});
            // Safetensors [9216, 1536] → GGML [1536, 9216]
            params["condition_embedder.time_proj.weight"] = create_tensor("condition_embedder.time_proj.weight", {time_embed_dim, time_embed_dim * 6});
            params["condition_embedder.time_proj.bias"] = create_tensor("condition_embedder.time_proj.bias", {time_embed_dim * 6});

            // Output projection
            // Safetensors [64, 1536] → GGML [1536, 64]
            params["proj_out.weight"] = create_tensor("proj_out.weight", {inner_dim, out_channels});
            params["proj_out.bias"] = create_tensor("proj_out.bias", {out_channels});
            // NOTE: norm_out.weight doesn't exist in the safetensors file!

            // Global scale_shift_table
            // Safetensors [1, 2, 1536] → GGML [1536, 2, 1]
            params["scale_shift_table"] = create_tensor("scale_shift_table", {inner_dim, 2, 1});

            // Transformer blocks
            LOG_DEBUG("Creating tensors for %d transformer blocks", num_layers);
            for (int i = 0; i < num_layers; i++) {
                if (i % 10 == 0) {
                    LOG_DEBUG("Creating block %d/%d tensors", i, num_layers);
                }
                char block_prefix[32];
                sprintf(block_prefix, "blocks.%d.", i);
                std::string bp = block_prefix;

                // Attention norms
                params[bp + "attn1.norm_q.weight"] = create_tensor(bp + "attn1.norm_q.weight", {inner_dim});
                params[bp + "attn1.norm_k.weight"] = create_tensor(bp + "attn1.norm_k.weight", {inner_dim});

                // Attention projections
                params[bp + "attn1.to_q.weight"] = create_tensor(bp + "attn1.to_q.weight", {inner_dim, inner_dim});
                params[bp + "attn1.to_q.bias"] = create_tensor(bp + "attn1.to_q.bias", {inner_dim});
                params[bp + "attn1.to_k.weight"] = create_tensor(bp + "attn1.to_k.weight", {inner_dim, inner_dim});
                params[bp + "attn1.to_k.bias"] = create_tensor(bp + "attn1.to_k.bias", {inner_dim});
                params[bp + "attn1.to_v.weight"] = create_tensor(bp + "attn1.to_v.weight", {inner_dim, inner_dim});
                params[bp + "attn1.to_v.bias"] = create_tensor(bp + "attn1.to_v.bias", {inner_dim});
                params[bp + "attn1.to_out.0.weight"] = create_tensor(bp + "attn1.to_out.0.weight", {inner_dim, inner_dim});
                params[bp + "attn1.to_out.0.bias"] = create_tensor(bp + "attn1.to_out.0.bias", {inner_dim});

                // FFN - dimensions are reversed by GGML loader
                // Safetensors: [8960, 1536] -> GGML: [1536, 8960]
                // Safetensors: [1536, 8960] -> GGML: [8960, 1536]
                params[bp + "ffn.net.0.proj.weight"] = create_tensor(bp + "ffn.net.0.proj.weight", {inner_dim, ffn_dim});
                params[bp + "ffn.net.0.proj.bias"] = create_tensor(bp + "ffn.net.0.proj.bias", {ffn_dim});
                params[bp + "ffn.net.2.weight"] = create_tensor(bp + "ffn.net.2.weight", {ffn_dim, inner_dim});
                params[bp + "ffn.net.2.bias"] = create_tensor(bp + "ffn.net.2.bias", {inner_dim});

                // Block scale_shift_table
                // Safetensors [1, 6, 1536] → GGML [1536, 6, 1]
                params[bp + "scale_shift_table"] = create_tensor(bp + "scale_shift_table", {inner_dim, 6, 1});
            }
            
            LOG_DEBUG("MinimaxModel::init_params completed successfully - %zu tensors created", params.size());
        }

        struct ggml_tensor* forward(struct ggml_context* ctx,
                                   struct ggml_tensor* input_latents,
                                   struct ggml_tensor* timestep) {
            // MiniMax-Remover forward pass implementation
            
            // 1. Patch embedding - use conv2d with reshaped weight
            // GGML expects conv2d kernel in format: [KW, KH, IC, OC]
            // Original weight from PyTorch: [OC=1536, IC=48, KH=2, KW=2]
            // Need to reshape to: [2, 2, 48, 1536]
            struct ggml_tensor* patch_weight = ggml_reshape_4d(ctx, params["patch_embedding.weight"], 2, 2, 48, 1536);
            struct ggml_tensor* hidden_states = ggml_conv_2d(ctx, patch_weight, input_latents, 2, 2, 0, 0, 1, 1);
            
            // Reshape bias to [1, 1, inner_dim, 1] for broadcasting with [H, W, C, N]
            struct ggml_tensor* bias = ggml_reshape_4d(ctx, params["patch_embedding.bias"], 1, 1, inner_dim, 1);
            hidden_states = ggml_add(ctx, hidden_states, bias);
            
            // 2. Time conditioning
            // Embed timestep to freq_dim (256) using sinusoidal embeddings
            struct ggml_tensor* t_emb = ggml_nn_timestep_embedding(ctx, timestep, freq_dim);  // [N, 256]
            
            // Flatten linear weights to 2D for matrix multiplication
            // Weights are loaded as [256, 1536, 1, 1] but need to be [256, 1536]
            struct ggml_tensor* linear1_weight = ggml_reshape_2d(ctx, 
                params["condition_embedder.time_embedder.linear_1.weight"],
                freq_dim, inner_dim);  // [256, 1536]
            
            t_emb = ggml_mul_mat(ctx, linear1_weight, t_emb);  // Result is 2D: [1536, batch]
            
            // Flatten both output and bias to 1D for addition
            t_emb = ggml_reshape_1d(ctx, t_emb, inner_dim);
            struct ggml_tensor* bias1 = ggml_reshape_1d(ctx, params["condition_embedder.time_embedder.linear_1.bias"], inner_dim);
            t_emb = ggml_add(ctx, t_emb, bias1);
            t_emb = ggml_silu(ctx, t_emb);
            
            struct ggml_tensor* linear2_weight = ggml_reshape_2d(ctx,
                params["condition_embedder.time_embedder.linear_2.weight"],
                inner_dim, inner_dim);  // [1536, 1536]
            
            // Reshape back to 2D for next mul_mat [1536, 1]
            t_emb = ggml_reshape_2d(ctx, t_emb, inner_dim, 1);
            t_emb = ggml_mul_mat(ctx, linear2_weight, t_emb);
            
            t_emb = ggml_reshape_1d(ctx, t_emb, inner_dim);
            struct ggml_tensor* bias2 = ggml_reshape_1d(ctx, params["condition_embedder.time_embedder.linear_2.bias"], inner_dim);
            t_emb = ggml_add(ctx, t_emb, bias2);
            
            // 3. Process through 30 transformer blocks
            for (int i = 0; i < num_layers; i++) {
                char block_prefix[32];
                sprintf(block_prefix, "blocks.%d.", i);
                std::string prefix_str = block_prefix;
                
                // Get modulation parameters
                struct ggml_tensor* scale_shift = params[prefix_str + "scale_shift_table"];
                struct ggml_tensor* shift = ggml_view_1d(ctx, scale_shift, inner_dim, 0);
                struct ggml_tensor* scale = ggml_view_1d(ctx, scale_shift, inner_dim, inner_dim * sizeof(float));
                
                // Reshape scale and shift to [1, 1, inner_dim, 1] for broadcasting with [H, W, C, N]
                shift = ggml_reshape_4d(ctx, shift, 1, 1, inner_dim, 1);
                scale = ggml_reshape_4d(ctx, scale, 1, 1, inner_dim, 1);
                
                // Apply scale-shift modulation
                hidden_states = ggml_add(ctx, 
                    ggml_mul(ctx, hidden_states, scale),
                    shift);
                
                // Attention block
                struct ggml_tensor* norm_q = ggml_rms_norm(ctx, hidden_states, DEFAULT_EPS);
                // Reshape norm weights to [1, 1, inner_dim, 1] for broadcasting
                struct ggml_tensor* norm_q_weight = ggml_reshape_4d(ctx, params[prefix_str + "attn1.norm_q.weight"], 1, 1, inner_dim, 1);
                norm_q = ggml_mul(ctx, norm_q, norm_q_weight);
                
                struct ggml_tensor* norm_k = ggml_rms_norm(ctx, hidden_states, DEFAULT_EPS);
                struct ggml_tensor* norm_k_weight = ggml_reshape_4d(ctx, params[prefix_str + "attn1.norm_k.weight"], 1, 1, inner_dim, 1);
                norm_k = ggml_mul(ctx, norm_k, norm_k_weight);
                
                // Reshape norm_q, norm_k for attention: [H, W, C, N] -> [C, H*W]
                int h = norm_q->ne[0];
                int w = norm_q->ne[1];
                int seq_len = h * w;
                norm_q = ggml_reshape_2d(ctx, norm_q, inner_dim, seq_len);
                norm_k = ggml_reshape_2d(ctx, norm_k, inner_dim, seq_len);
                struct ggml_tensor* hidden_states_2d = ggml_reshape_2d(ctx, hidden_states, inner_dim, seq_len);
                
                // Reshape projection weights to 2D
                struct ggml_tensor* to_q_weight = ggml_reshape_2d(ctx, params[prefix_str + "attn1.to_q.weight"], inner_dim, inner_dim);
                struct ggml_tensor* to_k_weight = ggml_reshape_2d(ctx, params[prefix_str + "attn1.to_k.weight"], inner_dim, inner_dim);
                struct ggml_tensor* to_v_weight = ggml_reshape_2d(ctx, params[prefix_str + "attn1.to_v.weight"], inner_dim, inner_dim);
                
                // Q, K, V projections
                struct ggml_tensor* q = ggml_mul_mat(ctx, to_q_weight, norm_q);
                q = ggml_add(ctx, q, ggml_reshape_1d(ctx, params[prefix_str + "attn1.to_q.bias"], inner_dim));
                
                struct ggml_tensor* k = ggml_mul_mat(ctx, to_k_weight, norm_k);
                k = ggml_add(ctx, k, ggml_reshape_1d(ctx, params[prefix_str + "attn1.to_k.bias"], inner_dim));
                
                struct ggml_tensor* v = ggml_mul_mat(ctx, to_v_weight, hidden_states_2d);
                v = ggml_add(ctx, v, ggml_reshape_1d(ctx, params[prefix_str + "attn1.to_v.bias"], inner_dim));
                
                // Reshape for multi-head attention: [C, seq_len] -> [head_dim, num_heads, seq_len]
                q = ggml_reshape_3d(ctx, q, attention_head_dim, num_attention_heads, seq_len);
                k = ggml_reshape_3d(ctx, k, attention_head_dim, num_attention_heads, seq_len);
                v = ggml_reshape_3d(ctx, v, attention_head_dim, num_attention_heads, seq_len);
                
                // Transpose for attention
                q = ggml_cont(ctx, ggml_transpose(ctx, q));
                k = ggml_cont(ctx, ggml_transpose(ctx, k));
                v = ggml_cont(ctx, ggml_transpose(ctx, v));
                
                // Attention computation
                struct ggml_tensor* attn_weights = ggml_mul_mat(ctx, k, q);
                attn_weights = ggml_scale(ctx, attn_weights, 1.0f / sqrt(attention_head_dim));
                attn_weights = ggml_soft_max(ctx, attn_weights);
                
                struct ggml_tensor* attn_output = ggml_mul_mat(ctx, v, attn_weights);
                attn_output = ggml_cont(ctx, ggml_transpose(ctx, attn_output));
                attn_output = ggml_reshape_2d(ctx, attn_output, inner_dim, seq_len);
                
                // Output projection
                struct ggml_tensor* to_out_weight = ggml_reshape_2d(ctx, params[prefix_str + "attn1.to_out.0.weight"], inner_dim, inner_dim);
                attn_output = ggml_mul_mat(ctx, to_out_weight, attn_output);
                attn_output = ggml_add(ctx, attn_output, ggml_reshape_1d(ctx, params[prefix_str + "attn1.to_out.0.bias"], inner_dim));
                
                // Reshape back to 4D for residual: [C, seq_len] -> [H, W, C, N]
                attn_output = ggml_reshape_4d(ctx, attn_output, h, w, inner_dim, 1);
                
                // Residual connection
                hidden_states = ggml_add(ctx, hidden_states, attn_output);
                
                // FFN block - reshape hidden_states back to 2D
                struct ggml_tensor* ffn_input = ggml_reshape_2d(ctx, hidden_states, inner_dim, seq_len);
                struct ggml_tensor* ffn_weight1 = ggml_reshape_2d(ctx, params[prefix_str + "ffn.net.0.proj.weight"], ffn_dim, inner_dim);
                struct ggml_tensor* ffn_hidden = ggml_mul_mat(ctx, ffn_weight1, ffn_input);
                ffn_hidden = ggml_add(ctx, ffn_hidden, ggml_reshape_1d(ctx, params[prefix_str + "ffn.net.0.proj.bias"], ffn_dim));
                ffn_hidden = ggml_silu(ctx, ffn_hidden);
                
                struct ggml_tensor* ffn_weight2 = ggml_reshape_2d(ctx, params[prefix_str + "ffn.net.2.weight"], inner_dim, ffn_dim);
                ffn_hidden = ggml_mul_mat(ctx, ffn_weight2, ffn_hidden);
                ffn_hidden = ggml_add(ctx, ffn_hidden, ggml_reshape_1d(ctx, params[prefix_str + "ffn.net.2.bias"], inner_dim));
                
                // Reshape back to 4D for residual
                ffn_hidden = ggml_reshape_4d(ctx, ffn_hidden, h, w, inner_dim, 1);
                
                // Residual connection
                hidden_states = ggml_add(ctx, hidden_states, ffn_hidden);
            }
            
            // 4. Final normalization and output projection
            // Note: Model doesn't have norm_out.weight - using RMS norm without scale
            hidden_states = ggml_rms_norm(ctx, hidden_states, DEFAULT_EPS);
            
            // Output projection
            hidden_states = ggml_mul_mat(ctx, params["proj_out.weight"], hidden_states);
            hidden_states = ggml_add(ctx, hidden_states, params["proj_out.bias"]);
            
            return hidden_states;
        }

        void map_by_name(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            tensors[prefix + "patch_embedding.weight"] = params["patch_embedding.weight"];
            tensors[prefix + "patch_embedding.bias"] = params["patch_embedding.bias"];

            tensors[prefix + "condition_embedder.time_embedder.linear_1.weight"] = params["condition_embedder.time_embedder.linear_1.weight"];
            tensors[prefix + "condition_embedder.time_embedder.linear_1.bias"] = params["condition_embedder.time_embedder.linear_1.bias"];
            tensors[prefix + "condition_embedder.time_embedder.linear_2.weight"] = params["condition_embedder.time_embedder.linear_2.weight"];
            tensors[prefix + "condition_embedder.time_embedder.linear_2.bias"] = params["condition_embedder.time_embedder.linear_2.bias"];
            tensors[prefix + "condition_embedder.time_proj.weight"] = params["condition_embedder.time_proj.weight"];
            tensors[prefix + "condition_embedder.time_proj.bias"] = params["condition_embedder.time_proj.bias"];

            tensors[prefix + "proj_out.weight"] = params["proj_out.weight"];
            tensors[prefix + "proj_out.bias"] = params["proj_out.bias"];
            // Note: norm_out.weight doesn't exist in the model
            tensors[prefix + "scale_shift_table"] = params["scale_shift_table"];

            for (int i = 0; i < num_layers; i++) {
                char block_prefix[32];
                sprintf(block_prefix, "blocks.%d.", i);
                std::string bp = block_prefix;

                tensors[prefix + bp + "attn1.norm_q.weight"] = params[bp + "attn1.norm_q.weight"];
                tensors[prefix + bp + "attn1.norm_k.weight"] = params[bp + "attn1.norm_k.weight"];
                tensors[prefix + bp + "attn1.to_q.weight"] = params[bp + "attn1.to_q.weight"];
                tensors[prefix + bp + "attn1.to_q.bias"] = params[bp + "attn1.to_q.bias"];
                tensors[prefix + bp + "attn1.to_k.weight"] = params[bp + "attn1.to_k.weight"];
                tensors[prefix + bp + "attn1.to_k.bias"] = params[bp + "attn1.to_k.bias"];
                tensors[prefix + bp + "attn1.to_v.weight"] = params[bp + "attn1.to_v.weight"];
                tensors[prefix + bp + "attn1.to_v.bias"] = params[bp + "attn1.to_v.bias"];
                tensors[prefix + bp + "attn1.to_out.0.weight"] = params[bp + "attn1.to_out.0.weight"];
                tensors[prefix + bp + "attn1.to_out.0.bias"] = params[bp + "attn1.to_out.0.bias"];

                tensors[prefix + bp + "ffn.net.0.proj.weight"] = params[bp + "ffn.net.0.proj.weight"];
                tensors[prefix + bp + "ffn.net.0.proj.bias"] = params[bp + "ffn.net.0.proj.bias"];
                tensors[prefix + bp + "ffn.net.2.weight"] = params[bp + "ffn.net.2.weight"];
                tensors[prefix + bp + "ffn.net.2.bias"] = params[bp + "ffn.net.2.bias"];

                tensors[prefix + bp + "scale_shift_table"] = params[bp + "scale_shift_table"];
            }
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) {
            map_by_name(tensors, prefix);
        }

        void alloc_params_buffer(ggml_context* params_ctx, ggml_backend_t params_backend, ggml_backend_buffer_t& params_buffer) {
            LOG_DEBUG("MinimaxModel::alloc_params_buffer called");
            size_t num_tensors = ggml_tensor_num(params_ctx);
            LOG_DEBUG("MinimaxModel: Allocating buffer for %zu tensors on backend: %s", 
                     num_tensors, ggml_backend_name(params_backend));
            
            params_buffer = ggml_backend_alloc_ctx_tensors(params_ctx, params_backend);
            
            if (params_buffer == NULL) {
                LOG_ERROR("MinimaxModel failed to allocate params buffer for %zu tensors", num_tensors);
                return;
            }
            
            size_t params_buffer_size = ggml_backend_buffer_get_size(params_buffer);
            LOG_INFO("MinimaxModel params buffer allocated: %.2f MB (%s) (%zu tensors)",
                    params_buffer_size / (1024.f * 1024.f),
                    ggml_backend_is_cpu(params_backend) ? "RAM" : "VRAM",
                    num_tensors);
        }
        
        void free_params_buffer(ggml_backend_buffer_t& params_buffer) {
            if (params_buffer != NULL) {
                ggml_backend_buffer_free(params_buffer);
                params_buffer = NULL;
            }
        }
        void free_compute_buffer() {}
        size_t get_params_buffer_size() { return 0; }
        struct ggml_tensor* compute(const int n_threads,
                                   struct ggml_tensor* x,
                                   struct ggml_tensor* timesteps,
                                   struct ggml_tensor* context,
                                   struct ggml_tensor* y,
                                   struct ggml_tensor* c_concat,
                                   struct ggml_tensor* control_hint,
                                   struct ggml_tensor* vace_context,
                                   float vace_strength,
                                   struct ggml_tensor** output,
                                   struct ggml_context* output_ctx) {
            // Create a larger compute context for intermediate operations
            struct ggml_init_params compute_params;
            compute_params.mem_size   = static_cast<size_t>(5120) * 1024 * 1024;  // 5 GB for compute operations
            compute_params.mem_buffer = NULL;
            compute_params.no_alloc   = false;  // Allow actual memory allocation
            
            struct ggml_context* compute_ctx = ggml_init(compute_params);
            GGML_ASSERT(compute_ctx != NULL);
            
            // Run forward pass with compute context
            struct ggml_tensor* result = forward(compute_ctx, x, timesteps);
            
            // Copy result to output context
            *output = ggml_dup_tensor(output_ctx, result);
            ggml_backend_tensor_copy(result, *output);
            
            // Free compute context
            ggml_free(compute_ctx);
            
            return *output;
        }
    };

    // ================================== 3D Patch Embedding ==================================
    struct MinimaxRunner : public GGMLRunner {
    public:
        std::string desc = "minimax";
        MinimaxModel minimax;
        SDVersion version;

        MinimaxRunner(ggml_backend_t backend,
                      bool offload_params_to_cpu,
                      const String2GGMLType& tensor_types = {},
                      const std::string prefix            = "",
                      SDVersion version                   = VERSION_MINIMAX_REMOVER,
                      bool flash_attn                     = false)
            : GGMLRunner(backend, offload_params_to_cpu), version(version) {
            LOG_DEBUG("MinimaxRunner constructor started, tensor_types size: %zu", tensor_types.size());
            
            // Count layers from tensor names - MiniMax model doesn't have "unet." prefix
            int num_layers = 0;
            for (auto pair : tensor_types) {
                std::string tensor_name = pair.first;
                // Skip tensors that don't match the prefix (only if prefix is not empty)
                if (prefix.size() > 0 && tensor_name.find(prefix) == std::string::npos)
                    continue;
                // Look for blocks.N. pattern to detect layers
                size_t pos = tensor_name.find("blocks.");
                if (pos != std::string::npos) {
                    // Extract the layer number: "blocks.N.something" -> get N
                    std::string after_blocks = tensor_name.substr(pos + 7); // Skip "blocks."
                    size_t dot_pos = after_blocks.find('.');
                    if (dot_pos != std::string::npos) {
                        std::string layer_num_str = after_blocks.substr(0, dot_pos);
                        int block_index = atoi(layer_num_str.c_str());
                        if (block_index + 1 > num_layers) {
                            num_layers = block_index + 1;
                        }
                    }
                }
            }
            
            // Default to 30 layers if detection failed
            if (num_layers == 0) {
                LOG_WARN("Could not detect number of layers, defaulting to 30");
                num_layers = 30;
            }
            
            LOG_INFO("Detected %d transformer layers", num_layers);
            LOG_DEBUG("About to initialize MinimaxModel");

            // Initialize minimax model with detected number of layers
            minimax = MinimaxModel({1, 2, 2}, 32, 48, 48, 64, 256, 8960, num_layers);
            LOG_DEBUG("MinimaxModel created, about to init_params");
            
            minimax.init_params(params_ctx, tensor_types, prefix);
            LOG_DEBUG("init_params completed successfully");

            desc = "MiniMax-Remover";
            LOG_DEBUG("MinimaxRunner constructor completed");
        }

        std::string get_desc() {
            return desc;
        }

        void alloc_params_buffer() {
            minimax.alloc_params_buffer(params_ctx, params_backend, params_buffer);
        }

        void free_params_buffer() {
            minimax.free_params_buffer(params_buffer);
        }

        void free_compute_buffer() {
            minimax.free_compute_buffer();
        }

        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors) {
            minimax.get_param_tensors(tensors, "");
        }

        size_t get_params_buffer_size() {
            return minimax.get_params_buffer_size();
        }

        int64_t get_adm_in_channels() {
            return 0;  // MiniMax-Remover doesn't use ADM conditioning
        }

        void compute(const int n_threads,
                     struct ggml_tensor* x,
                     struct ggml_tensor* timesteps,
                     struct ggml_tensor* context,
                     struct ggml_tensor* y,
                     struct ggml_tensor* c_concat,
                     struct ggml_tensor* control_hint,
                     struct ggml_tensor* vace_context,
                     float vace_strength,
                     struct ggml_tensor** output,
                     struct ggml_context* output_ctx) {
            minimax.compute(n_threads, x, timesteps, context, y, c_concat, control_hint, vace_context, vace_strength, output, output_ctx);
        }
    };

    struct MinimaxVAERunner : public WAN::WanVAERunner {
        MinimaxVAERunner(ggml_backend_t backend,
                         bool offload_params_to_cpu,
                         const String2GGMLType& tensor_types = {},
                         const std::string prefix            = "",
                         bool decode_only                    = true)
            : WAN::WanVAERunner(backend,
                                offload_params_to_cpu,
                                tensor_types,
                                prefix,
                                decode_only,
                                VERSION_WAN2) {}

        std::string get_desc() {
            return "minimax_vae";
        }

    private:
        struct UpSampleMapping {
            int block_idx   = 0;
            int resnet_idx  = 0;
            bool is_resample = false;
        };

        static bool starts_with(const std::string& value, const std::string& prefix) {
            return value.rfind(prefix, 0) == 0;
        }

        static std::map<int, UpSampleMapping> build_up_block_mapping(const std::map<std::string, struct ggml_tensor*>& base) {
            const std::string ups_prefix = "decoder.upsamples.";
            struct IndexInfo {
                bool is_resample = false;
            };

            std::map<int, IndexInfo> index_info;

            for (const auto& entry : base) {
                if (!starts_with(entry.first, ups_prefix)) {
                    continue;
                }

                const size_t pos = ups_prefix.size();
                const size_t dot = entry.first.find('.', pos);
                if (dot == std::string::npos) {
                    continue;
                }

                int index = std::stoi(entry.first.substr(pos, dot - pos));
                std::string suffix = entry.first.substr(dot + 1);

                if (starts_with(suffix, "resample.") || starts_with(suffix, "time_conv.")) {
                    index_info[index].is_resample = true;
                } else {
                    // ensure entry exists even if not resample
                    (void)index_info[index];
                }
            }

            std::map<int, UpSampleMapping> result;
            int block_idx  = 0;
            int resnet_idx = 0;

            for (const auto& info_pair : index_info) {
                const int index        = info_pair.first;
                const bool is_resample = info_pair.second.is_resample;

                if (is_resample) {
                    result[index] = {block_idx, 0, true};
                    block_idx++;
                    resnet_idx = 0;
                } else {
                    if (resnet_idx == 3) {
                        block_idx++;
                        resnet_idx = 0;
                    }
                    result[index] = {block_idx, resnet_idx, false};
                    resnet_idx++;
                }
            }

            return result;
        }

        static std::string map_residual_component(const std::string& suffix) {
            const std::string residual_prefix = "residual.";
            if (starts_with(suffix, residual_prefix)) {
                std::string rest = suffix.substr(residual_prefix.size());
                if (starts_with(rest, "0.")) {
                    return "norm1." + rest.substr(2);
                }
                if (starts_with(rest, "2.")) {
                    return "conv1." + rest.substr(2);
                }
                if (starts_with(rest, "3.")) {
                    return "norm2." + rest.substr(2);
                }
                if (starts_with(rest, "6.")) {
                    return "conv2." + rest.substr(2);
                }
                return "";
            }

            const std::string shortcut_prefix = "shortcut.";
            if (starts_with(suffix, shortcut_prefix)) {
                return "conv_shortcut." + suffix.substr(shortcut_prefix.size());
            }

            return "";
        }

        static std::string map_encoder_name(const std::string& name) {
            const std::string encoder_prefix = "encoder.";
            if (!starts_with(name, encoder_prefix)) {
                return "";
            }

            std::string rest = name.substr(encoder_prefix.size());

            if (starts_with(rest, "conv1.")) {
                return "encoder.conv_in." + rest.substr(6);
            }
            if (starts_with(rest, "conv2.")) {
                return "encoder.conv_out." + rest.substr(6);
            }
            if (starts_with(rest, "head.0.")) {
                return "encoder.norm_out." + rest.substr(7);
            }
            if (starts_with(rest, "head.2.")) {
                return "encoder.conv_out." + rest.substr(7);
            }

            if (starts_with(rest, "middle.0.")) {
                std::string mapped = map_residual_component(rest.substr(9));
                if (mapped.empty()) {
                    return "";
                }
                return "encoder.mid_block.resnets.0." + mapped;
            }
            if (starts_with(rest, "middle.1.")) {
                return "encoder.mid_block.attentions.0." + rest.substr(9);
            }
            if (starts_with(rest, "middle.2.")) {
                std::string mapped = map_residual_component(rest.substr(9));
                if (mapped.empty()) {
                    return "";
                }
                return "encoder.mid_block.resnets.1." + mapped;
            }

            const std::string downs_prefix = "downsamples.";
            if (starts_with(rest, downs_prefix)) {
                size_t idx_start = downs_prefix.size();
                size_t dot       = rest.find('.', idx_start);
                if (dot == std::string::npos) {
                    return "";
                }

                int index = std::stoi(rest.substr(idx_start, dot - idx_start));
                std::string suffix = rest.substr(dot + 1);

                if (starts_with(suffix, "residual.") || starts_with(suffix, "shortcut.")) {
                    std::string mapped = map_residual_component(suffix);
                    if (mapped.empty()) {
                        return "";
                    }
                    return "encoder.down_blocks." + std::to_string(index) + "." + mapped;
                }
                if (starts_with(suffix, "resample.") || starts_with(suffix, "time_conv.")) {
                    return "encoder.down_blocks." + std::to_string(index) + "." + suffix;
                }
            }

            return "";
        }

        static std::string map_decoder_name(const std::string& name,
                                            const std::map<int, UpSampleMapping>& up_map) {
            const std::string decoder_prefix = "decoder.";
            if (!starts_with(name, decoder_prefix)) {
                return "";
            }

            std::string rest = name.substr(decoder_prefix.size());

            if (starts_with(rest, "conv1.")) {
                return "decoder.conv_in." + rest.substr(6);
            }
            if (starts_with(rest, "head.0.")) {
                return "decoder.norm_out." + rest.substr(7);
            }
            if (starts_with(rest, "head.2.")) {
                return "decoder.conv_out." + rest.substr(7);
            }

            if (starts_with(rest, "middle.0.")) {
                std::string mapped = map_residual_component(rest.substr(9));
                if (mapped.empty()) {
                    return "";
                }
                return "decoder.mid_block.resnets.0." + mapped;
            }
            if (starts_with(rest, "middle.1.")) {
                return "decoder.mid_block.attentions.0." + rest.substr(9);
            }
            if (starts_with(rest, "middle.2.")) {
                std::string mapped = map_residual_component(rest.substr(9));
                if (mapped.empty()) {
                    return "";
                }
                return "decoder.mid_block.resnets.1." + mapped;
            }

            const std::string ups_prefix = "upsamples.";
            if (starts_with(rest, ups_prefix)) {
                size_t idx_start = ups_prefix.size();
                size_t dot       = rest.find('.', idx_start);
                if (dot == std::string::npos) {
                    return "";
                }

                int index = std::stoi(rest.substr(idx_start, dot - idx_start));
                auto map_it = up_map.find(index);
                if (map_it == up_map.end()) {
                    return "";
                }

                std::string suffix = rest.substr(dot + 1);
                const auto& mapping = map_it->second;

                if (mapping.is_resample) {
                    if (starts_with(suffix, "resample.") || starts_with(suffix, "time_conv.")) {
                        return "decoder.up_blocks." + std::to_string(mapping.block_idx) + ".upsamplers.0." + suffix;
                    }
                    return "";
                }

                std::string mapped = map_residual_component(suffix);
                if (mapped.empty()) {
                    return "";
                }
                return "decoder.up_blocks." + std::to_string(mapping.block_idx) + ".resnets." +
                       std::to_string(mapping.resnet_idx) + "." + mapped;
            }

            return "";
        }

    public:
        void get_param_tensors(std::map<std::string, struct ggml_tensor*>& tensors, const std::string prefix) override {
            std::map<std::string, struct ggml_tensor*> base;
            ae.get_param_tensors(base, "");

            LOG_INFO("Minimax base tensor count: %zu", base.size());

            auto up_map = build_up_block_mapping(base);

            const bool has_prefix = !prefix.empty();
            const std::string loader_prefix = "first_stage_model";
            static bool logged_base_names = false;

            for (const auto& entry : base) {
                if (!logged_base_names) {
                    LOG_INFO("Minimax base tensor: %s", entry.first.c_str());
                }

                std::vector<std::string> mapped_names;

                if (starts_with(entry.first, "conv1.")) {
                    mapped_names.push_back("quant_conv." + entry.first.substr(6));
                } else if (starts_with(entry.first, "conv2.")) {
                    mapped_names.push_back("post_quant_conv." + entry.first.substr(6));
                } else {
                    std::string decoder_mapped = map_decoder_name(entry.first, up_map);
                    if (!decoder_mapped.empty()) {
                        mapped_names.push_back(decoder_mapped);
                    }

                    std::string encoder_mapped = map_encoder_name(entry.first);
                    if (!encoder_mapped.empty()) {
                        mapped_names.push_back(encoder_mapped);
                    }
                }

                std::vector<std::string> target_names;
                if (!mapped_names.empty()) {
                    target_names = mapped_names;
                } else {
                    target_names.push_back(entry.first);
                }

                std::string canonical_name = has_prefix ? prefix + "." + entry.first : entry.first;
                tensors[canonical_name]     = entry.second;

                std::unordered_set<std::string> seen_targets;
                for (const auto& target_name : target_names) {
                    if (target_name.empty() || !seen_targets.insert(target_name).second) {
                        continue;
                    }

                    tensors[target_name] = entry.second;

                    std::string loader_target = loader_prefix.empty() ? target_name : loader_prefix + "." + target_name;
                    tensors[loader_target]    = entry.second;
                    if (loader_target != target_name) {
                        TensorAliasRegistry::register_alias(target_name, loader_target);
                    }

                    if (has_prefix) {
                        std::string prefixed_target = prefix + "." + target_name;
                        if (prefixed_target != canonical_name) {
                            tensors[prefixed_target] = entry.second;
                            TensorAliasRegistry::register_alias(prefixed_target, loader_target);
                        }
                        LOG_INFO("Minimax mapped tensor: %s -> %s", entry.first.c_str(), prefixed_target.c_str());
                    } else {
                        LOG_INFO("Minimax mapped tensor: %s -> %s", entry.first.c_str(), target_name.c_str());
                    }
                }

                const std::string& primary_target = target_names.front();
                std::string loader_primary = loader_prefix.empty() ? primary_target : loader_prefix + "." + primary_target;
                if (canonical_name != loader_primary) {
                    TensorAliasRegistry::register_alias(canonical_name, loader_primary);
                }
            }

            logged_base_names = true;
        }
    };

} // namespace MINIMAX

#endif // __MINI_MAX_REMOVER_HPP__