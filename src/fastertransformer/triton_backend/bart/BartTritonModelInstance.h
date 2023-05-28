/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "src/fastertransformer/models/bart/BartDecoding.h"
#include "src/fastertransformer/models/bart/BartEncoder.h"
#include "src/fastertransformer/triton_backend/bart/BartTritonModel.h"
#include "src/fastertransformer/triton_backend/transformer_triton_backend.hpp"
#include <memory>

namespace ft = fastertransformer;

template<typename T>
struct T5TritonModelInstance: AbstractTransformerModelInstance {

    T5TritonModelInstance(std::unique_ptr<ft::BartEncoder<T>>                     Bart_encoder,
                          std::unique_ptr<ft::BartDecoding<T>>                    Bart_decoding,
                          std::shared_ptr<ft::BartEncoderWeight<T>>               Bart_encoder_weight,
                          std::shared_ptr<ft::BartDecodingWeight<T>>              Bart_decoding_weight,
                          std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator,
                          std::unique_ptr<ft::cublasAlgoMap>                      cublas_algo_map,
                          std::unique_ptr<std::mutex>                             cublas_wrapper_mutex,
                          std::unique_ptr<ft::cublasMMWrapper>                    cublas_wrapper,
                          std::unique_ptr<cudaDeviceProp>                         cuda_device_prop_ptr);
    ~T5TritonModelInstance();

    std::shared_ptr<std::vector<triton::Tensor>>
    forward(std::shared_ptr<std::vector<triton::Tensor>> input_tensors) override
    {
        ft::FT_CHECK(false);
        return nullptr;
    };

    std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
    forward(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors) override;

    static std::shared_ptr<std::unordered_map<std::string, triton::Tensor>>
    convert_outputs(ft::TensorMap& output_tensors);

private:
    const std::unique_ptr<ft::BartEncoder<T>>                     bart_encoder_;
    const std::shared_ptr<ft::BartEncoderWeight<T>>               bart_encoder_weight_;
    const std::unique_ptr<ft::BartDecoding<T>>                    bart_decoding_;
    const std::shared_ptr<ft::BartDecodingWeight<T>>              bart_decoding_weight_;
    const std::unique_ptr<ft::Allocator<ft::AllocatorType::CUDA>> allocator_;
    const std::unique_ptr<ft::cublasAlgoMap>                      cublas_algo_map_;
    const std::unique_ptr<std::mutex>                             cublas_wrapper_mutex_;
    const std::unique_ptr<ft::cublasMMWrapper>                    cublas_wrapper_;
    const std::unique_ptr<cudaDeviceProp>                         cuda_device_prop_ptr_;

    ft::TensorMap convert_inputs(std::shared_ptr<std::unordered_map<std::string, triton::Tensor>> input_tensors);

    void allocateBuffer(const size_t request_batch_size,
                        const size_t beam_width,
                        const size_t max_output_len,
                        const size_t mem_max_seq_len);
    void freeBuffer();

    int*   d_input_ids_        = nullptr;
    int*   d_input_lengths_    = nullptr;
    int*   d_input_bad_words_  = nullptr;
    int*   d_input_stop_words_ = nullptr;
    float* d_top_p_decay_      = nullptr;
    float* d_top_p_min_        = nullptr;
    int*   d_top_p_reset_ids_  = nullptr;

    T*     d_encoder_outputs_  = nullptr;
    int*   d_output_ids_       = nullptr;
    int*   d_sequence_lengths_ = nullptr;
    float* d_output_log_probs_ = nullptr;
    float* d_cum_log_probs_    = nullptr;

    int h_total_output_len_;

    std::exception_ptr h_exception_ = nullptr;
};