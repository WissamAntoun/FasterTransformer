# Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import configparser
import logging
import multiprocessing
import os
import sys
from datetime import datetime
from pathlib import Path

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path + "/../../../../3rdparty/transformers/src/")

import numpy as np
import torch  # pytype: disable=import-error
from transformers import BartForConditionalGeneration, MBartForConditionalGeneration

LOGGER = logging.getLogger(__name__)

rename_mapping = {
    # "relative_attention_num_buckets": "relative_attention_num_buckets_or_max_pos_seq_len"
}
new_configs = {
    "structure": {
        "bart_with_bias": "true",  # TODO: this was true for AraBART
        "use_gated_activation": "false",  # TODO: this was false for AraBART
        "position_embedding_type": "absolute",
    }
}


def get_weight_data_type(data_type):
    if data_type == "fp32":
        return np.float32
    elif data_type == "fp16":
        return np.float16
    else:
        assert False, f"Invalid weight data type {data_type}"


def fuse_decoder_qkv(model, factor, saved_dir, np_weight_data_type):
    model_dict = {}
    for name, param in model.named_parameters():
        if name.find("decoder") != -1 and name.find("self_attn") != -1:
            model_dict[name.replace("model.", "")] = param

    for i in range(model.model.decoder.config.decoder_layers):
        shape = model_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"].T.shape
        qkv = torch.cat(
            [
                model_dict[f"decoder.layers.{i}.self_attn.q_proj.weight"].T,
                model_dict[f"decoder.layers.{i}.self_attn.k_proj.weight"].T,
                model_dict[f"decoder.layers.{i}.self_attn.v_proj.weight"].T,
            ],
            dim=-1,
        )

        qkv = qkv.reshape([shape[0], 3, shape[1]])
        qkv = qkv.cpu().detach().numpy().astype(np_weight_data_type)

        split_vals = np.split(qkv, factor, axis=-1)
        for j in range(factor):
            saved_path = (
                saved_dir / f"decoder.layers.{i}.self_attn.qkv_proj.weight.{j}.bin"
            )
            split_vals[j].tofile(saved_path.as_posix())

        qkv_bias = torch.cat(
            [
                model_dict[f"decoder.layers.{i}.self_attn.q_proj.bias"],
                model_dict[f"decoder.layers.{i}.self_attn.k_proj.bias"],
                model_dict[f"decoder.layers.{i}.self_attn.v_proj.bias"],
            ],
            dim=-1,
        )
        qkv_bias = qkv_bias.cpu().detach().numpy().astype(np_weight_data_type)
        split_vals = np.split(qkv_bias, factor, axis=-1)
        for j in range(factor):
            saved_path = (
                saved_dir / f"decoder.layers.{i}.self_attn.qkv_proj.bias.{j}.bin"
            )
            split_vals[j].tofile(saved_path.as_posix())


def split_and_convert_process(key, val, factor, saved_dir, config):
    if val.ndim == 2:
        val = val.transpose(1, 0)
    saved_key = key.replace("model.", "")
    LOGGER.debug(f"key: {key}, val.shape: {val.shape}")

    if key.find("shared.weight") != -1:
        # shared weights, only need to convert the weights of rank 0
        # TODO: check for embedding scaling bart/utils/ft_decoding.py#L346
        embedding_scale = np.sqrt(config.d_model) if config.scale_embedding else 1.0
        val = val * embedding_scale
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path.as_posix())

        saved_path = saved_dir / f"{saved_key}_T.bin"
        val.T.tofile(saved_path.as_posix())
    elif key.find("final_logits_bias") != -1:
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path.as_posix())

        saved_path = saved_dir / f"{saved_key}_T.bin"
        val.T.tofile(saved_path.as_posix())
    elif key.find("embed_positions.weight") != -1:
        # remove the first two ids since size is 1024.
        # Check MBartLearnedPositionalEmbedding in transformers
        saved_path = saved_dir / f"{saved_key}.bin"
        val[:, 2:].tofile(saved_path.as_posix())

        saved_path = saved_dir / f"{saved_key}_T.bin"
        val[:, 2:].T.tofile(saved_path.as_posix())
    elif key.find("lm_head.weight") != -1:
        # lm_head weights, only need to convert the weights of rank 0
        val = val.transpose(
            1, 0
        )  # For lm_head, we use TN gemm to compute, so we don't need to transpose
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path.as_posix())

    elif (
        key.find("layer_norm.weight") != -1
        or key.find("layer_norm.bias") != -1
        or key.find("layernorm_embedding.weight") != -1
        or key.find("layernorm_embedding.bias") != -1
    ):
        # shared weights, only need to convert the weights of rank 0
        saved_path = saved_dir / f"{saved_key}.bin"
        val.tofile(saved_path.as_posix())

    elif (
        key.find("self_attn.out_proj.weight") != -1
        or key.find("encoder_attn.out_proj.weight") != -1
        or key.find("self_attn.out_proj.bias") != -1
        or key.find("encoder_attn.out_proj.bias") != -1
        or key.find("fc2.weight") != -1
        or key.find("fc2.bias") != -1
    ):
        split_vals = np.split(val, factor, axis=0)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif (
        key.find("fc1.weight") != -1
        or key.find("fc1.bias") != -1
        or (
            key.find("encoder") != -1
            and (
                key.find("self_attn.q_proj.weight") != -1
                or key.find("self_attn.k_proj.weight") != -1
                or key.find("self_attn.v_proj.weight") != -1
                or key.find("self_attn.q_proj.bias") != -1
                or key.find("self_attn.k_proj.bias") != -1
                or key.find("self_attn.v_proj.bias") != -1
            )
        )
        or key.find("encoder_attn.q_proj.weight") != -1
        or key.find("encoder_attn.k_proj.weight") != -1
        or key.find("encoder_attn.v_proj.weight") != -1
        or key.find("encoder_attn.q_proj.bias") != -1
        or key.find("encoder_attn.k_proj.bias") != -1
        or key.find("encoder_attn.v_proj.bias") != -1
    ):
        split_vals = np.split(val, factor, axis=-1)
        for j in range(factor):
            saved_path = saved_dir / f"{saved_key}.{j:d}.bin"
            split_vals[j].tofile(saved_path.as_posix())

    elif key.find("decoder") != -1 and (
        key.find("self_attn.q_proj.weight") != -1
        or key.find("self_attn.k_proj.weight") != -1
        or key.find("self_attn.v_proj.weight") != -1
        or key.find("self_attn.q_proj.bias") != -1
        or key.find("self_attn.k_proj.bias") != -1
        or key.find("self_attn.v_proj.bias") != -1
    ):
        pass
    elif (
        key.find("encoder.embed_tokens.weight") != -1
        or key.find("decoder.embed_tokens.weight") != -1
    ):
        LOGGER.warning(f"Not save {key}, using shared.weight directly.")
    else:
        LOGGER.warning(f"cannot find key '{key}' with shape {val.shape}")


def convert_checkpoint(args):
    saved_dir = Path(args.saved_dir) / f"{args.inference_tensor_para_size:d}-gpu"
    saved_dir.mkdir(parents=True, exist_ok=True)

    if args.is_mbart:
        bart_model = MBartForConditionalGeneration.from_pretrained(args.in_file)
    else:
        bart_model = BartForConditionalGeneration.from_pretrained(args.in_file)

    config = configparser.ConfigParser()

    config["encoder"] = {}
    for key, val in bart_model.model.encoder.config.to_dict().items():
        config["encoder"][key] = f"{val}"
    config["encoder"]["weight_data_type"] = args.weight_data_type
    config["decoder"] = {}

    for key, val in bart_model.model.decoder.config.to_dict().items():
        config["decoder"][key] = f"{val}"
    config["decoder"]["weight_data_type"] = args.weight_data_type

    for key, val in rename_mapping.items():
        config["encoder"][val] = config["encoder"].pop(key)
        config["decoder"][val] = config["decoder"].pop(key)

    for key, val in new_configs.items():
        config[key] = {}
        for val_key, val_val in val.items():
            config[key][val_key] = val_val
    with open((saved_dir / f"config.ini").as_posix(), "w") as configfile:
        config.write(configfile)
    np_weight_data_type = get_weight_data_type(args.weight_data_type)

    i_gpu_num = args.inference_tensor_para_size

    # pool = multiprocessing.Pool(args.processes)
    # pool.starmap_async(
    #     split_and_convert_process,
    #     [
    #         (
    #             name,
    #             param.cpu().detach().numpy().astype(np_weight_data_type),
    #             i_gpu_num,
    #             saved_dir,
    #         )
    #         for name, param in bart_model.state_dict().items()
    #     ],
    # )

    # pool.close()
    # pool.join()

    for name, param in bart_model.state_dict().items():
        split_and_convert_process(
            name,
            param.cpu().detach().numpy().astype(np_weight_data_type),
            i_gpu_num,
            saved_dir,
            bart_model.model.config,
        )

    fuse_decoder_qkv(bart_model, i_gpu_num, saved_dir, np_weight_data_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "-saved_dir", "-o", type=str, help="file name of output file", required=True
    )
    parser.add_argument(
        "-in_file",
        "-i",
        type=str,
        help="file name of input checkpoint file",
        required=True,
    )
    parser.add_argument(
        "-inference_tensor_para_size",
        "-i_g",
        type=int,
        help="How many gpus for inference",
        required=True,
    )
    parser.add_argument(
        "-processes",
        "-p",
        type=int,
        help="How many processes to spawn for conversion (default: 4)",
        default=4,
    )
    parser.add_argument(
        "-weight_data_type", type=str, default="fp32", choices=["fp32", "fp16"]
    )
    parser.add_argument(
        "--is_mbart", action="store_true", help="Whether the model is mbart"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Provide verbose messages"
    )
    args = parser.parse_args()
    log_format = "%(asctime)s %(name)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format=log_format
    )
    LOGGER.info("\n=============== Argument ===============")
    for key in vars(args):
        LOGGER.info(f"{key}: {vars(args)[key]}")
    LOGGER.info("========================================")

    start_time = datetime.now()
    convert_checkpoint(args)
    stop_time = datetime.now()
    run_time = stop_time - start_time
    LOGGER.info("Spend {} (h:m:s) to convert the model".format(run_time))
