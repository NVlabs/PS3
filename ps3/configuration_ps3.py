# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from typing import Union
from packaging import version

import transformers
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging


logger = logging.get_logger(__name__)


class PS3VisionConfig(PretrainedConfig):

    model_type = "ps3_vision_model"
    base_config_key = "vision_config"

    def __init__(
        self,
        # timm model args
        model_name: str = None,
        hidden_size: int = 1152,
        pool: str = 'avg',
        drop_path: float = None,
        patch_drop: float = None,
        pretrained: bool = False,
        dynamic_img_size: bool = True,
        # ps3 args
        ps3_scales: list[int] = [378, 756, 1512], 
        select_based_on_layer: list[int] = [0, 9, 18, 26], 
        max_select_num: int = 1280, 
        max_select_num_each_scale: list[int] = None, 
        separate_pos_emb: bool = True, 
        highres_selection_feature: bool = True, 
        highres_selection_module_hidden_dim: int = 512,
        highres_selection_module_out_dim: int = 512,
        highres_selection_module_depth: int = 3,
        highres_selection_module_kernel_size: int = 28,
        # radio args
        radio: bool = False,
        radio_adapter_mlp_version: str = None,
        radio_adapter_mlp_input_dim: int = None,
        radio_adapter_mlp_hidden_dim: int = None,
        radio_adapter_mlp_output_dim: int = None,
        radio_adapter_mlp_num_inner: int = None,
        img_size: int = None,
        drop: float = 0.0,
        class_token: bool = None,
        final_norm: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model_name = model_name
        self.hidden_size = hidden_size
        self.pool = pool
        self.drop_path = drop_path
        self.patch_drop = patch_drop
        self.pretrained = pretrained
        self.dynamic_img_size = dynamic_img_size
        self.ps3_scales = ps3_scales
        self.select_based_on_layer = select_based_on_layer
        self.max_select_num = max_select_num
        self.max_select_num_each_scale = max_select_num_each_scale
        self.separate_pos_emb = separate_pos_emb
        self.highres_selection_feature = highres_selection_feature
        self.highres_selection_module_hidden_dim = highres_selection_module_hidden_dim
        self.highres_selection_module_out_dim = highres_selection_module_out_dim
        self.highres_selection_module_depth = highres_selection_module_depth
        self.highres_selection_module_kernel_size = highres_selection_module_kernel_size
        self.radio = radio
        self.radio_adapter_mlp_version = radio_adapter_mlp_version
        self.radio_adapter_mlp_input_dim = radio_adapter_mlp_input_dim
        self.radio_adapter_mlp_hidden_dim = radio_adapter_mlp_hidden_dim
        self.radio_adapter_mlp_output_dim = radio_adapter_mlp_output_dim
        self.radio_adapter_mlp_num_inner = radio_adapter_mlp_num_inner
        self.img_size = img_size
        self.drop = drop
        self.class_token = class_token
        self.final_norm = final_norm

        # Dummy config to make vila training code happy
        self.vision_tower_name = model_name
        self.image_size = ps3_scales[-1]
        self.patch_size = 14
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        if version.parse(version.parse(transformers.__version__).base_version) >= version.parse("4.47.0"):
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the vision config dict if we are loading from PS3Config
        if config_dict.get("model_type") == "ps3":
            config_dict = config_dict["vision_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)
        

class PS3TextConfig(PretrainedConfig):

    model_type = "ps3_text_model"
    base_config_key = "text_config"

    def __init__(
        self, 
        output_dim: int = 1152,
        prompt_proj_dim: int = 1152,
        context_length: int = 77,
        vocab_size: int = 49408,
        hf_tokenizer_name: str = None,
        tokenizer_kwargs: dict = None,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,  # layer scale initial value
        embed_cls: bool = False,
        pad_id: int = 0,
        no_causal_mask: bool = False,  # disable causal masking
        final_ln_after_pool: bool = False,  # apply final LayerNorm after pooling
        pool_type: str = 'argmax',
        proj_bias: bool = False,
        output_tokens: bool = False,
        act_kwargs: dict = {},
        norm_kwargs: dict = {},
        **kwargs
    ):
        super().__init__(**kwargs)

        self.output_dim = output_dim
        self.prompt_proj_dim = prompt_proj_dim
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.hf_tokenizer_name = hf_tokenizer_name
        self.tokenizer_kwargs = tokenizer_kwargs
        self.width = width
        self.heads = heads
        self.layers = layers
        self.mlp_ratio = mlp_ratio
        self.ls_init_value = ls_init_value
        self.embed_cls = embed_cls
        self.pad_id = pad_id
        self.no_causal_mask = no_causal_mask
        self.final_ln_after_pool = final_ln_after_pool
        self.pool_type = pool_type
        self.proj_bias = proj_bias
        self.output_tokens = output_tokens
        self.act_kwargs = act_kwargs
        self.norm_kwargs = norm_kwargs
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        if version.parse(version.parse(transformers.__version__).base_version) >= version.parse("4.47.0"):
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        cls._set_token_in_kwargs(kwargs)

        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # get the text config dict if we are loading from PS3Config
        if config_dict.get("model_type") == "ps3":
            config_dict = config_dict["text_config"]

        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        return cls.from_dict(config_dict, **kwargs)


class PS3Config(PretrainedConfig):

    model_type = "ps3"
    sub_configs = {"text_config": PS3TextConfig, "vision_config": PS3VisionConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)

        if text_config is None:
            text_config = {}
            logger.info("`text_config` is `None`. Initializing the `PS3TextConfig` with default values.")

        if vision_config is None:
            vision_config = {}
            logger.info("`vision_config` is `None`. initializing the `PS3VisionConfig` with default values.")

        self.text_config = PS3TextConfig(**text_config)
        self.vision_config = PS3VisionConfig(**vision_config)