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

import math
import warnings
import types
from dataclasses import dataclass
from functools import partial
from typing import Optional
from copy import deepcopy
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.jit import Final
from einops import rearrange

import timm
from timm.layers import Mlp, DropPath, use_fused_attn, LayerNorm2d, PatchEmbed, resample_abs_pos_embed, AttentionPoolLatent
from timm.layers.trace_utils import _assert
from timm.layers.format import Format, nchw_to
from timm.models.vision_transformer import LayerScale
from timm.models.convnext import ConvNeXtBlock

from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithNoAttention

from .modeling_ps3_text import TextTransformer
from .configuration_ps3 import PS3VisionConfig, PS3TextConfig, PS3Config


@dataclass
class PS3VisionModelOutput(BaseModelOutputWithNoAttention):
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    pooled_output: Optional[torch.FloatTensor] = None
    selection_maps: Optional[Tuple[torch.LongTensor, ...]] = None
    selection_probs: Optional[Tuple[torch.FloatTensor, ...]] = None
    bottomup_selection_probs: Optional[Tuple[torch.FloatTensor, ...]] = None
    topdown_selection_probs: Optional[Tuple[torch.FloatTensor, ...]] = None


class PS3VisionEncoder(nn.Module):

    def __init__(
        self, 
        config: PS3VisionConfig,
        **kwargs
    ):
        super().__init__()

        self.config = config

        #TODO: support gradient checkpointing
        self.gradient_checkpointing = False


        ###################################################
        ####### Vit backbone initialize thought timm ######
        ###################################################

        # setup kwargs that may not be common across all models
        timm_kwargs = {}
        if config.drop_path is not None:
            timm_kwargs['drop_path_rate'] = config.drop_path
        if config.patch_drop is not None:
            timm_kwargs['patch_drop_rate'] = config.patch_drop
        if config.dynamic_img_size is not None:
            timm_kwargs['dynamic_img_size'] = config.dynamic_img_size
        if config.img_size is not None:
            timm_kwargs['img_size'] = config.img_size
        if config.class_token is not None:
            timm_kwargs['class_token'] = config.class_token

        # Create the vision transformer backbone using timm
        self.trunk = timm.create_model(
            config.model_name,
            num_classes=0,
            global_pool=config.pool,
            pretrained=config.pretrained,
            block_fn=Block_w_KVCache,
            embed_layer=SelectedPatchEmbed,
            **timm_kwargs,
        )
        
        # for vila to easily access the embed_dim and num_layers
        self.width = self.trunk.num_features
        self.layers = len(self.trunk.blocks)
        self.patch_size = self.trunk.patch_embed.patch_size[0]


        ###################################################
        ############## ps3 model initialize ###############
        ###################################################

        # Customization for RADIO
        self.trunk.radio = config.radio
        if config.radio:
            try:
                from .utils_radio_adapter_mlp import create_mlp_from_config
            except ImportError:
                raise ImportError("Please import the create_mlp_from_config function from https://github.com/NVlabs/RADIO/blob/main/radio/adaptor_mlp.py.")
            # Remove the final norm
            if hasattr(self.trunk, "norm") and not config.final_norm:
                self.trunk.norm = nn.Identity()
            # Add the Siglip feature adapter MLP
            self.trunk.siglip_proj = create_mlp_from_config(config.radio_adapter_mlp_version, config.radio_adapter_mlp_input_dim, config.radio_adapter_mlp_hidden_dim, config.radio_adapter_mlp_output_dim, config.radio_adapter_mlp_num_inner)
            # Change the attn pool module because it was initialized using RADIO ViT Backbone's dimension, but we want it to be the same as SigLIP's attn pool
            self.trunk.attn_pool = AttentionPoolLatent(1152, num_heads=16, mlp_ratio=3.7362, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        # make the timm vit to be able to interpolate pos embedding
        assert self.trunk.num_prefix_tokens == 0, "S3 currently does not support prefix tokens for timm models!"

        # Add custom forward methods to the trunk model
        self.trunk.forward_tokenize = types.MethodType(forward_tokenize, self.trunk)
        self.trunk.forward_after_tokenize = types.MethodType(forward_after_tokenize_w_kvcache, self.trunk)
        self.trunk.attn_pool.forward = types.MethodType(forward_attn_pool_with_mask, self.trunk.attn_pool)
        self.trunk._pos_embed = types.MethodType(_pos_embed, self.trunk)
        self.trunk.selected_pos_embed = types.MethodType(selected_pos_embed, self.trunk)

        # Configure multi-scale processing
        self.ps3_scales = config.ps3_scales
        self.ps3_scales.sort()
        self.s3_image_size = self.ps3_scales[-1]

        # Set number of hidden layers to return (for memory efficiency)
        self.num_hidden_layers_to_return = self.layers
        warnings.warn(f"The number of hidden layers to return hidden states is currently set to {self.num_hidden_layers_to_return}. If this value is large, it can consume a lot of memory. Consider setting it to a smaller value if you won't use all the hidden states from every layer!")
        self.low_res_token_num = self.trunk.low_res_token_num = (self.ps3_scales[0] // self.trunk.patch_embed.patch_size[0]) ** 2

        # Token selection parameters
        self.max_select_num_each_scale = config.max_select_num_each_scale
        if config.max_select_num_each_scale is None:
            # Automatically distribute token selection budget across scales based on pixel counts
            pixels_each_scale = [s**2 for s in self.ps3_scales]
            self.max_select_num_each_scale = [int(config.max_select_num * pixels_each_scale[i] / sum(pixels_each_scale[1:])) for i in range(1, len(pixels_each_scale))]
            self.max_select_num_each_scale[-1] += config.max_select_num - sum(self.max_select_num_each_scale)

        # Feature projection for token selection
        self.selection_feature_proj = Mlp(in_features=self.width * len(self.config.select_based_on_layer), hidden_features=self.width * len(self.config.select_based_on_layer), out_features=self.width * (len(self.ps3_scales) - 1), norm_layer=nn.LayerNorm)
        self.prior_prompt = nn.Parameter(torch.randn(self.width))

        # Optional separate positional embeddings for each scale
        if config.separate_pos_emb:
            self.pos_emb_residual = nn.ParameterList([
                nn.Parameter(torch.zeros(1, (self.ps3_scales[i] // self.trunk.patch_embed.patch_size[0]) ** 2, self.width))
                for i in range(1, len(self.ps3_scales))
            ])
        
        # High-resolution feature extraction for token selection
        if self.config.highres_selection_feature:
            self.highres_selection_feature_module = ShallowConvNet(config)
            self.prompt_proj_for_highres = nn.Linear(self.width, config.highres_selection_module_out_dim * (len(self.ps3_scales) - 1))
        
        assert kwargs.get('patch_dropout', 0) == 0, 'S3 currently does not support patch_dropout because otherwise we will get incomplete low-res feature map!'

    def token_selection_param_names(self):
        """
        Return the names of parameters used for token selection.
        
        Returns:
            list: List of parameter names involved in the token selection mechanism
        """
        # return ["selection_feature_proj", "prior_prompt", "highres_selection_feature_module", "prompt_proj_for_highres"]
        return ["selection_feature_proj", "highres_selection_feature_module", "prompt_proj_for_highres"]
    
    def image_size_scale_i(self, x, scale_idx):
        """
        Calculate the image size at a specific scale index.
        
        Args:
            x (torch.Tensor): Input image tensor
            scale_idx (int): Index of the scale to calculate size for
            
        Returns:
            tuple: (height, width) of the image at the specified scale
        """
        # For the lowest scale, we use square images at all times
        if scale_idx == 0:
            return (self.ps3_scales[0], self.ps3_scales[0])
        
        h, w = x.shape[-2:]
        if h >= w:
            zoom_ratio = self.ps3_scales[scale_idx] // self.ps3_scales[0]
            size = (self.ps3_scales[0] * zoom_ratio, max(int(self.ps3_scales[0] / h * w) // self.trunk.patch_embed.patch_size[0], 1) * self.trunk.patch_embed.patch_size[0] * zoom_ratio)
        else:
            zoom_ratio = self.ps3_scales[scale_idx] // self.ps3_scales[0]
            size = (max(int(self.ps3_scales[0] / w * h) // self.trunk.patch_embed.patch_size[0], 1) * self.trunk.patch_embed.patch_size[0] * zoom_ratio, self.ps3_scales[0] * zoom_ratio)
        return size

    def feature_size_scale_i(self, x, scale_idx):
        """
        Calculate the feature map size at a specific scale index for an image.
        
        Args:
            x (torch.Tensor): Input image tensor
            scale_idx (int): Index of the scale to calculate feature size for
            
        Returns:
            tuple: (height, width) of the feature map at the specified scale
        """
        size = self.image_size_scale_i(x, scale_idx)
        size = (max(size[0] // self.trunk.patch_embed.patch_size[0], 1), max(size[1] // self.trunk.patch_embed.patch_size[0], 1))
        return size

    def max_highres_token_num(self, x, only_select_first_n_scale=None):
        """
        Calculate the maximum number of high-resolution tokens for an image.
        
        Args:
            x (torch.Tensor): Input image tensor
            only_select_first_n_scale (int, optional): If provided, only consider the first n scales
            
        Returns:
            int: Maximum number of high-resolution tokens
        """
        feature_size_each_scale = [self.feature_size_scale_i(x, i) for i in range(1, len(self.ps3_scales) if only_select_first_n_scale is None else max(only_select_first_n_scale) + 1)]
        max_token_num_each_scale = [size[0] * size[1] for size in feature_size_each_scale]
        return sum(max_token_num_each_scale)
    
    def resize_to_scale_i(self, x, scale_idx):
        """
        Resize the input image to a specific scale.
        
        Args:
            x (torch.Tensor): Input image tensor
            scale_idx (int): Index of the scale to resize to
            
        Returns:
            torch.Tensor: Resized image
        """
        size = self.image_size_scale_i(x, scale_idx)
        return F.interpolate(x, size=size, mode='bicubic')

    def _calculate_select_probs(self, feature_map, highres_feature_map, prompt, input):
        """
        Calculate selection probabilities based on feature maps and prompt.
        
        Args:
            feature_map (torch.Tensor): Feature map for patch selection at each scale, with shape B * num_scale * C * H * W
            highres_feature_map (torch.Tensor, optional): High-resolution feature map, with shape B * C * H * W
            prompt (torch.Tensor): Prompt embedding for selection, with shape B * C
            input (torch.Tensor): Input image tensor for size calculations
            
        Returns:
            list: Selection probabilities for each scale
        """
        # Calculate selection logits using cosine similarity between feature map and prompt
        select_logits = (F.normalize(feature_map, dim=2) * F.normalize(prompt, dim=-1)[..., None, None].unsqueeze(1)).sum(dim=2)  # B * num_scale * H * W
        
        # Incorporate high-res feature map if available for more accurate selection
        if highres_feature_map is not None:
            high_res_prompt = rearrange(self.prompt_proj_for_highres(prompt), "b (ns c) -> b ns c", ns=len(self.ps3_scales) - 1)
            highres_select_logits = (F.normalize(highres_feature_map, dim=1).unsqueeze(1) * F.normalize(high_res_prompt, dim=-1)[..., None, None]).sum(dim=2)  # B * num_scale * H * W
            select_logits = (select_logits + F.adaptive_max_pool2d(highres_select_logits, select_logits.shape[-2:])) / 2
        
        # Apply scaling, clamping, and softmax to get probabilities
        select_logits = select_logits * 5.0
        select_logits = select_logits.clamp(-10, 10)
        select_logits = F.pad(select_logits[..., None], (0, 1))
        select_probs = F.softmax(select_logits, dim=-1)[..., 0]  # B * num_scale * H * W
        
        # Use the same selection probabilities for all scales
        select_probs = select_probs * 0 + select_probs[:, :1]
        
        # Resize selection probabilities to match feature map sizes at each scale
        select_probs = [F.interpolate(select_probs[:, i:i+1].float(), 
                                      size=self.feature_size_scale_i(input, i+1), 
                                      mode='nearest').squeeze(1).to(select_probs) 
                        for i in range(len(self.ps3_scales) - 1)]
        
        return select_probs

    def get_selection_probs(self, input: torch.Tensor, feature_map: torch.Tensor, highres_feature_map=None, prompt=None, gt_selection_maps=None, smooth_selection_prob=False):
        """
        Calculate the probability of selecting each token for high-resolution processing.
        
        Args:
            input (torch.Tensor): Input image tensor
            feature_map (torch.Tensor): Feature map from low-resolution processing for patch selection
            highres_feature_map (torch.Tensor, optional): High-resolution feature map for patch selection
            prompt (torch.Tensor, optional): prompt embedding for top-down selection
            gt_selection_maps (torch.Tensor, optional): Ground truth selection maps
            smooth_selection_prob (bool): Whether to smooth the selection probabilities
            
        Returns:
            tuple: (selection probabilities, prior selection probabilities, posterior selection probabilities)
        """
        assert feature_map.dim() == 4, "Feature map must be with shape B * C * H * W"
        if prompt is not None:
            assert prompt.dim() == 2, "Prompt must be with shape B * C"
            assert prompt.shape[1] == self.width, "Feature map and prompt must have the same channel dimension"
        if gt_selection_maps is not None:
            assert gt_selection_maps.dim() == 3, "GT selection maps must be with shape B * H * W"

        # If gt_selection_maps is provided, select based on it
        select_with_gt = gt_selection_maps is not None

        # If no prompt is provided, select based on prior
        select_with_prior = prompt is None

        B = feature_map.shape[0]

        # get selection logits for each scale
        feature_map = self.selection_feature_proj(feature_map.permute(0, 2, 3, 1))  # B * H * W * (num_scale * C)
        feature_map = rearrange(feature_map, "b h w (ns c) -> b ns c h w", ns=len(self.ps3_scales) - 1)
        
        # Calculate prior selection probabilities
        prior_select_probs = self._calculate_select_probs(
            feature_map, 
            highres_feature_map, 
            self.prior_prompt[None].repeat(B, 1), 
            input
        )

        # Calculate posterior selection probabilities if prompt is provided
        if not select_with_prior:
            posterior_select_probs = self._calculate_select_probs(
                feature_map,
                highres_feature_map,
                prompt,
                input
            )
        else:
            posterior_select_probs = None

        if select_with_gt:
            gt_selection_maps = F.interpolate(gt_selection_maps.unsqueeze(1).float(), size=self.feature_size_scale_i(input, 1), mode='area').squeeze(1).to(gt_selection_maps)
            gt_selection_maps = [F.interpolate(gt_selection_maps.unsqueeze(1).float(), size=self.feature_size_scale_i(input, i+1), mode='nearest').squeeze(1).to(gt_selection_maps) for i in range(len(self.ps3_scales) - 1)]

        if select_with_prior:
            select_probs = prior_select_probs   # list of B * H * W tensors, len(list) = num_scale - 1
        else:
            select_probs = posterior_select_probs   # list of B * H * W tensors, len(list) = num_scale - 1

        if select_with_gt:
            select_probs = [(gt_prob > 0) * 1 + (gt_prob <= 0) * prob for gt_prob, prob in zip(gt_selection_maps, select_probs)]

        # smoothing the selection probs
        if smooth_selection_prob:
            select_probs = [F.interpolate(F.interpolate(x.unsqueeze(1), size=(6, 6), mode='area'), size=x.shape[-2:], mode='nearest').squeeze(1) for x in select_probs]
        
        return select_probs, prior_select_probs, posterior_select_probs

    @torch.no_grad()
    def get_selection_maps(self, select_probs, old_selection_maps=None, num_select_token=None, only_select_first_n_scale=None):
        """
        Convert selection probabilities to binary selection maps.
        
        Args:
            select_probs (list): List of tensors with selection probabilities
            old_selection_maps (list, optional): The map of previously selected patches. We only select patches that haven't been selected before.
            num_select_token (int, optional): Number of tokens to select
            only_select_first_n_scale (list, optional): Only select from first n scales
            
        Returns:
            list: Binary selection maps for each scale
        """
        B = select_probs[0].shape[0]

        select_probs_all_instances = select_probs
        old_selection_maps_all_instances = old_selection_maps
        only_select_first_n_scale_all_instances = only_select_first_n_scale
        selection_maps_all_instances = [[] for _ in range(len(select_probs))]

        for instance_id in range(B):
            select_probs = [prob[instance_id:instance_id+1] for prob in select_probs_all_instances]
            old_selection_maps = [map[instance_id:instance_id+1] for map in old_selection_maps_all_instances] if old_selection_maps_all_instances is not None else None
            only_select_first_n_scale = only_select_first_n_scale_all_instances[instance_id] if only_select_first_n_scale_all_instances is not None else None

            # If old_selection_maps is provided, then only select new tokens
            if old_selection_maps is not None:
                select_probs = [select_probs[i] * (1 - old_selection_maps[i]) + (-1) * old_selection_maps[i] for i in range(len(self.ps3_scales) - 1)]
            
            # Start selecting tokens
            select_probs_flatten = [prob.reshape(1, -1) for prob in select_probs]
            select_num_each_scale = deepcopy(self.max_select_num_each_scale)
            if num_select_token is not None:
                assert num_select_token <= sum(self.max_select_num_each_scale), "Number of selected tokens must be less than the maximum number of selected tokens"
                select_num_each_scale = [int(x * num_select_token / sum(self.max_select_num_each_scale)) for x in self.max_select_num_each_scale]
                select_num_each_scale[-1] += num_select_token - sum(select_num_each_scale)
            if only_select_first_n_scale is not None:
                select_num_each_scale = [int(x / sum(select_num_each_scale[:only_select_first_n_scale]) * sum(select_num_each_scale)) if i < only_select_first_n_scale else 1
                                        for i, x in enumerate(select_num_each_scale)]
                # make sure sum of select num is the same for every instance
                select_num_each_scale[-1] += sum(self.max_select_num_each_scale) - sum(select_num_each_scale)
            for i, prob in enumerate(select_probs_flatten):
                if (prob[0] != -1).sum() < select_num_each_scale[i]:
                    if i < len(select_probs_flatten) - 1:
                        select_num_each_scale[i + 1] += select_num_each_scale[i] - (prob[0] != -1).sum()
                    select_num_each_scale[i] = (prob[0] != -1).sum()

            selected_ids = [prob.topk(k=select_num_each_scale[i], dim=-1).indices for i, prob in enumerate(select_probs_flatten)]
            selection_maps = [torch.zeros_like(prob) for prob in select_probs_flatten]
            for i, ids in enumerate(selected_ids):
                selection_maps[i][torch.arange(1).unsqueeze(-1), ids] = 1
            selection_maps = [rearrange(mp, 'b (h w) -> b h w', h=prob.shape[-2]) for mp, prob in zip(selection_maps, select_probs)]  # list of B * H * W

            selection_maps_all_instances = [selection_maps_all_instances[i] + [selection_maps[i]] for i in range(len(selection_maps))]
        
        selection_maps_all_instances = [torch.cat(selection_maps, dim=0) for selection_maps in selection_maps_all_instances]
        
        return selection_maps_all_instances


    def update_selection_maps(self, old_selection_maps, new_selection_maps):
        """
        Combine old and new selection maps for progressive token selection.
        
        This function merges previously selection maps with the new ones,
        ensuring there's no overlap between them. This is used during iterative
        high-resolution processing to accumulate selected regions across iterations.
        
        Args:
            old_selection_maps (list): Previously selected tokens as binary masks
                                      for each scale. Each mask has shape B x H x W.
            new_selection_maps (list): Newly selected tokens as binary masks
                                      for each scale. Each mask has shape B x H x W.
            
        Returns:
            list: Updated selection maps combining old and new selections
                 without overlap, maintaining the same format as inputs.
        """
        B = old_selection_maps[0].shape[0]
        # Flatten selection maps for easier processing
        old_selection_flatten = torch.cat([smap.reshape(B, -1) for smap in old_selection_maps], dim=-1)  # B * N
        new_selection_flatten = torch.cat([smap.reshape(B, -1) for smap in new_selection_maps], dim=-1)  # B * N
        
        # Verify that old and new selections don't overlap
        assert torch.all(old_selection_flatten.bool() * new_selection_flatten.bool() == False), f"Old and new selection maps must be exclusive, but now {(old_selection_flatten * new_selection_flatten == 0).sum()}"
            
        # Combine old and new selections
        selection_flatten = old_selection_flatten + new_selection_flatten

        # Reshape back to original format
        updated_selection_maps = list(torch.split(selection_flatten, [smap.shape[1] * smap.shape[2] for smap in old_selection_maps], dim=-1))
        updated_selection_maps = [rearrange(mp, 'b (h w) -> b h w', h=old_smap.shape[-2]) for mp, old_smap in zip(updated_selection_maps, old_selection_maps)]  # list of B * H * W

        return updated_selection_maps

    def aggregate_features(self, features_each_step, selection_maps_each_step):
        """
        Aggregate features from multiple selection steps into a unified feature set.
        
        This function combines features from different high-resolution processing iterations
        into a single coherent set of features. It ensures that features from each step
        are placed in their correct positions according to the selection maps.
        
        Args:
            features_each_step (list[list[torch.Tensor]]): Features from each selection step. list of lists, len(list) = num_step, each list in the list is a list of B * N * C tensors, len(list) = num_layers
            selection_maps_each_step (list[list[torch.Tensor]]): Selection maps from each step. list of lists, len(list) = num_step, each list in the list is a list of B * H * W tensors, len(list) = num_scale - 1
            
        Returns:
            list: Aggregated features for each layer, where each tensor has shape B x N_total x C,
                 with N_total being the total number of selected tokens across all steps.
        """
        B = selection_maps_each_step[0][0].shape[0]
        C = features_each_step[0][0].shape[-1]
        
        # Flatten selection maps for each step
        selection_flatten_each_step = [torch.cat([smap.reshape(B, -1) for smap in selection_maps], dim=-1) for selection_maps in selection_maps_each_step]  # B * N
            
        # Initialize empty feature tensors for each layer
        features_flatten = [torch.zeros(B, selection_flatten_each_step[0].shape[1], C, device=selection_flatten_each_step[0].device, dtype=selection_flatten_each_step[0].dtype) for _ in features_each_step[0]]
        
        # Fill in features from each step at their corresponding positions
        for layer_id in range(len(features_flatten)):
            for step_id in range(len(features_each_step)):
                features_flatten[layer_id][selection_flatten_each_step[step_id].bool()] = features_each_step[step_id][layer_id].flatten(0, 1)
        
        # Create the final aggregated features by selecting only the positions that have been filled
        agg_selection_flatten = sum(selection_flatten_each_step)
        agg_features = [features[agg_selection_flatten.bool()].reshape(B, -1, C) for features in features_flatten]

        return agg_features

    def format_features_into_feature_maps(self, features, selection_maps):
        """
        Format token features back into spatial feature maps.
        
        Args:
            features (torch.Tensor): Token features (B x N x C) containing both low-res and high-res features
            selection_maps (list): Selection maps indicating token positions. list of B * H * W tensors, len(list) = num_scale - 1. Each tensor is a binary mask.
            
        Returns:
            list: Feature maps for each scale
        """
        B, _, C = features.shape
        high_res_selection_num = sum([x.sum(dim=(-1, -2)) for x in selection_maps])
        assert torch.all(high_res_selection_num + self.low_res_token_num == features.shape[1]), \
            f"Number of selected tokens must be the same as the number of features, but now {high_res_selection_num + self.low_res_token_num} != {features.shape[1]}"

        # Add the selection map for the low-res tokens. Assume always selecting all low-res tokens.
        selection_maps = [torch.ones(B, int(self.low_res_token_num**0.5), int(self.low_res_token_num**0.5), device=selection_maps[0].device, dtype=selection_maps[0].dtype)] + selection_maps
        
        # Flatten the selection maps and features across all scales and all instances
        flatten_selection_maps = torch.cat([x.flatten(1, 2) for x in selection_maps], dim=-1).flatten(0, 1)  # (B * N_full)
        flatten_features = features.flatten(0, 1)  # (B * N) * C

        # Create a full feature tensor by placing the flattened features into the positions specified by the flattened selection maps
        full_features = torch.zeros(flatten_selection_maps.shape[0], C, dtype=features.dtype, device=features.device)
        full_features[flatten_selection_maps == 1] = flatten_features
        full_features = full_features.reshape(B, -1, C)

        # Format the full features into feature maps for each scale
        feature_map_size_each_scale = [(x.shape[1], x.shape[2]) for x in selection_maps]
        full_features = full_features.split([x[0] * x[1] for x in feature_map_size_each_scale], dim=1)
        full_feature_maps = [rearrange(x, 'b (h w) c -> b c h w', h=feature_map_size_each_scale[i][0], w=feature_map_size_each_scale[i][1]) for i, x in enumerate(full_features)]

        return full_feature_maps

    def forward_low_res(self, x: torch.Tensor, output_hidden_states=False, return_kv_cache=False):
        """
        Process the input image at the lowest resolution.
        
        Args:
            x (torch.Tensor): Input image tensor
            output_hidden_states (bool): Whether to return hidden states from all layers
            return_kv_cache (bool): Whether to return key-value cache for later use
            
        Returns:
            dict:
                - x: low-res features
                - hidden_states: hidden states from all layers
                - output_kv_cache: low-res key-value cache
        """
        # Resize the image to the smallest scale and process it
        x = self.resize_to_scale_i(x.to(torch.float32), 0).to(x.dtype)
        x = self.trunk.forward_tokenize(x)

        # Forward pass through the transformer backbone
        outs = self.trunk.forward_after_tokenize(x, output_hidden_states=output_hidden_states, return_kv_cache=return_kv_cache)
        return outs
    
    def tokenize_high_res(self, input, selection_maps):
        """
        Tokenize high-resolution patches based on selection maps.
        
        This function processes the input image at multiple high-resolution scales,
        but only for the patches indicated by the selection maps. It applies
        appropriate resizing, tokenization, and positional embedding for each scale.
        
        Args:
            input (torch.Tensor): Input image tensor with shape B x C x H x W
            selection_maps (list): List of binary masks indicating which patches to process
                                  at each scale. Each mask has shape B x H x W where 1 indicates
                                  the patch is selected and 0 indicates not selected.
            
        Returns:
            torch.Tensor: High-resolution tokens with shape B x N x C, where N is the
                         total number of selected high-resolution patches across all scales.
        """
        B = input.shape[0]
        high_res_tokens = []
        for scale_id, scale in enumerate(self.ps3_scales[1:]):
            # Skip if no token is selected at this scale
            if torch.all(selection_maps[scale_id] == 0):
                continue

            # Resize image to current scale
            x = self.resize_to_scale_i(input.to(torch.float32), scale_id + 1).to(input.dtype)

            # Calculate feature size and tokenize only selected patches
            cur_feature_size = self.feature_size_scale_i(input, scale_id + 1)
            x = self.trunk.forward_tokenize(x, selection_maps[scale_id], cur_feature_size[0], cur_feature_size[1])
            
            # Apply scale-specific positional embedding if configured
            if self.config.separate_pos_emb:
                cur_pos_emb_residual = self.pos_emb_residual[scale_id]
                cur_pos_emb_residual = rearrange(cur_pos_emb_residual, "1 (h w) c -> 1 c h w", h=int(cur_pos_emb_residual.shape[1]**0.5))
                cur_pos_emb_residual = cur_pos_emb_residual[:, :, :cur_feature_size[0], :cur_feature_size[1]]
                cur_pos_emb_residual = rearrange(cur_pos_emb_residual, "1 c h w -> 1 (h w) c")
                cur_pos_emb_residual = cur_pos_emb_residual.repeat(B, 1, 1)[selection_maps[scale_id].flatten(1, 2).bool()].view(B, -1, self.width)
                x = x + cur_pos_emb_residual

            high_res_tokens.append(x)
        
        # Combine tokens from all scales
        high_res_tokens = torch.cat(high_res_tokens, dim=1)  # B * N * C

        return high_res_tokens
    
    def forward_high_res(self, x: torch.Tensor, selection_maps, kv_cache=None, output_hidden_states=False, pool_gt_token_only=False, gt_selection_maps=None, return_kv_cache=False):
        """
        Process the input image at high resolution and only at selected regions.
        
        Args:
            x (torch.Tensor): Input image tensor
            selection_maps (list[torch.Tensor]): List of binary masks indicating which patches to process. Each mask is with shape B * H * W. 1 indicates the patch is selected and 0 indicates not selected.
            kv_cache (torch.Tensor, optional): Key-value cache from low-resolution processing
            output_hidden_states (bool): Whether to return hidden states from all layers
            pool_gt_token_only (bool): Whether to pool only tokens specified by ground truth selection maps
            gt_selection_maps (torch.Tensor, optional): Ground truth selection maps
            return_kv_cache (bool): Whether to return key-value cache
            
        Returns:
            Various combinations of hidden states, KV cache, and output features depending on arguments
        """
        assert selection_maps[0].dim() == 3, "Each selection map must be with shape B * H * W"
        assert len(selection_maps) == len(self.ps3_scales) - 1, "Number of selection map must be the same as the number of high-res scales"

        input = x
        B = x.shape[0]

        # tokenize for high-res
        # If every sample in the batch has the same number of selected patches, then we can batchify the tokenization process
        if all([torch.all(selection_maps[scale_id].flatten(1, 2).sum(dim=-1) == selection_maps[scale_id].flatten(1, 2).sum(dim=-1)[0]) for scale_id in range(len(self.ps3_scales) - 1)]):
            high_res_tokens = self.tokenize_high_res(input, selection_maps)
        
        # If different samples have different number of selected patches, then we need to process each sample individually
        else:
            high_res_tokens = []
            original_input = input
            original_selection_maps = selection_maps
            for i in range(B):
                input = original_input[i:i+1]
                selection_maps = [m[i:i+1] for m in original_selection_maps]
                cur_high_res_tokens = self.tokenize_high_res(input, selection_maps)
                high_res_tokens.append(cur_high_res_tokens)
            high_res_tokens = torch.cat(high_res_tokens, dim=0)

        # If gt_selection_maps is given, then only pool the features within the gt_selection_mask
        if pool_gt_token_only:
            assert gt_selection_maps is not None
            gt_selection_maps = [F.interpolate(gt_selection_maps.unsqueeze(1).float(), size=self.feature_size_scale_i(input, i+1), mode='area').squeeze(1).to(gt_selection_maps) for i in range(len(self.ps3_scales) - 1)]
            gt_selection_maps_flat = torch.cat([smap.reshape(B, -1) for smap in gt_selection_maps], dim=-1)
            selection = torch.cat([smap.reshape(B, -1) for smap in selection_maps], dim=-1)  # B * N
            gt_selection_mask = gt_selection_maps_flat[selection.bool()].reshape(B, -1)
        else:
            gt_selection_mask = None

        # forward
        outs = self.trunk.forward_after_tokenize(high_res_tokens, kv_cache=kv_cache, output_hidden_states=output_hidden_states, 
                                                 pool_mask=gt_selection_mask, return_kv_cache=return_kv_cache)
        if output_hidden_states:
            outs["hidden_states"] = outs["hidden_states"][-self.num_hidden_layers_to_return:]

        return outs

    def forward(self, x: torch.Tensor, prompt=None, gt_selection_maps=None, is_global_text=None, output_hidden_states=False, pool_gt_token_only=False, num_look_close=None, num_token_look_close=None, smooth_selection_prob=False, only_select_first_n_scale=None):
        """
        Forward pass of the PS3 Vision Encoder.
        
        The model first processes the image at low resolution, then selectively processes
        regions at higher resolutions based on selection probabilities.
        
        Args:
            x (torch.Tensor): Input image tensor
            prompt (torch.Tensor, optional): Prompt embedding for top-down selection
            gt_selection_maps (torch.Tensor, optional): Ground truth selection maps
            is_global_text (torch.Tensor, optional): Boolean tensor indicating if text is global
            output_hidden_states (bool): Whether to return hidden states from all layers
            pool_gt_token_only (bool): Whether to pool only tokens specified by ground truth
            num_look_close (int, optional): Number of iterations for high-res processing
            num_token_look_close (int, optional): Number of tokens to process in high-res
            smooth_selection_prob (bool): Whether to smooth the selection probabilities
            only_select_first_n_scale (list, optional): Only select from first n scales
            
        Returns:
            PS3VisionModelOutput: Model outputs including features, selection maps, etc.
        """
        # get low-res features
        low_res_outs = self.forward_low_res(x, output_hidden_states=True, return_kv_cache=True if not (num_look_close == 0 or num_token_look_close == 0) else False)
        low_res_hidden_states, low_res_kv_cache, low_res_pooled = low_res_outs["hidden_states"], low_res_outs["output_kv_cache"], low_res_outs["x"]
        
        # Extract features for token selection from specified layers
        selection_features = torch.cat([low_res_hidden_states[i] for i in self.config.select_based_on_layer], dim=-1)

        # Calculate selection probabilities for high-resolution processing
        selection_features = rearrange(selection_features, "b (h w) c -> b c h w", h=self.feature_size_scale_i(x, 0)[0])
        highres_selection_features = self.highres_selection_feature_module(F.interpolate(x.to(torch.float32), size=1512, mode='bicubic').to(x.dtype)) if self.config.highres_selection_feature else None
        select_probs, prior_select_probs, posterior_select_probs = self.get_selection_probs(x, selection_features, highres_selection_features, prompt, gt_selection_maps, smooth_selection_prob=smooth_selection_prob)

        # process high-res
        if output_hidden_states:
            # Only process the low-res features
            if num_look_close == 0 or num_token_look_close == 0:
                return PS3VisionModelOutput(
                    last_hidden_state=low_res_hidden_states[-1],
                    hidden_states=low_res_hidden_states,
                    selection_probs=select_probs
                )

            # Determine how many high-resolution iterations to perform
            if num_look_close == "all" or num_token_look_close == 'all':  # Process all the high-res patches
                num_look_close = math.ceil(self.max_highres_token_num(x, only_select_first_n_scale) / sum(self.max_select_num_each_scale))
            elif num_look_close is not None and num_look_close > 0:  # Run the high-res selection and encoding for num_look_close times
                num_look_close = min(num_look_close, math.ceil(self.max_highres_token_num(x, only_select_first_n_scale) / sum(self.max_select_num_each_scale)))
            elif num_token_look_close is not None and num_token_look_close > 0:  # Run the high-res selection and encoding for num_token_look_close tokens
                num_token_look_close = min(num_token_look_close, self.max_highres_token_num(x, only_select_first_n_scale))
                num_look_close = math.ceil(num_token_look_close / sum(self.max_select_num_each_scale))
                num_token_look_close_last_iter = num_token_look_close % sum(self.max_select_num_each_scale)
                num_token_look_close_last_iter = num_token_look_close_last_iter if num_token_look_close_last_iter > 0 else sum(self.max_select_num_each_scale)
            else:
                raise ValueError(f"Invalid config: num_look_close={num_look_close}, num_token_look_close={num_token_look_close}")

            # First iteration of high-resolution processing
            selection_maps = self.get_selection_maps(select_probs, 
                                                     num_select_token=num_token_look_close if num_token_look_close is not None and num_token_look_close <= sum(self.max_select_num_each_scale) else None,
                                                     only_select_first_n_scale=only_select_first_n_scale)
            hidden_states = self.forward_high_res(x, selection_maps, kv_cache=low_res_kv_cache, output_hidden_states=True, return_kv_cache=False)["hidden_states"]
            
            # Store results from first iteration
            hidden_states_each_step = [hidden_states]
            selection_maps_each_step = [selection_maps]
            
            # Additional iterations of high-resolution processing
            for k in range(num_look_close - 1):
                old_selection_maps = selection_maps
                selection_maps = self.get_selection_maps(select_probs, 
                                                         old_selection_maps=old_selection_maps, num_select_token=num_token_look_close_last_iter if num_token_look_close is not None and k == num_look_close - 2 else None,
                                                         only_select_first_n_scale=only_select_first_n_scale)
                hidden_states = self.forward_high_res(x, selection_maps, kv_cache=low_res_kv_cache, output_hidden_states=True, return_kv_cache=False)["hidden_states"]
                hidden_states_each_step.append(hidden_states)
                selection_maps_each_step.append(selection_maps)
                selection_maps = self.update_selection_maps(old_selection_maps, selection_maps)

            # Aggregate features from all iterations
            hidden_states = self.aggregate_features(hidden_states_each_step, selection_maps_each_step)
            low_res_hidden_states = low_res_hidden_states[-self.num_hidden_layers_to_return:]
            hidden_states = [torch.cat([low_res_x, x], dim=1) for low_res_x, x in zip(low_res_hidden_states, hidden_states)]  # Since when using kv cache, the forward function only returns the high-res hidden states, we need to manually concatenate it with the low-res hidden states
            last_hidden_state = hidden_states[-1]

            return PS3VisionModelOutput(
                last_hidden_state=last_hidden_state,
                hidden_states=hidden_states,
                selection_maps=selection_maps,
                selection_probs=select_probs
            )
        
        else:
            selection_maps = self.get_selection_maps(select_probs, num_select_token=num_token_look_close if num_token_look_close is not None and num_token_look_close <= sum(self.max_select_num_each_scale) else None)
            pooled = self.forward_high_res(x, selection_maps, kv_cache=low_res_kv_cache, pool_gt_token_only=pool_gt_token_only, gt_selection_maps=gt_selection_maps)["x"]

            # For samples with global text, replace the pooled high-res feature with the pooled low-res feature
            if is_global_text is not None:
                pooled = is_global_text * low_res_pooled + is_global_text.logical_not() * pooled

            return PS3VisionModelOutput(
                pooled_output=pooled,
                selection_maps=selection_maps,
                selection_probs=select_probs,
                bottomup_selection_probs=prior_select_probs,
                topdown_selection_probs=posterior_select_probs
            )



#############################################
########## PreTrainedModel Wrapper ##########
#############################################


class PS3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = PS3Config
    base_model_prefix = "ps3"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "PS3VisionEncoder",
    ]
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def _init_weights(self, module):
        pass


class PS3TextModel(PS3PreTrainedModel):
    config_class = PS3TextConfig
    main_input_name = "text"

    def __init__(self, config: PS3TextConfig):
        super().__init__(config)

        self.text_model = TextTransformer(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(self, text):
        return self.text_model(text)


class PS3VisionModel(PS3PreTrainedModel):
    config_class = PS3VisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: PS3VisionConfig):
        super().__init__(config)

        self.vision_model = PS3VisionEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values, 
        num_look_close=None, 
        num_token_look_close=None, 
        prompt=None, 
        gt_selection_maps=None, 
        smooth_selection_prob=False,
        only_select_first_n_scale=None,
        is_global_text=None, 
        pool_gt_token_only=False,
        output_hidden_states=True,
    ):
        if not output_hidden_states:
            warnings.warn("Currently setting output_hidden_states=False is only intended for pre-training.")
            assert num_look_close is None, "num_look_close must be None in pre-training mode since during pre-training, the model only looks close once at all times"
            assert num_token_look_close is None, "num_token_look_close must be None in pre-training mode since during pre-training, the model only looks close once at all times"
            assert only_select_first_n_scale is None, "only_select_first_n_scale must be None in pre-training mode since during pre-training, the model selects all the scales at all times"
        
        if only_select_first_n_scale is not None and isinstance(only_select_first_n_scale, int):
            only_select_first_n_scale = [only_select_first_n_scale for _ in range(len(pixel_values))]

        return self.vision_model(
            x=pixel_values,
            prompt=prompt,
            gt_selection_maps=gt_selection_maps,
            is_global_text=is_global_text,
            output_hidden_states=output_hidden_states,
            pool_gt_token_only=pool_gt_token_only,
            num_look_close=num_look_close,
            num_token_look_close=num_token_look_close,
            smooth_selection_prob=smooth_selection_prob,
            only_select_first_n_scale=only_select_first_n_scale,
        )


class PS3Model(PS3PreTrainedModel):
    config_class = PS3Config
    main_input_name = "pixel_values"

    def __init__(self, config: PS3Config):
        super().__init__(config)

        if not isinstance(config.text_config, PS3TextConfig):
            raise TypeError(
                "config.text_config is expected to be of type PS3TextConfig but is of type"
                f" {type(config.text_config)}."
            )

        if not isinstance(config.vision_config, PS3VisionConfig):
            raise TypeError(
                "config.vision_config is expected to be of type PS3VisionConfig but is of type"
                f" {type(config.vision_config)}."
            )

        text_config = config.text_config
        vision_config = config.vision_config

        # First, initialize the text and vision models with proper attention implementation
        text_model = PS3TextModel._from_config(text_config)
        vision_model = PS3VisionModel._from_config(vision_config)

        # Second, get the text and vision submodules (for backward compatibility)
        self.text_model = text_model.text_model
        self.vision_model = vision_model.vision_model

        # Initialize weights and apply final processing
        self.post_init()
    
    def encode_text(self, text):
        return self.text_model(text)

    def encode_image(
        self, 
        pixel_values, 
        num_look_close=None, 
        num_token_look_close=None, 
        prompt=None, 
        gt_selection_maps=None, 
        smooth_selection_prob=False,
        only_select_first_n_scale=None,
        is_global_text=None, 
        pool_gt_token_only=False,
        output_hidden_states=True,
    ):
        if not output_hidden_states:
            warnings.warn("Currently setting output_hidden_states=False is only intended for pre-training.")
            assert num_look_close is None, "num_look_close must be None in pre-training mode since during pre-training, the model only looks close once at all times"
            assert num_token_look_close is None, "num_token_look_close must be None in pre-training mode since during pre-training, the model only looks close once at all times"
            assert only_select_first_n_scale is None, "only_select_first_n_scale must be None in pre-training mode since during pre-training, the model selects all the scales at all times"

        return self.vision_model(
            x=pixel_values,
            prompt=prompt,
            gt_selection_maps=gt_selection_maps,
            is_global_text=is_global_text,
            output_hidden_states=output_hidden_states,
            pool_gt_token_only=pool_gt_token_only,
            num_look_close=num_look_close,
            num_token_look_close=num_token_look_close,
            smooth_selection_prob=smooth_selection_prob,
            only_select_first_n_scale=only_select_first_n_scale,
        )

    


###############################################################################################
########## Define a shallow convnet to extract high-res features for token selection ##########
###############################################################################################


class ConvNeXtBlock_HFSavePreTrainedFix(ConvNeXtBlock):
    """ 
    Adapted from timm ConvNeXt: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/convnext.py
    Original license: Apache-2.0, Copyright 2022, Ross Wightman

    In huggingface transformer's save_pretrained function, if a parameter's name contains "gamma",
    automatically replace "gamma" with "weight" in the state_dict. This will cause the gamma parameter
    to be saved in a different name and won't be correctly loaded later. Same issue happens for 
    "beta" -> "bias". The original ConvNeXtBlock has a parameter called "gamme:.
    This class is to fix this issue by renaming that parameter into somthing else.
    Ref: https://github.com/huggingface/transformers/issues/29554
    """

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.weight = deepcopy(self.gamma)
        del(self.gamma)

    def forward(self, x):
        shortcut = x
        x = self.conv_dw(x)
        if self.use_conv_mlp:
            x = self.norm(x)
            x = self.mlp(x)
        else:
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
            x = self.mlp(x)
            x = x.permute(0, 3, 1, 2)
        if self.weight is not None:
            x = x.mul(self.weight.reshape(1, -1, 1, 1))

        x = self.drop_path(x) + self.shortcut(shortcut)
        return x
    

class ShallowConvNet(nn.Module):
    def __init__(self, config: PS3VisionConfig):
        super().__init__()

        hidden_dim = config.highres_selection_module_hidden_dim
        out_dim = config.highres_selection_module_out_dim
        depth = config.highres_selection_module_depth
        kernel_size = config.highres_selection_module_kernel_size

        self.stem = nn.Sequential(
            nn.Conv2d(3, hidden_dim, kernel_size=kernel_size, stride=kernel_size, bias=True),
            LayerNorm2d(hidden_dim),
        )

        blocks = []
        for i in range(depth):
            blocks.append(ConvNeXtBlock_HFSavePreTrainedFix(
                in_chs=hidden_dim,
                out_chs=hidden_dim,
                kernel_size=7,
                dilation=1,
            ))
        self.blocks = nn.Sequential(*blocks)

        self.out_proj = nn.Conv2d(hidden_dim, out_dim, kernel_size=1, stride=1, bias=True)
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.out_proj(x)
        return x


################################################################
########## Redefine some models and functions in timm ##########
################################################################


class SelectedPatchEmbed(PatchEmbed):
    """
    Adapted from timm PatchEmbed: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/patch_embed.py
    Original license: Apache-2.0, Copyright 2020, Ross Wightman
    
    Adapted to support only running patch embedding on selected patches.
    """

    def selected_proj(self, x, selection_map):
        B = x.shape[0]
        out_channels, _, kh, kw = self.proj.weight.shape

        # Convolution kernel
        weight = self.proj.weight   # out_channels * in_channels * kh * kw
        bias = self.proj.bias if self.proj.bias is not None else None   # out_channels

        # Unfold the input into patches
        unfolded_input = F.unfold(x, kernel_size=(kh, kw), stride=(kh, kw)).transpose(-1, -2)   # B * N * (in_channels * kh * kw)

        # Select patches
        selection_map = selection_map.flatten(1)  # B * (H * W)
        assert torch.all(selection_map.sum(dim=-1) == selection_map.sum(dim=-1)[0]), "Selection map must have the same number of selection for all samples in a batch"
        assert selection_map.shape[1] == unfolded_input.shape[1], "Selection map must have the same number of selection as the number of patches"
        unfolded_input = unfolded_input[selection_map.bool()].view(B, -1, unfolded_input.shape[-1])

        # Reshape the kernel
        reshaped_weight = weight.view(out_channels, -1)   # out_channels * (in_channels * kh * kw)

        # Perform matrix multiplication
        output = unfolded_input @ reshaped_weight.transpose(0, 1)[None]
        if bias is not None:
            output += bias[None, None]

        return output

    def forward(self, x, selection_map=None):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            elif not self.dynamic_img_pad:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                )
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        
        if selection_map is not None:
            x = self.selected_proj(x, selection_map)
            x = self.norm(x)
            return x
        else:
            x = self.proj(x)
            if self.flatten:
                x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
            elif self.output_fmt != Format.NCHW:
                x = nchw_to(x, self.output_fmt)
            x = self.norm(x)
            return x



def resample_abs_pos_embed(
        posemb: torch.Tensor,
        new_size: List[int],
        old_size: Optional[List[int]] = None,
        num_prefix_tokens: int = 1,
        interpolation: str = 'bicubic',
        antialias: bool = True,
        use_cpe: bool = False,
):
    """
    Adapted from timm resample_abs_pos_embed: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/pos_embed.py
    Original license: Apache-2.0, Copyright 2022, Ross Wightman
    
    Adapted to support cropped position embedding (cpe).
    """

    # sort out sizes, assume square if old size not provided
    num_pos_tokens = posemb.shape[1]
    num_new_tokens = new_size[0] * new_size[1] + num_prefix_tokens
    if num_new_tokens == num_pos_tokens and new_size[0] == new_size[1]:
        return posemb

    if old_size is None:
        hw = int(math.sqrt(num_pos_tokens - num_prefix_tokens))
        old_size = hw, hw

    if num_prefix_tokens:
        posemb_prefix, posemb = posemb[:, :num_prefix_tokens], posemb[:, num_prefix_tokens:]
    else:
        posemb_prefix, posemb = None, posemb

    # do the interpolation
    embed_dim = posemb.shape[-1]
    orig_dtype = posemb.dtype
    posemb = posemb.float()  # interpolate needs float32
    posemb = posemb.reshape(1, old_size[0], old_size[1], -1).permute(0, 3, 1, 2)
    if not use_cpe:
        posemb = F.interpolate(posemb, size=(max(new_size), max(new_size)), mode=interpolation, antialias=antialias)
    posemb = posemb[:, :, :new_size[0], :new_size[1]]  # if the image is not square, only take part of the pos emb
    posemb = posemb.permute(0, 2, 3, 1).reshape(1, -1, embed_dim)
    posemb = posemb.to(orig_dtype)

    # add back extra (class, etc) prefix tokens
    if posemb_prefix is not None:
        posemb = torch.cat([posemb_prefix, posemb], dim=1)

    return posemb


def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
    """
    Adapted from timm vision_transformer.py: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    Original license: Apache-2.0, Copyright 2020, Ross Wightman
    
    Adapted to support cropped position embedding (cpe).
    """

    if self.pos_embed is None:
        return x.view(x.shape[0], -1, x.shape[-1])

    if self.dynamic_img_size:
        B, H, W, C = x.shape
        prev_grid_size = self.patch_embed.grid_size
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            new_size=(H, W),
            old_size=prev_grid_size,
            num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            use_cpe=self.radio,
        )
        x = x.view(B, -1, C)
    else:
        pos_embed = self.pos_embed

    to_cat = []
    if self.cls_token is not None:
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
    if self.reg_token is not None:
        to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

    if self.no_embed_class:
        # deit-3, updated JAX (big vision)
        # position embedding does not overlap with class token, add then concat
        x = x + pos_embed
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
    else:
        # original timm, JAX, and deit vit impl
        # pos_embed has entry for class token, concat then add
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        x = x + pos_embed

    return self.pos_drop(x)


def selected_pos_embed(self, x: torch.Tensor, selection_map, im_H, im_W) -> torch.Tensor:
    """
    Adapted from timm vision_transformer.py: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    Original license: Apache-2.0, Copyright 2020, Ross Wightman
    
    Adapted to support position embedding on selected patches only and cropped position embedding (cpe).
    """

    if self.pos_embed is None:
        return x.view(x.shape[0], -1, x.shape[-1])

    if self.dynamic_img_size:
        B, C = x.shape[0], x.shape[-1]
        H, W = im_H, im_W
        pos_embed = resample_abs_pos_embed(
            self.pos_embed,
            (H, W),
            num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            use_cpe=self.radio,
        )
        x = x.view(B, -1, C)
    else:
        pos_embed = self.pos_embed

    to_cat = []
    if self.cls_token is not None:
        to_cat.append(self.cls_token.expand(x.shape[0], -1, -1))
    if self.reg_token is not None:
        to_cat.append(self.reg_token.expand(x.shape[0], -1, -1))

    if self.no_embed_class:
        # deit-3, updated JAX (big vision)
        # position embedding does not overlap with class token, add then concat
        selection_map = selection_map.flatten(1)
        pos_embed = pos_embed.repeat(B, 1, 1)[selection_map.bool()].view(B, -1, pos_embed.shape[-1])
        x = x + pos_embed
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
    else:
        # original timm, JAX, and deit vit impl
        # pos_embed has entry for class token, concat then add
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        selection_map = selection_map.flatten(1)
        pos_embed_no_prefix = pos_embed[:, len(to_cat):].repeat(B, 1, 1)[selection_map.bool()].view(B, -1, pos_embed.shape[-1])
        pos_embed = torch.cat([pos_embed[:, :len(to_cat)].repeat(B, 1, 1), pos_embed_no_prefix], dim=1)
        x = x + pos_embed

    return self.pos_drop(x)



class Attention_w_KVCache(nn.Module):
    """
    Adapted from timm vision_transformer.py: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    Original license: Apache-2.0, Copyright 2020, Ross Wightman
    
    Adapted to support attention with low-res KV cache.
    """

    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, kv_cache=None, return_kv_cache=False) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if kv_cache is not None:
            k_cache, v_cache = kv_cache[0], kv_cache[1]
            k = torch.cat([k_cache, k], dim=-2)
            v = torch.cat([v_cache, v], dim=-2)
        
        if return_kv_cache:
            output_kv_cache = torch.stack([k, v], dim=0)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_kv_cache:
            return x, output_kv_cache
        else:
            return x
    

class Block_w_KVCache(nn.Module):
    """
    Adapted from timm vision_transformer.py: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    Original license: Apache-2.0, Copyright 2020, Ross Wightman
    
    Adapted to support attention with low-res KV cache.
    """

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention_w_KVCache(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            bias=proj_bias,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, kv_cache=None, return_kv_cache=False) -> torch.Tensor:
        if return_kv_cache:
            z, output_kv_cache = self.attn(self.norm1(x), kv_cache=kv_cache, return_kv_cache=True)
            x = x + self.drop_path1(self.ls1(z))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x, output_kv_cache
        else:
            x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), kv_cache=kv_cache)))
            x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
            return x


def forward_tokenize(self, x, selection_map=None, im_H=None, im_W=None):
    if selection_map is None:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
    else:
        x = self.patch_embed(x, selection_map)
        x = self.selected_pos_embed(x, selection_map, im_H, im_W)
    return x


def forward_after_tokenize_w_kvcache(self, x, kv_cache=None, return_kv_cache=False, output_hidden_states=False, pool_mask=None):
    """
    Adapted from timm vision_transformer.py: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
    Original license: Apache-2.0, Copyright 2020, Ross Wightman

    Forward pass of the original timm ViT, but only the part after tokenization (patch_embed and _pos_embed).
    """

    # forward features
    x = self.patch_drop(x)
    x = self.norm_pre(x)

    hidden_states = []
    output_kv_cache = []
    for i, blk in enumerate(self.blocks):
        if self.grad_checkpointing and not torch.jit.is_scripting():
            if return_kv_cache:
                x, output_kv_cache_layeri = checkpoint(blk, x, kv_cache[i] if kv_cache is not None else None, True)
            else:
                x = checkpoint(blk, x, kv_cache[i] if kv_cache is not None else None)
        else:
            if return_kv_cache:
                x, output_kv_cache_layeri= blk(x, kv_cache[i] if kv_cache is not None else None, return_kv_cache=True)
            else:
                x = blk(x, kv_cache[i] if kv_cache is not None else None)
        if output_hidden_states:
            hidden_states.append(x)
        if return_kv_cache:
            output_kv_cache.append(output_kv_cache_layeri)

    x = self.norm(x)

    if self.radio:
        x = self.siglip_proj(x)

    # forward head
    assert self.attn_pool is not None
    if self.attn_pool is not None:
        x = self.attn_pool(x, pool_mask)
    elif self.global_pool == 'avg':
        x = x[:, self.num_prefix_tokens:].mean(dim=1)
    elif self.global_pool:
        x = x[:, 0]  # class token
    x = self.fc_norm(x)
    x = self.head_drop(x)

    outs = {
        "x": self.head(x),
        "hidden_states": hidden_states if output_hidden_states else None,
        "output_kv_cache": torch.stack(output_kv_cache, dim=0) if return_kv_cache else None,
    }
    
    return outs


def forward_attn_pool_with_mask(self, x, mask=None):
    """
    Adapted from timm attention_pool.py: https://github.com/huggingface/pytorch-image-models/blob/main/timm/layers/attention_pool.py
    Original license: Apache-2.0, Copyright 2020, Ross Wightman
    
    Adapted to support attention pooling only on a masked (selected) region.
    """

    # mask: (B, N)

    B, N, C = x.shape

    if mask is not None:
        mask[mask.sum(dim=-1) == 0] = 1

    if self.pos_embed is not None:
        # FIXME interpolate
        x = x + self.pos_embed.unsqueeze(0).to(x.dtype)

    q_latent = self.latent.expand(B, -1, -1)
    q = self.q(q_latent).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)

    kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)

    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask.bool().reshape(B, 1, 1, N) if mask is not None else None)
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            attn_bias = torch.zeros(B, 1, 1, N, dtype=q.dtype)
            attn_bias.masked_fill_(mask.bool().logical_not().reshape(B, 1, 1, N), float("-inf"))
            attn += attn_bias
        attn = attn.softmax(dim=-1)
        x = attn @ v
    x = x.transpose(1, 2).reshape(B, self.latent_len, C)
    x = self.proj(x)
    x = self.proj_drop(x)

    x = x + self.mlp(self.norm(x))

    # optional pool if latent seq_len > 1 and pooled output is desired
    if self.pool == 'token':
        x = x[:, 0]
    elif self.pool == 'avg':
        x = x.mean(1)
    return x
