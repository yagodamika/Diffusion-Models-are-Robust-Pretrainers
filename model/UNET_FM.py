from diffusers import UNet2DModel
from diffusers import UNet2DModel 
from typing import List, Optional, Tuple, Union
import torch
from dataclasses import dataclass
from diffusers.models.unets.unet_2d import UNet2DOutput

class UNet_FM(UNet2DModel):
    
    def __init__(self,
					sample_size: Optional[Union[int, Tuple[int, int]]] = None,
					in_channels: int = 3,
					out_channels: int = 3,
					center_input_sample: bool = False,
					time_embedding_type: str = "positional",
					freq_shift: int = 0,
					flip_sin_to_cos: bool = True,
					down_block_types: Tuple[str] = ("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
					up_block_types: Tuple[str] = ("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
					block_out_channels: Tuple[int] = (224, 448, 672, 896),
					layers_per_block: int = 2,
					mid_block_scale_factor: float = 1,
					downsample_padding: int = 1,
					downsample_type: str = "conv",
					upsample_type: str = "conv",
					dropout: float = 0.0,
					act_fn: str = "silu",
					attention_head_dim: Optional[int] = 8,
					norm_num_groups: int = 32,
					attn_norm_num_groups: Optional[int] = None,
					norm_eps: float = 1e-5,
					resnet_time_scale_shift: str = "default",
					add_attention: bool = True,
					class_embed_type: Optional[str] = None,
					num_class_embeds: Optional[int] = None,): 
        UNet2DModel.__init__(self, 
								sample_size,
								in_channels,
								out_channels,
								center_input_sample,
								time_embedding_type,
								freq_shift,
								flip_sin_to_cos,
								down_block_types,
								up_block_types,
								block_out_channels,
								layers_per_block,
								mid_block_scale_factor,
								downsample_padding,
								downsample_type,
								upsample_type,
								dropout,
								act_fn,
								attention_head_dim,
								norm_num_groups,
								attn_norm_num_groups,
								norm_eps,
								resnet_time_scale_shift,
								add_attention,
								class_embed_type,
								num_class_embeds)
    
    def get_features(
        self,
        sample: torch.FloatTensor,
        blocknum: int,
        timestep: Union[torch.Tensor, float, int],
        class_labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet2DOutput, Tuple]:
        r"""
        The [`UNet2DModel`] forward method.

        Args:
            sample (`torch.FloatTensor`):
                The noisy input tensor with the following shape `(batch, channel, height, width)`.
            timestep (`torch.FloatTensor` or `float` or `int`): The number of timesteps to denoise an input.
            class_labels (`torch.FloatTensor`, *optional*, defaults to `None`):
                Optional class labels for conditioning. Their embeddings will be summed with the timestep embeddings.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d.UNet2DOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d.UNet2DOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_2d.UNet2DOutput`] is returned, otherwise a `tuple` is
                returned where the first element is the sample tensor.
        """
        # 0. center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps * torch.ones(sample.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when doing class conditioning")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # 2. pre-process
        skip_sample = sample
        sample = self.conv_in(sample)


        bn = 0
        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            bn += 1
            if hasattr(downsample_block, "skip_conv"):
                sample, res_samples, skip_sample = downsample_block(
                    hidden_states=sample, temb=emb, skip_sample=skip_sample
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
			
            if bn==blocknum:
                return UNet2DOutput(sample=sample)
   
            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample, emb)
        bn += 1
        if bn==blocknum:
            return UNet2DOutput(sample=sample)

        # 5. up
        skip_sample = None
        for upsample_block in self.up_blocks:
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "skip_conv"):
                sample, skip_sample = upsample_block(sample, res_samples, emb, skip_sample)
            else:
                sample = upsample_block(sample, res_samples, emb)
            bn += 1
            if bn==blocknum:
                return UNet2DOutput(sample=sample)

        raise ValueError("Block number given is not compatible with model")