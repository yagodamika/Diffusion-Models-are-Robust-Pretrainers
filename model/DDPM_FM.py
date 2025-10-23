
import torch
from diffusers import DDPMPipeline
from typing import List, Optional, Tuple, Union
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers import UNet2DModel

class DDPM_FM(DDPMPipeline): 
    """
    def __init__(self,unet, scheduler): 
        #DDPMPipeline.__init__(self, unet, scheduler)
        super().__init__(unet, scheduler)
    """
            
    #@torch.no_grad()
    def get_features(
        self,
        x: torch.Tensor,
        t: int,
        blocknum: int,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, 
        add_noise: bool=True
    ) -> torch.FloatTensor:
        if add_noise:
            # A. Sample gaussian noise 
            if isinstance(self.unet.config.sample_size, int):
                image_shape = (
                    batch_size,
                    self.unet.config.in_channels,
                    self.unet.config.sample_size,
                    self.unet.config.sample_size,
                )
            else:
                image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

            if self.device.type == "mps":
                # randn does not work reproducibly on mps
                noise = randn_tensor(image_shape, generator=generator)
                noise = noise.to(self.device)
            else:
                noise = randn_tensor(image_shape, generator=generator, device=self.device)


            # B. get x_t from x and noise 
            x_t = self.scheduler.add_noise(x, noise, torch.tensor([t],  dtype=torch.int64).to(self.device))
            input = x_t
        else:
            input = x

        # C. get features from passing x_t through Unet
        output = self.unet.get_features(input, blocknum=blocknum, timestep=t).sample       
        
        return output
