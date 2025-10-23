from torch import nn

class DiffSSLModel(nn.Module):
    def __init__(self, diffusion, head, device, mode, blocknum, add_noise=True, t=None):
        super().__init__()
        self.diffusion = diffusion
        self.head = head
        self.mode = mode
        self.device = device
        self.blocknum = blocknum
        self.add_noise = add_noise
        assert self.mode in ['freeze', "finetune"], f"Mode {self.mode} not supported"
        
        if t !=None:
            self.t = t
        else: 
            self.t = None
            
        if self.mode == 'freeze': 
            for param in self.diffusion.unet.parameters(): #should be encoder? diffusion?
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = False
        elif self.mode == 'finetune': 
            for param in self.diffusion.unet.parameters(): #should be encoder? diffusion?
                param.requires_grad = False
            for param in self.head.parameters():
                param.requires_grad = True
    
    def forward(self, x, t=None):
        if self.t != None:
            timestep = self.t
        else:
            timestep = t
        
        x = self.diffusion.get_features(x, timestep, self.blocknum, add_noise = self.add_noise)
        x = self.head(x)
        return x
        