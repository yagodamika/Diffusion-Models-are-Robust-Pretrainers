from torch import nn

class BaselineModel(nn.Module):
    def __init__(self, diffusion, head, device, mode, blocknum):
        super().__init__()
        self.diffusion = diffusion
        self.head = head
        self.mode = mode
        self.device = device
        self.blocknum = blocknum
        assert self.mode in ['freeze', "finetune"], f"Mode {self.mode} not supported"
        
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
    
    def forward(self, x, t):
        x = self.diffusion.get_features(x, t, self.blocknum)
        x = self.head(x)
        return x
        