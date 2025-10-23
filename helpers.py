
def set_requires_grad(part_name ,model, bool):
    if part_name == "diffusion":
        for param in model.diffusion.unet.parameters():
            param.requires_grad = bool
    if part_name == "head":
        for param in model.head.parameters():
            param.requires_grad = bool
    if part_name =="all":
        for param in model.head.parameters():
            param.requires_grad = bool
        for param in model.diffusion.unet.parameters():
            param.requires_grad = bool

