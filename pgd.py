import torch

def transform_range(image):
    """
    Transforms an image in range [-1,1]
    into range [0,1]
    """
    new_image = (image + 1.0) / 2.0
    return new_image

def range_to_minus_plus(image):
    """
    Transforms an image in range [0,1]
    into range [-1,1]
    """
    new_image = (image * 2.0) - 1
    return new_image

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(args, criterion, img, target, alpha,
               attack_iters, norm, device, model, lower, upper, epsilon=0, targeted=False):
    #(vae, tokenizer, text_encoder, unet, scheduler) = model
    #img = transform_range(img)
    with torch.enable_grad():
        delta = torch.zeros_like(img).to(device)
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(delta.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower - img, upper - img)
        delta.requires_grad = True
        for _ in range(attack_iters):
            # output = model(normalize(X ))

            adv_images = img + delta
            #output, _ = classifier(adv_images, text_prompts)
            #adv_images = range_to_minus_plus(adv_images)
            output = model(adv_images, args.t)
            if targeted:
                loss = -criterion(output, torch.tensor(target).to(device))
            else:
                loss = criterion(output, torch.tensor(target).to(device))

            loss.backward()
            grad = delta.grad.detach()
            d = delta[:, :, :, :]
            g = grad[:, :, :, :]
            x = img[:, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0], -1), dim=1).view(-1, 1, 1, 1)
                scaled_g = g / (g_norm + 1e-10)
                d = (d + scaled_g * alpha).view(d.size(0), -1).renorm(p=2, dim=0, maxnorm=epsilon).view_as(d)
            d = clamp(d, lower - x, upper - x)
            delta.data[:, :, :, :] = d
            delta.grad.zero_()
    output = (img + delta).clone().detach()
    #output.requires_grad=False
    #output = range_to_minus_plus(output)
    return output