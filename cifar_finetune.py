import argparse
import torch
import numpy as np
import sys
import os 
import model.Head as Head
from model.dimension_dict import *
from model.DDPM_FM import *
from model.UNET_FM import UNet_FM
from model.FullModel import DiffSSLModel
from torchvision.datasets import *
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm 
from helpers import *
from pgd import *
from AutoAttack.attacks.autoattack import AutoAttack
from TRADES.trades import trades_loss
import torchattacks


def create_argparser():
    parser = argparse.ArgumentParser()
    
    # Dataset Params 
    parser.add_argument("--num_classes", type=int, default=10) # class number of dataset
    parser.add_argument("--dataset_name", type=str, default="cifar10") # dataset name
    parser.add_argument("--cifar10_path", type=str, default="/disk5/datasets/cifar10")
    parser.add_argument("--batch_size", type=int, default=32) 
    parser.add_argument("--num_workers", type=int, default=4) 
    parser.add_argument("--momentum", type=int, default=0.9) 
    parser.add_argument("--transforms", type=bool, default=False) 

    # General Params
    parser.add_argument("--head_type", type=str, default="attention") # linear, attention
    parser.add_argument("--blocknum", type=int, default=1) # block number to extract from unet
    parser.add_argument("--t", type=int, default=5) # timestep 
    parser.add_argument("--output_dir", type=str, default="/home/mika/mika/project/try") # output dir  
    parser.add_argument("--only_eval", type=bool, default=False) 
    parser.add_argument("--resume_checkpoint", type=str, default=None) 
    parser.add_argument("--cache_dir", type=str, default="/disk5/mika/cache")
    
    # Training Params
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=30)
    
    #TRADES
    parser.add_argument("--trades", type=bool, default=False) 
    parser.add_argument("--trades_beta", type=int, default=5) 
    
    # Attack Params
    parser.add_argument("--attack", type=str, default=None) # "pgd", "autoattack"
    parser.add_argument("--attack_iters_train", type=int, default=2)
    parser.add_argument("--attack_iters_test", type=int, default=10)
    parser.add_argument("--attack_norm", type=str, default="l_inf")
    parser.add_argument("--alpha_test", type=float, default=2/255)
    parser.add_argument("--epsilon_test", type=float, default=8/255)
    parser.add_argument("--alpha_train", type=float, default=1/255)
    parser.add_argument("--epsilon_train", type=float, default=1/255)
    parser.add_argument("--aa_eps", type=float, default=8/255) # AutoAttack epsilon

    # Linear Head Params
    parser.add_argument("--pool_size1", type=int, default=1) # pooling sizes for linear head
    parser.add_argument("--pool_size2", type=int, default=1) # pooling sizes for linear head
    
    # Attention Head Params
    parser.add_argument("--num_heads", type=int, default=8) 
    parser.add_argument("--mlp_ratio", type=int, default=4) # MLP hidden layer size
    parser.add_argument("--num_blocks", type=int, default=2) # Number of blocks of Attn + MLP
    parser.add_argument("--pre_pool_size", type=int, default=16)
    parser.add_argument("--norm_type", type=str, default="layer") # normalization type: "layer", "batch"
    #parser.add_argument("--attention_dim", type=int, default=64) #128 should be divisible by num heads
    

    return parser
           
def train(args, model, train_dataloader, test_dataloader):
    
    # Define parameters to optimize
    optimized_params_lst = [{'params': model.head.parameters()}]
    optimizer = torch.optim.SGD(optimized_params_lst, lr=args.lr, momentum=args.momentum)
    loss_fn = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 7, 0.1)
    
    if args.resume_checkpoint != None:
        checkpoint_dict = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
        start_epoch = checkpoint_dict['epoch']
    else:
        start_epoch = 0
    
    losses = []
    model.train()
    batch_num = 0
    
    if args.attack == "autoattack":
        autoattack = AutoAttack(model, norm='Linf', eps=args.aa_eps)
    
    print("Starting training")
    for epoch in range(start_epoch, args.epochs):
        print(f"starting epoch {epoch}")
        #for batch in (tqdm(train_dataloader, total=len(train_dataloader))):
        for batch in train_dataloader:
            
            imgs, targets = batch 
            imgs = imgs.to(args.device)
            targets = targets.to(args.device) 

            if args.attack != None and args.trades==False:
                set_requires_grad("all", model, True)
                if args.attack == "pgd":
                    imgs = attack_pgd(args, loss_fn, imgs, targets, args.alpha_train,
                                    args.attack_iters_train, args.attack_norm,
                                    args.device, model, 0, 1, args.epsilon_train)
                elif args.attack == "autoattack":
                    imgs = autoattack(imgs, targets)

                set_requires_grad("diffusion", model, False)

            output = model(imgs, args.t)
            
            if args.trades:
                loss = trades_loss(model=model,t=args.t,
                           x_natural=imgs,
                           y=targets,
                           optimizer=optimizer,
                           step_size=args.alpha_train,
                           epsilon=args.epsilon_train,
                           perturb_steps=args.attack_iters_train,
                           beta=args.trades_beta,
			               distance='l_inf')
            else:    
                loss = loss_fn(output, targets)
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if len(losses) == 100:
                losses = losses[1:]
            losses.append(loss.item())
            
            batch_num += 1
            
        scheduler.step()
        test(model, test_dataloader, args, epoch+1) 
        
        # Save checkpoint every epoch
        save_file = os.path.join(args.output_dir, f'epoch_{epoch+1}.pth')
        print(f"Saving checkpoint @ Epoch: {epoch+1} to {save_file}")
        save_dict ={
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch+1
                    }
        save_dict['model_head'] = model.head.state_dict()       
        torch.save(save_dict, os.path.join(args.output_dir, f'epoch_{epoch+1}.pth'))


def test(model, dataloader, args, epoch=0):
    print("Starting test")
    criterion = torch.nn.CrossEntropyLoss()
    
    model.eval()
    num_correct = 0
    total = 0

    num_val_batches = len(dataloader)
    if args.attack!=None:
        set_requires_grad("all", model, True)
    if args.attack == "autoattack":
        autoattack = AutoAttack(model, norm='Linf', eps=args.aa_eps)
    if args.attack == "fgsm":
        atk = torchattacks.FGSM(model, eps=8/255)
    if args.attack == "bim":
        atk = torchattacks.BIM(model, eps=8/255, alpha=2/255, steps=10)
    if args.attack == "pgd_10":
        atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=10, random_start=True)
    if args.attack == "pgd_10_original":
        atk = torchattacks.PGD(model, eps=1/255, alpha=1/255, steps=10, random_start=True)
    if args.attack == "pgd_20":
        atk = torchattacks.PGD(model, eps=8/255, alpha=2/255, steps=20, random_start=True)
    if args.attack == "cw":
        atk = torchattacks.CW(model, c=0.1, kappa=0, steps=1000, lr=0.01)
    if args.attack == "fab": 
        atk = torchattacks.FAB(model, norm='Linf', eps=8/255, steps=50, n_restarts=1)
    if args.attack == "apgd":
        atk = torchattacks.APGD(model, norm='Linf', eps=8/255, steps=100, n_restarts=1, loss='ce')

    #for batch in tqdm(dataloader, total=num_val_batches):
    for batch in dataloader:
        imgs, targets = batch
        imgs = imgs.to(args.device)
        targets = targets.to(args.device)
        
        if args.attack != None:
            if args.attack == "pgd":
                imgs = attack_pgd(args, criterion, imgs, targets, args.alpha_test,
                        args.attack_iters_test, args.attack_norm, args.device, model, 0, 1, args.epsilon_test)
            if args.attack == "autoattack":
                imgs = autoattack(imgs, args.t, targets)
            if args.attack in ["fgsm","bim","pgd_10","pgd_20", "cw", "fab", "apgd"]:
                imgs = atk(imgs, targets)

                
        with torch.no_grad():
            output = model(imgs, args.t)
            pred = torch.argmax(output, dim=1)

        num_correct += (pred == targets).sum()
        total += pred.shape[0]

    print(f'accuracy: {num_correct / total}, Num correct: {num_correct}, Total: {total}')


def main():
    
    print("Parsing arguments")
    args = create_argparser().parse_args()
    args.device = 'cuda'
    
    # Output directory
    print("Creating output directory")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get feature dimensions dictionary
    if args.dataset_name == "cifar10":
        dim_dict = feature_dims_cifar10
        model_id = "google/ddpm-cifar10-32"
        preprocess = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15), 
        transforms.ToTensor()
        ])
        if args.transforms:
            transform_train = transforms.Compose([ transforms.RandomCrop(32, padding=4), 
                                                transforms.RandomHorizontalFlip(), 
                                                transforms.ToTensor(), 
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                     (0.2023, 0.1994, 0.2010)), ]) 
            transform_test = transforms.Compose([ transforms.ToTensor(), 
                                                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                    (0.2023, 0.1994, 0.2010)), ]) 
        preprocess_train = transform_train if args.transforms else preprocess
        preprocess_test = transform_test if args.transforms else preprocess
        
        print("Initializing datasets")
        if not args.only_eval:
            train_data = CIFAR10(args.cifar10_path, transform=preprocess_train,
                                                download=False, train=True)
        test_data = CIFAR10(args.cifar10_path, transform=preprocess_test,
                                            download=False, train=False)
    else:
        raise ValueError("Invalid dataset name provided")
    
    # Build classification head
    print("Initializing classification head")
    if args.head_type =="linear":
        print("Initializing linear head")
        head = Head.LinearHead(args, DM_FEAT_DIM_DICT)
    elif args.head_type == "attention":
        print("Initializing attention head")
        head = Head.AttentionHead(args, DM_FEAT_DIM_DICT, DM_FEAT_SIZE_DICT,
                                  args.num_heads, args.mlp_ratio, args.num_blocks)
    else:
        raise NotImplementedError
    head = head.to(args.device)

    # Get diffusion model
    print("Initializing diffusion model")
    ddpm = DDPM_FM.from_pretrained(model_id, cache_dir=args.cache_dir)  
    ddpm.unet.__class__ = UNet_FM #TODO: make sure that the weights of UNET stay the same
    ddpm = ddpm.to(args.device)
    
    if args.only_eval:
        mode = "freeze"
    else:
        mode = "finetune"
        
    # Define whole model - DiffSSL model
    model = DiffSSLModel(ddpm, head, args.device, mode, args.blocknum, t=args.t)
    
    # Load Checkpoint
    if args.resume_checkpoint != None:
        if os.path.exists(args.resume_checkpoint):
            print("Loading checkpoint from ", args.resume_checkpoint)
            state_dict = torch.load(args.resume_checkpoint, map_location=torch.device('cpu'))
            model.head.load_state_dict(state_dict['model_head'])
        
    # Load dataloaders
    print("Initializing dataloaders")
    if not args.only_eval:
        train_loader = DataLoader(train_data,
                                  batch_size=args.batch_size, pin_memory=True,
                                  num_workers=args.num_workers, shuffle=False, sampler=None) 
    test_loader = DataLoader(test_data,
                                  batch_size=args.batch_size, pin_memory=True,
                                  num_workers=args.num_workers, shuffle=False, sampler=None) 
    
    # Train / Test
    if not args.only_eval: 
        train(args, model, train_loader, test_loader)
        test(model, test_loader, args)
    else:
        test(model, test_loader, args)

if __name__ == "__main__":
    main()
    
    
    
    