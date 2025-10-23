## Diffusion Models are Robust Pretrainers

Official implementation of the Signal Processing Letters paper: "Diffusion Models as Robust Pretrainers for Classification and Detection".

We study models built on top of off-the-shelf diffusion models and demonstrate their practical significance: 
they provide a low-cost path to robust representations, allowing lightweight heads to be trained on frozen features without full adversarial training. 
Our empirical evaluations on ImageNet, CIFAR-10, and PASCAL VOC show that diffusion-based classifiers and detectors achieve meaningful adversarial robustness with minimal compute.

## Installation
Here is a list of libraries you need to install to execute the code:
* pytorch 
* argparse
* numpy
* torchvision
* tqdm
* torchattacks
* einops
* diffusers
* typing
* dataclasses

## Run Training
To run attention head model training on CIFAR-10:

```python cifar_finetune.py --head_type attention --batch_size 32 --blocknum 1 --t 10 --output_dir <dir> --lr 1e-2 --epochs 100 --num_heads 8 --mlp_ratio 4 --num_blocks 2 --pre_pool_size 16 --norm_type layer```

To run linear head model training on CIFAR-10:

```python cifar_finetune.py --head_type linear --batch_size 32 --blocknum 1 --t 10 --output_dir <dir> --lr 1e-2 --epochs 20 --pool_size1 1 --pool_size2 1```

You can change the arguments as you wish. 

## Run Test
To run evaluation on CIFAR-10:

```python cifar_finetune.py --head_type attention --batch_size 32 --blocknum 1 --t 10 --lr 1e-2 --epochs 20 --num_heads 8 --mlp_ratio 4 --num_blocks 2 --pre_pool_size 16 --norm_type layer --only_eval True --attack apgd --attack_iters_test 10 --attack_norm l_inf --resume_checkpoint <model dictionary file directory>```
    
Here for instance we test the model under the apgd attack with L infinity norm. 
Make sure to add the flag "--only_eval True" and "--resume_checkpoint" with the directory of the dictionary of the model you want to evaluate.  
