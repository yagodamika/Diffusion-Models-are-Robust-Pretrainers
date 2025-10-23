
FEAT_DIM_LIST = [128]*1 + [256]*7 + [128]*1 
FEAT_SIZE_LIST = [16]*1 + [8]*1 + [4]*3 + [8]*1 + [16]*1+ [32]*2 

DM_FEAT_DIM_DICT = {}
DM_FEAT_SIZE_DICT = {}
for idx, val in enumerate(FEAT_DIM_LIST):
    DM_FEAT_DIM_DICT[idx+1] = val

for idx, val in enumerate(FEAT_SIZE_LIST):
    DM_FEAT_SIZE_DICT[idx+1] = val
    
feature_dims_cifar10 = {
    1: [128, 16, 16],
    2: [256, 8, 8],
    3: [256, 4, 4],
    4: [256, 4, 4],
    5: [256, 4, 4],
    6: [256, 8, 8],
    7: [256, 16, 16],
    8: [256, 32, 32],
    9: [128, 32, 32]
				}