import torch
import random
import numpy as np
import utils


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main():
    args, unknown = utils.parse_args()
    
    set_seed(args.seed)
    from .models.HPoolGCL import PoolGCL_ModelTrainer
    embedder =HPoolGCL_ModelTrainer(args)

    embedder.train()
    

if __name__ == "__main__":
    main()


