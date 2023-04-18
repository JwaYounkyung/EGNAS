import argparse
import time
import torch
import configs
import tensor_utils as utils
from population import Population
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main(args):
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.random_seed)

    utils.makedirs(args.dataset)  
    
    print(args.super_ratio)
    
    pop = Population(args)
    pop.evolve_net()

#     # run on single model
#     num_epochs = 200
#     actions = ['gcn', 'mean', 'softplus', 16, 8, 'gcn', 'max', 'tanh', 16, 6] 
#     pop.single_model_run(num_epochs, actions)
    
    
if __name__ == "__main__":
    args = configs.build_args('GeneticGNN')
    main(args)