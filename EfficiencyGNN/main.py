import argparse
import time
import torch
import configs
import tensor_utils as utils
from population import Population
import os
import time
#os.environ['KMP_DUPLICATE_LIB_OK']='True'

def main(args):
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        utils.set_random_seed(args.random_seed)

    utils.makedirs(args.dataset)  
    
    print(args.super_ratio)
    
    begin_time = time.time()
    pop = Population(args)
    pop.evolve_net()
    print('entire experiment time: %.2f min' %((time.time() - begin_time)/60))

#     # run on single model
#     num_epochs = 200
#     actions = ['gcn', 'mean', 'softplus', 16, 8, 'gcn', 'max', 'tanh', 16, 6] 
#     pop.single_model_run(num_epochs, actions)
    
    
if __name__ == "__main__":
    args = configs.build_args('GeneticGNN')
    main(args)