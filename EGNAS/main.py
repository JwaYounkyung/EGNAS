import argparse
import time
import torch
import configs
import tensor_utils as utils
from population import Population
import os
import time
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"

def main(args):
    if args.cuda and not torch.cuda.is_available():  # cuda is not available
        args.cuda = False
    
    if args.cuda:
        torch.cuda.set_device(args.gpu)
        utils.set_random_seed(args.random_seed)
    
    begin_time = time.time()
    pop = Population(args)
    if args.combined_evolution:
        pop.evolve_net_combined()
    else:
        pop.evolve_net()
    print('entire experiment time: %.2f min' %((time.time() - begin_time)/60))
    
if __name__ == "__main__":
    args = configs.build_args('GeneticGNN')
    print(args)
    main(args)