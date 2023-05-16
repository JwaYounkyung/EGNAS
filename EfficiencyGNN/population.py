from search_space import HybridSearchSpace
from individual import Individual
from super_individual import Super_Individual
from random import sample, choices
import numpy as np
from gnn_model_manager import GNNModelManager
import copy
import random
import time

class Population(object):
    
    def __init__(self, args):
        
        self.args = args
        hybrid_search_space = HybridSearchSpace(self.args.num_gnn_layers)
        self.hybrid_search_space = hybrid_search_space
        
        # prepare data set for training the gnn model
        self.load_trining_data()
    
    def load_trining_data(self):
        self.gnn_manager = GNNModelManager(self.args)
        self.gnn_manager.load_data(self.args.dataset)
        
        # dataset statistics
        print(self.gnn_manager.data)
 
    def init_population(self):
        
        struct_individuals = []
        
        for i in range(self.args.num_individuals):
            net_genes = self.hybrid_search_space.get_net_instance()
            if self.args.num_gnn_layers == 2:
                param_genes = [self.args.in_drop, self.args.in_drop, self.args.lr, self.args.weight_decay]
            elif self.args.num_gnn_layers == 3:
                param_genes = [self.args.in_drop, self.args.in_drop, self.args.in_drop, self.args.lr, self.args.weight_decay]

            instance = Individual(net_genes, param_genes)
            struct_individuals.append(instance)
        
        self.struct_individuals = struct_individuals

        if self.args.num_gnn_layers == 2:
            param_genes = [self.args.in_drop, self.args.in_drop, self.args.lr, self.args.weight_decay]
        elif self.args.num_gnn_layers == 3:
            param_genes = [self.args.in_drop, self.args.in_drop, self.args.in_drop, self.args.lr, self.args.weight_decay]
        params_individuals = [param_genes]
        for j in range(self.args.num_individuals_param-1):
            param_genes = self.hybrid_search_space.get_param_instance()
            params_individuals.append(param_genes)
    
        self.params_individuals = params_individuals


    def init_combined_population(self):
        
        struct_individuals = []
        threshold = self.args.num_individuals * self.args.param_initial_rate
        
        for i in range(self.args.num_individuals):
            net_genes = self.hybrid_search_space.get_net_instance()
            
            # initialize the parameters
            if i < threshold:
                if self.args.num_gnn_layers == 2:
                    param_genes = [self.args.in_drop, self.args.in_drop, self.args.lr, self.args.weight_decay]
                elif self.args.num_gnn_layers == 3:
                    param_genes = [self.args.in_drop, self.args.in_drop, self.args.in_drop, self.args.lr, self.args.weight_decay]
            else:
                param_genes = self.hybrid_search_space.get_param_instance()
            
            instance = Individual(net_genes, param_genes)
            struct_individuals.append(instance)
        
        self.struct_individuals = struct_individuals


    def init_param_population(self, init_individuals):
        super_population = []
        
        init_pop = Super_Individual(self.args.num_individuals, self.args.super_ratio, init_individuals, self.params_individuals[0])
        init_pop.cal_superfitness()
        super_population.append(init_pop)
        for i in range(self.args.num_individuals_param-1):
            individuals = copy.deepcopy(init_individuals)
            param_genes = self.params_individuals[i+1]
            for i in range(self.args.num_individuals): 
                individuals[i].param_genes = param_genes
                individuals[i].cal_fitness(self.gnn_manager)
            new_pop = Super_Individual(self.args.num_individuals, self.args.super_ratio, individuals, param_genes)
            new_pop.cal_superfitness()
            super_population.append(new_pop)
            
        self.super_population = super_population
                    
    
    # run on the single model with more training epochs
    def single_model_run(self, num_epochs, actions):
        self.args.epochs = num_epochs
        self.gnn_manager.train(actions)
        
        
    def cal_fitness(self, individuals=None):
        """calculate fitness scores of all individuals,
          e.g., the classification accuracy from GNN"""
        if individuals is None:
            individuals = self.struct_individuals

        for individual in individuals:
            individual.cal_fitness(self.gnn_manager)
        
        return individuals
            
    def parent_selection(self):
        "select k individuals by fitness probability"
        k = self.args.num_parents
        
        # select the parents for structure evolution
        fitnesses = [i.get_fitness() for i in self.struct_individuals]
        fit_probs = fitnesses / np.sum(fitnesses)
        struct_parents = choices(self.struct_individuals, k=k, weights=fit_probs)
        
        return struct_parents
    
    def parent_selection_param(self):
        "select k individuals by fitness probability"
        k = self.args.num_parents_param
        
        # select the parents for structure evolution
        fitnesses = [i.get_fitness() for i in self.super_population]
        fit_probs = fitnesses / np.sum(fitnesses)
        param_parents = choices(self.super_population, k=k, weights=fit_probs)
        
        return param_parents
    
    def crossover_net(self, parents):  
        "produce offspring from parents for better net architecture"
        p_size = len(parents)
        maximum = p_size * (p_size - 1) / 2
        if self.args.num_offsprings > maximum:
            raise RuntimeError("number of offsprings should not be more than " 
                               + maximum)
            
        # randomly choose crossover parent pairs
        parent_pairs = []
        while len(parent_pairs) < self.args.num_offsprings:
            indexes = sample(range(p_size), k=2)
            pair = (indexes[0], indexes[1])
            if indexes[0] > indexes[1]:
                pair = (indexes[1], indexes[0])
            if not pair in parent_pairs:
                parent_pairs.append(pair)
        
        # crossover to generate offsprings
        offsprings = []
        gene_size = len(parents[0].get_net_genes())
        for i, j in parent_pairs:
            parent_gene_i = parents[i].get_net_genes()
            parent_gene_j = parents[j].get_net_genes()
            # select a random crossover point
            point_index = parent_gene_j.index(sample(parent_gene_j, 1)[0]) # possible 0
            
            offspring_gene_i = parent_gene_i[:point_index]
            offspring_gene_i.extend(parent_gene_j[point_index:])
            offspring_gene_j = parent_gene_j[:point_index]
            offspring_gene_j.extend(parent_gene_i[point_index:])

            shared_params_i = dict()
            shared_params_j = dict()     

            if self.args.shared_params:
                # partial parameter sharing
                parent_params_i = parents[i].get_ind_params()
                parent_params_j = parents[j].get_ind_params()

                # two parent
                if point_index == 0:
                    shared_params_i = copy.deepcopy(parent_params_i)
                    shared_params_j = copy.deepcopy(parent_params_j)
                # middle point
                elif point_index == 5:
                    shared_params_i[0] = parent_params_i[0][:]
                    shared_params_i[1] = parent_params_j[1][:]

                    shared_params_j[0] = parent_params_j[0][:]
                    shared_params_j[1] = parent_params_i[1][:]
                # one parent
                elif 0 < point_index < 5:
                    shared_params_i[0] = parent_params_i[0][:]
                    shared_params_j[0] = parent_params_j[0][:]
                else:
                    shared_params_i[1] = parent_params_i[1][:]
                    shared_params_j[1] = parent_params_j[1][:]

            # create offspring individuals
            offspring_i = Individual(offspring_gene_i, 
                                     parents[i].get_param_genes(), shared_params_i)
            offspring_j = Individual(offspring_gene_j, 
                                     parents[j].get_param_genes(), shared_params_j)
            
            offsprings.append([offspring_i, offspring_j])
            
        return offsprings   

    def crossover_param(self, parents):
        p_size = len(parents)
        maximum = p_size * (p_size - 1) / 2
        if self.args.num_offsprings_param > maximum:
            raise RuntimeError("number of offsprings should not be more than " 
                               + maximum)
        # randomly choose crossover parent pairs
        parent_pairs = []
        while len(parent_pairs) < self.args.num_offsprings_param:
            indexes = sample(range(p_size), k=2)
            pair = (indexes[0], indexes[1])
            if indexes[0] > indexes[1]:
                pair = (indexes[1], indexes[0])
            if not pair in parent_pairs:
                parent_pairs.append(pair)

        offsprings = []
        params_size = len(parents[0].get_param_genes())
        for i, j in parent_pairs:
            parent_gene_i = parents[i].get_param_genes()
            parent_gene_j = parents[j].get_param_genes()
            # select a random crossover point
            point_index = parent_gene_j.index(sample(parent_gene_j, 1)[0])
            offspring_gene_i = parent_gene_j[:point_index]
            offspring_gene_i.extend(parent_gene_i[point_index:])
            offspring_gene_j = parent_gene_i[:point_index]
            offspring_gene_j.extend(parent_gene_j[point_index:])
            
            # create offspring individuals
            offspring_i = copy.deepcopy(parents[i])
            offspring_j = copy.deepcopy(parents[j])
            offspring_i.set_newparam(offspring_gene_i)
            offspring_j.set_newparam(offspring_gene_j)
            
            offsprings.append([offspring_i, offspring_j])
        return offsprings  

    def crossover_net_single(self, parents, i, j):
        parent_gene_i = parents[i].get_net_genes()
        parent_gene_j = parents[j].get_net_genes()
        # select a random crossover point
        point_index = parent_gene_j.index(sample(parent_gene_j, 1)[0]) # possible 0
        
        offspring_gene_i = parent_gene_i[:point_index]
        offspring_gene_i.extend(parent_gene_j[point_index:])
        offspring_gene_j = parent_gene_j[:point_index]
        offspring_gene_j.extend(parent_gene_i[point_index:])

        shared_params_i = dict()
        shared_params_j = dict()
        if self.args.shared_params:
            # partial parameter sharing
            parent_params_i = parents[i].get_ind_params()
            parent_params_j = parents[j].get_ind_params()

            # two parent 
            ## edge point
            if point_index == 0:
                shared_params_i = copy.deepcopy(parent_params_i)
                shared_params_j = copy.deepcopy(parent_params_j)
            ## middle point
            elif point_index == 5:
                shared_params_i[0] = parent_params_i[0][:]
                shared_params_i[1] = parent_params_j[1][:]

                shared_params_j[0] = parent_params_j[0][:]
                shared_params_j[1] = parent_params_i[1][:]
            # one parent
            elif 0 < point_index < 5:
                shared_params_i[0] = parent_params_i[0][:]
                shared_params_j[0] = parent_params_j[0][:]
            else:
                shared_params_i[1] = parent_params_i[1][:]
                shared_params_j[1] = parent_params_j[1][:]

        # create offspring individuals
        offspring_i = Individual(offspring_gene_i, 
                                    parents[i].get_param_genes(), shared_params_i)
        offspring_j = Individual(offspring_gene_j, 
                                    parents[j].get_param_genes(), shared_params_j)
        
        return [offspring_i, offspring_j]

    def crossover_param_single(self, parents, i, j):
        parent_gene_i = parents[i].get_param_genes()
        parent_gene_j = parents[j].get_param_genes()
        
        # select a random crossover point
        point_index = parent_gene_j.index(sample(parent_gene_j, 1)[0])
        offspring_gene_i = parent_gene_i[:point_index]
        offspring_gene_i.extend(parent_gene_j[point_index:])
        offspring_gene_j = parent_gene_j[:point_index]
        offspring_gene_j.extend(parent_gene_i[point_index:])

        shared_params_i = dict()
        shared_params_j = dict()
        if self.args.shared_params:
            parent_params_i = parents[i].get_ind_params()
            parent_params_j = parents[j].get_ind_params()
            shared_params_i = copy.deepcopy(parent_params_i)
            shared_params_j = copy.deepcopy(parent_params_j)

        # create offspring individuals
        offspring_i = Individual(parents[i].get_net_genes(), 
                                    offspring_gene_i, shared_params_i)
        offspring_j = Individual(parents[j].get_net_genes(), 
                                    offspring_gene_j, shared_params_j)
        
        return [offspring_i, offspring_j]

    def crossover(self, parents):
        p_size = len(parents)
        maximum = self.args.num_individuals / 2

        # randomly choose crossover parent pairs
        parent_pairs = []
        while len(parent_pairs) < maximum:
            indexes = sample(range(p_size), k=2)
            pair = (indexes[0], indexes[1])
            if indexes[0] > indexes[1]:
                pair = (indexes[1], indexes[0])
            if not pair in parent_pairs:
                parent_pairs.append(pair)

        offsprings = []
        for i, j in parent_pairs:
            if np.random.uniform(0, 1, 1) <= 0.5:
                offsprings.extend(self.crossover_net_single(parents, i, j))
            else:
                offsprings.extend(self.crossover_param_single(parents, i, j))

        return offsprings

    def mutation_net(self, offsprings):
        """perform mutation for all new offspring individuals"""
        for pair in offsprings:
            random_prob = np.random.uniform(0, 1, 1)
            if random_prob <= self.args.mutate_prob:
                index, gene = self.hybrid_search_space.get_one_net_gene()
                pair[0].mutation_net_gene(index, gene, 'struct')
            random_prob = np.random.uniform(0, 1, 1)
            if random_prob <= self.args.mutate_prob:
                index, gene = self.hybrid_search_space.get_one_net_gene()
                pair[1].mutation_net_gene(index, gene, 'struct')
                
    def mutation_param(self, offsprings):
        """perform mutation for all new offspring individuals"""
        for pair in offsprings:
            random_prob = np.random.uniform(0, 1, 1)
            if random_prob <= self.args.mutate_prob:
                index, gene = self.hybrid_search_space.get_one_param_gene()
                pair[0].mutation_net_gene(index, gene, 'param')
            random_prob = np.random.uniform(0, 1, 1)
            if random_prob <= self.args.mutate_prob:
                index, gene = self.hybrid_search_space.get_one_param_gene()
                pair[1].mutation_net_gene(index, gene, 'param')
    
    def mutation(self, offsprings):
        for offspring in offsprings:
            random_prob = np.random.uniform(0, 1, 1)
            if random_prob <= self.args.mutate_prob:
                index, gene = self.hybrid_search_space.get_one_net_gene()
                offspring.mutation_net_gene(index, gene, 'struct')
                index, gene = self.hybrid_search_space.get_one_param_gene()
                offspring.mutation_net_gene(index, gene, 'param')
                
    def find_least_fittest(self, individuals):
        fitness = 10000
        index =-1
        for elem_index, elem in enumerate(individuals):
            if fitness > elem.get_fitness():
                fitness = elem.get_fitness()
                index = elem_index
                
        return index
                           
    def cal_fitness_offspring(self, offsprings):
        survivors = []
        for pair in offsprings:
            offspring_1 = pair[0]
            offspring_2 = pair[1]
            offspring_1.cal_fitness(self.gnn_manager)
            offspring_2.cal_fitness(self.gnn_manager)
            if offspring_1.get_fitness() > offspring_2.get_fitness():
                survivors.append(offspring_1)
            else:
                survivors.append(offspring_2)
        
        return survivors
    
    def cal_fitness_offspring_param(self, offsprings):
        survivors = []
        for pair in offsprings:
            offspring_1 = pair[0].get_population()
            offspring_2 = pair[1].get_population()
            for i in range(self.args.num_individuals):
                offspring_1[i].cal_fitness(self.gnn_manager)
            for j in range(self.args.num_individuals):
                offspring_2[i].cal_fitness(self.gnn_manager)
            pair[0].cal_superfitness()
            pair[1].cal_superfitness()
            
            if pair[0].get_fitness() > pair[1].get_fitness():
                survivors.append(pair[0])
            else:
                survivors.append(pair[1])
        
        return survivors
    
    def update_struct(self, elem):
        for i in range(self.args.num_individuals):
            individual = self.struct_individuals[i]
            if self.compare_action(individual.get_net_genes(), elem.get_net_genes()):
                if individual.get_fitness() < elem.get_fitness():
                    self.struct_individuals[i] = elem
                    return False
        return True
            
    def update_population_struct(self, survivors):
        """update current population with new offsprings"""
        for elem in survivors:
            if self.update_struct(elem):
                out_index = self.find_least_fittest(self.struct_individuals)
                self.struct_individuals[out_index] = elem

    def environmental_selection_struct(self, offsprings):
        """select individuals for next generation"""
        # calculate fitness for offsprings
        all_offsprings = []
        for pair in offsprings:
            offspring_1 = pair[0]
            offspring_2 = pair[1]
            offspring_1.cal_fitness(self.gnn_manager)
            offspring_2.cal_fitness(self.gnn_manager)
            all_offsprings.extend([offspring_1, offspring_2])

        update_population = all_offsprings + self.struct_individuals

        # binary tournament selection
        survivors = []
        for i in range(self.args.num_individuals):
            index_1 = np.random.randint(0, len(update_population), 1)[0]
            index_2 = np.random.randint(0, len(update_population), 1)[0]
            if update_population[index_1].get_fitness() > update_population[index_2].get_fitness():
                survivors.append(update_population[index_1])
            else:
                survivors.append(update_population[index_2])
        
        self.struct_individuals = survivors

    def compare_action(self, a1, a2):
        for i in range(len(a1)):
            if a1[i] != a2[i]:
                return False
        return True


    def update_population_param(self, survivors):
        """update current population with new offsprings"""
        for elem in survivors:
            out_index = self.find_least_fittest(self.super_population)
            self.super_population[out_index] = elem
        params_individuals = []
        for super_elem in self.super_population:
            params_individuals.append(super_elem.get_param_genes())
        self.params_individuals = params_individuals

    
    def print_models(self, iter):
        val_accs, test_accs = [], []
        num_params, times = [], []
        print('===begin, current population ({} in {} generations)===='.format(
                                    (iter+1), self.args.num_generations))
        
        best_individual = self.struct_individuals[0]
        for elem_index, elem in enumerate(self.struct_individuals):
            if best_individual.get_fitness() < elem.get_fitness():
                best_individual = elem
            val_accs.append(elem.get_fitness())
            test_accs.append(elem.get_test_acc())
            num_params.append(elem.num_params)
            times.append(elem.times)
            print('struct space: {}, param space: {}, validate_acc={}, test_acc={}'.format(
                                elem.get_net_genes(), 
                                elem.get_param_genes(),
                                elem.get_fitness(),
                                elem.get_test_acc()))
        print('------the best model-------')
        print('struct space: {}, param space: {}, validate_acc={}, test_acc={}'.format(
                           best_individual.get_net_genes(), 
                           best_individual.get_param_genes(),
                           best_individual.get_fitness(),
                           best_individual.get_test_acc()))    
           
        print('====end====\n')
        
        return best_individual, val_accs, test_accs, num_params, times
    
    def print_results(self, total_val_accs, total_test_accs, total_num_params, total_times):
        # top 5 validataion model's test accuracy mean and std
        top5 = np.argsort(total_val_accs)[-5:]
        print('\ntop 5 validation model\'s test accuracy mean and std: %.2f %.2f' % (np.mean(np.array(total_test_accs)[top5])*100, np.std(np.array(total_test_accs)[top5])*100))
        print('FLOPs and Params (M) and times mean (ms): ', np.mean(np.array(total_num_params)[top5][:,0])*10**(-6), np.mean(np.array(total_num_params)[top5][:,1])*10**(-6), np.mean(np.array(total_times)[top5][:,0])*1000, np.mean(np.array(total_times)[top5][:,1])*1000)

        # best test accuracy among top 5 validation models
        print(np.array(total_test_accs)[top5])
        argmax = np.argmax(np.array(total_test_accs)[top5])
        max_test_index = top5[argmax]
        print('best test accuracy among top 5 validation models : val %.2f test %.2f (%d)' % (total_val_accs[max_test_index]*100, total_test_accs[max_test_index]*100, max_test_index))
        print('FLOPs and Params (M) and times (ms): ', total_num_params[max_test_index][0]*10**(-6), total_num_params[max_test_index][1]*10**(-6), total_times[max_test_index][0]*1000, total_times[max_test_index][1]*1000)

                    
    def evolve_net(self):
        # initialize population
        self.init_population()
        # calculate fitness for population
        self.cal_fitness()
        
        actions = []
        params = []
        val_accs, test_accs = [], []
        total_val_accs, total_test_accs = [], []
        total_num_params, total_times = [], []
        
        for j in range(self.args.num_generations):
            start_time = time.time()
            # GNN hyper parameter evolution
            print('===================GNN hyper parameter evolution====================')
            initl_param_individual = copy.deepcopy(self.struct_individuals)
            self.init_param_population(initl_param_individual)

            for i in range(self.args.num_generations_param):
                param_parents = self.parent_selection_param()            
                param_offsprings = self.crossover_param(param_parents)
                self.mutation_param(param_offsprings)
                param_survivors = self.cal_fitness_offspring_param(param_offsprings)
                self.update_population_param(param_survivors) # update the population      
            
            # update the structure population with the best hyper-parameter
            print('##################update structure population##################')
            out_index = self.find_least_fittest(self.super_population)
            self.struct_individuals = self.super_population[out_index].get_population()
            
            # GNN structure evolution
            print('--------------------GNN structure evolution-------------------------')
            struct_parents = self.parent_selection() # parents selection
            struct_offsprings = self.crossover_net(struct_parents) # crossover to produce offsprings
            self.mutation_net(struct_offsprings) # perform mutation
            struct_survivors = self.cal_fitness_offspring(struct_offsprings) # calculate fitness for offsprings
            self.update_population_struct(struct_survivors) # update the population 
            
            best_individual, vals, tests, trainable_params, times = self.print_models(j)
            
            # print best individual
            actions.append(best_individual.get_net_genes())
            params.append(best_individual.get_param_genes())
            val_accs.append(best_individual.get_fitness())
            test_accs.append(best_individual.get_test_acc())

            # print everything
            total_val_accs.extend(vals)
            total_test_accs.extend(tests)
            total_num_params.extend(trainable_params)
            total_times.extend(times)
        
            print(actions)           
            print(params)           
            print(val_accs)           
            print(test_accs)           
            print('generation time: ', time.time() - start_time)   

        print('total_val_accs: ', total_val_accs, len(total_val_accs))
        print('total_test_accs: ', total_test_accs, len(total_test_accs))
        print('total_num_params: ', total_num_params, len(total_num_params))
        print('total_times: ', total_times, len(total_times))
        
        self.print_results(total_val_accs, total_test_accs, total_num_params, total_times)

    def evolve_net_combined(self):
        actions = []
        params = []
        val_accs, test_accs = [], []
        total_val_accs, total_test_accs = [], []
        total_num_params, total_times = [], []
        
        # initialize population
        self.init_combined_population()
        # calculate fitness for population
        self.cal_fitness()
        
        for j in range(self.args.num_generations):
            start_time = time.time()
            # GNN hyper parameter evolution
            print('===================GNN combined evolution====================')
            parents = self.parent_selection()
            offsprings = self.crossover(parents)
            self.mutation(offsprings)
            survivors = self.cal_fitness(offsprings)
            self.update_population_struct(survivors)

            best_individual, vals, tests, trainable_params, times = self.print_models(j)
            
            # print best individual
            actions.append(best_individual.get_net_genes())
            params.append(best_individual.get_param_genes())
            val_accs.append(best_individual.get_fitness())
            test_accs.append(best_individual.get_test_acc())

            # print everything
            total_val_accs.extend(vals)
            total_test_accs.extend(tests)
            total_num_params.extend(trainable_params)
            total_times.extend(times)
        
            print(actions)           
            print(params)           
            print(val_accs)           
            print(test_accs)           
            print('generation time: ', time.time() - start_time)   

        print('total_val_accs: ', total_val_accs, len(total_val_accs))
        print('total_test_accs: ', total_test_accs, len(total_test_accs))
        print('total_num_params: ', total_num_params, len(total_num_params))
        print('total_times: ', total_times, len(total_times))
        
        self.print_results(total_val_accs, total_test_accs, total_num_params, total_times)
