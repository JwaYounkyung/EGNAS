import re
import numpy as np

import matplotlib.pyplot as plt
import pickle
import numpy as np

log_files = ['EGNAS_Cora_noearly_82.40.log','EGNAS_Cora_noearly_82.42.log','EGNAS_Cora_noearly_82.48.log','EGNAS_Cora_noearly_82.50.log','EGNAS_Cora_noearly_82.68.log','EGNAS_Cora_noearly_82.80.log','EGNAS_Cora_noearly_83.20_2.log','EGNAS_Cora_noearly_83.20.log','EGNAS_Cora_noearly_83.28.log','EGNAS_Cora_noearly_83.70(best).log']
log_files_nohalf = ['EGNAS_Cora_nohalf_81.32.log','EGNAS_Cora_nohalf_81.80.log','EGNAS_Cora_nohalf_81.88.log','EGNAS_Cora_nohalf_82.10.log','EGNAS_Cora_nohalf_82.20.log','EGNAS_Cora_nohalf_82.50.log','EGNAS_Cora_nohalf_82.70.log','EGNAS_Cora_nohalf_83.00(best).log']

def extract_graph(log_file):
    with open('results/'+ log_file, 'r') as file:
        content = file.read()
        match = re.findall(r"total_val_accs:\s*\[([\d.,\s]+)\]", content)
        total_val_accs = [float(x.strip()) for x in match[0].split(",")]
        match = re.findall(r"total_test_accs:\s*\[([\d.,\s]+)\]", content)
        total_test_accs = [float(x.strip()) for x in match[0].split(",")]
    
    # best validation accuracy among generations
    total = np.array([[total_val_accs[i], total_test_accs[i]] for i in range(len(total_val_accs))])
    total = np.array(total).reshape(50, 20, 2)

    # best val acc over 20 individuals
    arg_valbest = np.argmax(total[:,:,0], axis=1)
    best_total = np.zeros((50, 2))
    for i in range(50):
        best_total[i] = total[i][arg_valbest[i]]

    return best_total

def result_plot(means, stds, file_name='', plot_save=False):
    plt.rcParams.update({'font.size': 18})
    plt.clf() 
    plt.figure(figsize=(12, 6))
    plt.xlabel('Generations')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.ylim([0.77, 0.84])

    epochs = range(len(means[0]))
    # plt.xticks(range(0, len(means[0]), 5))
    # plt.xlim([-5, 55])  # x축 범위를 넓힘

    # Add error bars and confidence intervals for EGNAS
    plt.errorbar(epochs, means[0], yerr=stds[0], fmt='bo-', markersize=3, capsize=3)
    # plt.fill_between(epochs, means[0] - stds[0], means[0] + stds[0], color='b', alpha=0.1)

    # Add error bars and confidence intervals for "no Half Epochs"
    plt.errorbar(epochs, means[1], yerr=stds[1], fmt='ro-', markersize=3, capsize=3)
    # plt.fill_between(epochs, means[1] - stds[1], means[1] + stds[1], color='r', alpha=0.1)

    plt.plot(epochs, means[0], 'bo-', label='EGNAS', markersize=3)
    plt.fill_between(epochs, means[0] - stds[0], means[0] + stds[0], color='b', alpha=0.1)
    plt.plot(epochs, means[1], 'ro-', label='no Half Epochs', markersize=3)
    plt.fill_between(epochs, means[1] - stds[1], means[1] + stds[1], color='r', alpha=0.1)

    plt.legend(loc='lower right')
    if plot_save:
        plt.savefig('image/{}.png'.format(file_name))
    # plt.show()

def result_plot2(means, file_name='', plot_save=False):
    plt.rcParams.update({'font.size': 18})
    plt.clf() 
    plt.figure(figsize=(8, 6))
    plt.xlabel('Generations')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.ylim([0.77, 0.84])

    epochs = range(len(means[0]))
    # plt.xticks(range(0, len(means[0]), 5))
    plt.plot(epochs, means[0], 'bo-', label='test', markersize=3)
    plt.plot(epochs, means[1], 'ro-', label='val', markersize=3)

    plt.legend(loc='lower right')
    if plot_save:
        plt.savefig('image/{}.png'.format(file_name))
    # plt.show()

# extract mean, std
best_val = np.zeros((10, 50))
best_test = np.zeros((10, 50))
for i in range(10):
    best = extract_graph('EGNAS/'+log_files[i])
    best_val[i] = best[:,0]
    best_test[i] = best[:,1]

mean_test = np.mean(best_test, axis=0)
std_test = np.std(best_test, axis=0)
mean_val = np.mean(best_val, axis=0)

# extract mean, std nohalf
best_val_nohalf = np.zeros((8, 50))
best_test_nohalf = np.zeros((8, 50))
for i in range(8):
    best_nohalf = extract_graph('EGNAS_nohalf/'+log_files_nohalf[i])
    best_val_nohalf[i] = best_nohalf[:,0]
    best_test_nohalf[i] = best_nohalf[:,1]

mean_test_nohalf = np.mean(best_test_nohalf, axis=0)
std_test_nohalf = np.std(best_test_nohalf, axis=0)
mean_val_nohalf = np.mean(best_val_nohalf, axis=0)

result_plot([mean_test, mean_test_nohalf], [std_test, std_test_nohalf], 'nohalf', True)
# result_plot2([mean_test, mean_val], 'EGNAS_overfitting', True)
# result_plot2([mean_test_nohalf, mean_val_nohalf], 'nohalf_overfitting', True)