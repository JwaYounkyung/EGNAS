import re
import numpy as np

import matplotlib.pyplot as plt
import pickle
import numpy as np

log_file = "results/EGNAS_Cora_noearly_83.70(best).log"
log_file2 = "results/EGNAS_Cora_nohalf_83.00.log"
file_name = log_file.split('/')[-1].split('(')[0]
print(file_name)

def extract_graph(log_file):
    with open(log_file, 'r') as file:
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

    mean_val_accs = np.mean(total[:,:,0], axis=1)
    mean_test_accs = np.mean(total[:,:,1], axis=1)

    return mean_val_accs, mean_test_accs, best_total

def result_plot(val, test, file_name, plot_save=False):
    plt.rcParams.update({'font.size': 18})
    plt.clf() 
    plt.figure(figsize=(15, 6))

    epochs = range(len(val))

    plt.plot(epochs, val, label='EGNAS', marker='o')
    plt.plot(epochs, test, label='EGNAS w/o half epochs', marker='o')
    plt.xlabel('Generations')
    plt.ylabel('Accuracy')
    
    # plt.title('Cora')
    plt.legend()
    if plot_save:
        plt.savefig('results/image/{}.png'.format(file_name))
    plt.show()


mean_val1, mean_test1, best_total1 = extract_graph(log_file) # no early
mean_val2, mean_test2, best_total2 = extract_graph(log_file2) # no half

# result_plot(best_total[:,0], best_total[:,1], file_name, plot_save=True)
result_plot(best_total1[:,1], best_total2[:,1], file_name, plot_save=True)
result_plot(mean_test1, mean_test2, file_name+'_avg', plot_save=True)
