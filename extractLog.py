import re
import numpy as np

log_file = "EfficiencyGNN/EGNAS.log" 
total_val_accs = []
total_test_accs = []

with open(log_file, 'r') as file:
    content = file.read()
    match = re.findall(r"total_val_accs:\s*\[([\d.,\s]+)\]", content)
    total_val_accs = [float(x.strip()) for x in match[0].split(",")]
    match = re.findall(r"total_test_accs:\s*\[([\d.,\s]+)\]", content)
    total_test_accs = [float(x.strip()) for x in match[0].split(",")]


def print_results(total_val_accs, total_test_accs):
    # top 5 test model's test accuracy mean and std
    print(len(total_val_accs), len(total_test_accs))
    top5 = np.argsort(total_test_accs)[-5:]
    print('top 5 test model\'s test accuracy mean and std: %.2f %.2f' % (np.mean(np.array(total_test_accs)[top5])*100, np.std(np.array(total_test_accs)[top5])*100))

    # best test accuracy among top 5 validation models
    print(np.array(total_test_accs)[top5])
    argmax = np.argmax(np.array(total_test_accs)[top5])
    max_test_index = top5[argmax]
    print('best test accuracy: val %.2f test %.2f (%d)' % (total_val_accs[max_test_index]*100, total_test_accs[max_test_index]*100, max_test_index))

print_results(total_val_accs, total_test_accs)