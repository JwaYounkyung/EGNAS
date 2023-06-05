import re
import numpy as np

log_file = "results/GraphNAS.log"
total_val_accs = []
total_test_accs = []

# train_time:0.012415885925292969, inference_time:0.0037789344787597656, params:35383
with open(log_file, 'r') as file:
    content = file.read()
    matches = re.findall(r"val_score:([\d.]+), test_score:([\d.]+)", content)
    total_val_accs = [float(match[0]) for match in matches]
    total_test_accs = [float(match[1]) for match in matches]
    matches = re.findall(r"train_time:([\d.]+), inference_time:([\d.]+), params:([\d.]+)", content)
    total_num_params = [int(match[2]) for match in matches]
    total_times = [[float(match[0]), float(match[1])] for match in matches]


def print_results(total_val_accs, total_test_accs, total_num_params, total_times):
    # top 5 validataion model's test accuracy mean and std
    top5 = np.argsort(total_val_accs)[-5:]
    print('\ntotal number of architecture evaluations: %d' % len(total_val_accs))
    print('top 5 validation model\'s test accuracy mean and std: %.2f %.2f' % (np.mean(np.array(total_test_accs)[top5])*100, np.std(np.array(total_test_accs)[top5])*100))
    print('Params (M) and times mean (ms): ', np.mean(np.array(total_num_params)[top5])*10**(-6), np.mean(np.array(total_times)[top5][:,0])*1000, np.mean(np.array(total_times)[top5][:,1])*1000)

    # best test accuracy among top 5 validation models
    print(np.array(total_test_accs)[top5])
    argmax = np.argmax(np.array(total_test_accs)[top5])
    max_test_index = top5[argmax]
    print('best test accuracy among top 5 validation models : val %.2f test %.2f (%d)' % (total_val_accs[max_test_index]*100, total_test_accs[max_test_index]*100, max_test_index))
    print('Params (M) and times (ms): ', total_num_params[max_test_index]*10**(-6), total_times[max_test_index][0]*1000, total_times[max_test_index][1]*1000)

    # top 5 test model's test accuracy mean and std
    top5 = np.argsort(total_test_accs)[-5:]
    print('\ntop 5 test model\'s test accuracy mean and std: %.2f %.2f' % (np.mean(np.array(total_test_accs)[top5])*100, np.std(np.array(total_test_accs)[top5])*100))
    print('Params (M) and times mean (ms): ', np.mean(np.array(total_num_params)[top5])*10**(-6), np.mean(np.array(total_times)[top5][:,0])*1000, np.mean(np.array(total_times)[top5][:,1])*1000)

    # best test accuracy among top 5 validation models
    print(np.array(total_test_accs)[top5])
    argmax = np.argmax(np.array(total_test_accs)[top5])
    max_test_index = top5[argmax]
    print('best test accuracy: val %.2f test %.2f (%d)' % (total_val_accs[max_test_index]*100, total_test_accs[max_test_index]*100, max_test_index))
    print('Params (M) and times (ms): ', total_num_params[max_test_index]*10**(-6), total_times[max_test_index][0]*1000, total_times[max_test_index][1]*1000)

print_results(total_val_accs, total_test_accs, total_num_params, total_times)