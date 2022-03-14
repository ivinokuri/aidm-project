import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import roc_auc_score, average_precision_score

from DPlan import DPlan
from DPlanEnv import DPlanEnv


def prepare_data():
    """
    Prepare data to be in the same format
    """
    train_data = "./data/training-set.csv"
    data = pd.read_csv(train_data)
    sets = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance",
            "Hypothyroid", "Subnormal", "Downstairs", "Upstairs", "Cottonwood", "Douglas-fir", "CupPick"]
    for set in sets:
        result = []
        for row in data.values:
            if row[-2] == 'Normal' or row[-2] == set:
                result.append(row)
        resDF = pd.DataFrame(result)
        del resDF[0]
        del resDF[2]
        del resDF[3]
        del resDF[4]
        del resDF[43]
        resDF.to_csv("./data/train/" + set + '.csv', index=False, header=False)
    test_data = './data/original-testing-set.csv'
    data = pd.read_csv(test_data)
    del data['id']
    del data['proto']
    del data['service']
    del data['state']
    del data['attack_cat']
    data.to_csv("./data/testing-set.csv", index=False, header=False)


def main(parsed_args):
    """
    Running DPlan training and test
    :param parsed_args: runtime arguments
    """
    print("Init process")
    data_path = "./"
    data_folders = ["data"]
    data_subsets = {"data": ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance",
            "Hypothyroid", "Subnormal", "Downstairs", "Upstairs", "Cottonwood", "Douglas-fir", "CupPick"]}
    testdata_subset = "testing-set.csv"

    runs = 10
    model_path = "./model"
    result_path = "./results"
    result_file = "results.csv"
    is_train = True
    is_test = True

    if not os.path.exists(model_path):
        os.mkdir(model_path)
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    for data_f in data_folders:
        subsets = data_subsets[data_f]

        for subset in subsets:
            data_name = subset
            unknown_dataname = "train/" + subset + ".csv"
            undata_path = os.path.join(data_path, data_f, unknown_dataname)
            table = pd.read_csv(undata_path)
            undataset = table.values

            testdata_path = os.path.join(data_path, data_f, subset + "-" + testdata_subset)
            test_table = pd.read_csv(testdata_path)
            test_dataset = test_table.values

            rocs = []
            prs = []
            for i in range(runs):
                print("Dataset: {}".format(subset))
                print("Run: {}".format(i))

                weights_file = os.path.join(model_path, "{}_{}_{}_weights.h4f".format(subset, i, data_name))
                tf.compat.v1.reset_default_graph()
                env = DPlanEnv(data=undataset)
                model = DPlan(parsed_args=parsed_args, env=env, features=len(undataset[0])-1)

                if is_train:
                    model.fit(weights_file=weights_file)

                if is_test:
                    test_X, test_y = test_dataset[:, :-1], test_dataset[:, -1]
                    model.load_weights(weights_file)
                    pred_y = model.predict(test_X)

                    roc = roc_auc_score(test_y, pred_y)
                    pr = average_precision_score(test_y, pred_y)
                    print("{} Run {}: AUC-ROC: {:.4f}, AUC-PR: {:.4f}".format(
                        subset,
                        i,
                        roc,
                        pr))

                    rocs.append(roc)
                    prs.append(pr)

            if is_test:
                writeResults(subset, rocs, prs, os.path.join(result_path, result_file))


def writeResults(name, rocs, prs, file_path):
    """
    Appending results to the results csv
    :param name:
    :param rocs:
    :param prs:
    :param file_path:
    :return:
    """
    roc_mean = np.mean(rocs)
    roc_std = np.std(rocs)
    pr_mean = np.mean(prs)
    pr_std = np.std(prs)

    header = True
    if not os.path.exists(file_path):
        header = False

    with open(file_path, 'a') as f:
        if not header:
            f.write("{}, {}, {}\n".format("Name", "AUC-ROC(mean/std)",
                                                  "AUC-PR(mean/std)"))

        f.write("{}, {:.3f}/{:.4f}, {:.3f}/{:.4f}\n".format(name, roc_mean, roc_std,
                                                pr_mean, pr_std))


if __name__ == '__main__':
    parsed_args = {}
    parsed_args['memory_size'] = 100000
    parsed_args['hidden_size'] = 20
    parsed_args['batch_size'] = 32
    parsed_args['max_epsilon'] = 1
    parsed_args['min_epsilon'] = 0.1
    parsed_args['epsilon_course'] = 10000
    parsed_args['epochs'] = 30000
    parsed_args['steps'] = 2000
    parsed_args['lr'] = 0.00025

    prepare_data()
    main(parsed_args)
