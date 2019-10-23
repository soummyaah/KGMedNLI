import time
import subprocess
import itertools
from multiprocessing import Process
import os
import threading
import json
import copy

def get_config_path_from_options(config, options):
    curr_config = copy.deepcopy(config)
    curr_config["hidden_size"] = options[0]
    curr_config["dropout"] = options[1]
    curr_config["multipassiterations"] = options[2]
    curr_config["lstm"] = options[3]
    curr_config["weightedattention"] = options[4]
    epoch = options[5]
    curr_config_dir = curr_config["target_file"]
    outputDir = curr_config_dir + \
                    "testing" + "/" + str(epoch) + "/"
    curr_config["target_file"] = outputDir + "error_out"
    curr_config["checkpoint"] = curr_config_dir + \
                                    "esim_" + str(epoch) + ".pth.tar"

    os.system("mkdir -p " + outputDir)
    exists = os.path.isfile(curr_config["checkpoint"])
    filePath = outputDir + "config.json"
    outFile = open(filePath, "w")
    outFile.write(json.dumps(curr_config))
    outFile.close()
    if exists:
        return outputDir
    else:
        return None

def create_new_process(config_path, i):
    if i % 2 == 0:
        os.system("export CUDA_VISIBLE_DEVICES=0")
    else:
        os.system("export CUDA_VISIBLE_DEVICES=1")
    os.system("mkdir -p " + config["target_file"])
    cmd = "CUDA_VISIBLE_DEVICES=" + str(i%2) + " python -u /home/soumyasharma/ESIM-github/ESIM/scripts/testing/test_mednli.py --config " + config_path + "config.json | tee " + config_path + "console.output"
    print(cmd)
    child = subprocess.Popen(cmd, shell=True)
    return child

def wait_timeout(config, processes, task_list, seconds):
    curr = 0
    while curr != len(task_list):
        for idx, proc in enumerate(processes):
            result = proc.poll()
            if result is None:
                continue
            else:
                if curr < len(task_list):
                    child_process = create_new_process(get_config_path_from_options(config, task_list[curr]), idx)
                    processes[idx] = child_process
                    curr += 1
        time.sleep(seconds)
    for proc in processes:
        result = proc.poll()
        while result is None:
            time.sleep(seconds)
            result = proc.poll()

    return

if __name__ == "__main__":

    embedding_dim_options = [500]
    dropout_options = [0.5]
    multipassiterations_options = [1]
    lstm_options = [True]  # , False]
    weightedattention_options = [False] #[True, False]
    epoch = [4,5,6,7,8,9,10,11,12]

    s = [embedding_dim_options, dropout_options, multipassiterations_options, lstm_options, weightedattention_options, epoch]
    options_list = list(itertools.product(*s))
    print("Hello!")
    config = {
        "test_data": "../../data/preprocessed/MedNLI/test_data.pkl",
        "test_embeddings": "../../data/preprocessed/MedNLI/test_elmo.pkl",

        "checkpoint": "../../data/checkpoints/MedNLI-Elmo-final/bioelmo+esim/distmult0multi1hidden_size500dropout0.5lr0.0001grad10lstmTweightedAttF/",
        "target_file": "../../data/checkpoints/MedNLI-Elmo-final/bioelmo+esim/distmult0multi1hidden_size500dropout0.5lr0.0001grad10lstmTweightedAttF/",

        "dataset": "elmo",
        "distmult": 0,
        "distmultPath": "../../data/dataset/glove_style_embedding.vec",
        "distmultEmbeddingDim": 100,

        "embedding_dim": 1024,
        "hidden_size": 500,
        "dropout": 0.5,
        "num_classes": 3,
        "epochs": 64,
        "batch_size": 32,
        "lr": 0.0001,
        "patience": 5,
        "max_gradient_norm": 10.0,
        "testing": False,
        "multipassiterations": 2,
        "lstm": True,
        "weightedattention": False,
        "sentiment": False
    }
    print("Hello!")

    no_processes = min(6, len(options_list))
    processes = []
    for idx in range(no_processes):
        config_path = get_config_path_from_options(config, options_list[idx])
        if config_path is not None:
            processes.append(create_new_process(config_path, idx))

    wait_timeout(config, processes, options_list[no_processes:], 200)
