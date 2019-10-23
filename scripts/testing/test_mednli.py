"""
Test the ESIM model on some preprocessed dataset.
"""
# Soumya Sharma, 2019.

import os
import sys

from matplotlib.rcsetup import validate_nseq_float

sys.path.insert(0, "../../")
import time
import pickle
import argparse
import torch
import json
import torch.nn as nn

from torch.utils.data import DataLoader
from esim.data import MedNLIDataset, MedNLIEmbedding
from esim.model import ESIM
from esim.utils import correct_predictions, get_tensor_of_tensor


def _test(model, dataloader, embeddings, batch_size, testing):
    """
    Test the accuracy of a model on some labelled test dataset.

    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.

    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    running_accuracy = 0.0
    out_classes = []
    labels = []
    all_premises = []
    all_hypotheses = []
    similarity_matrices = []
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        count = 0
        for batch in dataloader:
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            premises = batch["premises"]  # .to(device)
            premises_lengths = torch.LongTensor(batch["premises_lengths"]).to(device)
            hypotheses = batch["hypotheses"]  # .to(device)
            hypotheses_lengths = torch.LongTensor(batch["hypotheses_lengths"]).to(device)
            label = torch.LongTensor(batch["labels"]).to(device)
            premise_polarities = get_tensor_of_tensor(batch["premise_polarities"], batch['max_premise_length']).to(
                device)
            hypothesis_polarities = get_tensor_of_tensor(batch["hypothesis_polarities"],
                                                         batch['max_hypothesis_length']).to(device)

            batch_embeddings = embeddings.get_batch(count)

            _, probs, similarity_matrix = model(premises,
                             premises_lengths,
                             hypotheses,
                             hypotheses_lengths,
                             premise_polarities,
                             hypothesis_polarities,
                             batch_embeddings,
                             batch['max_premise_length'],
                             batch['max_hypothesis_length'])

            accuracy, out_class = correct_predictions(probs, label)
            batch_time += time.time() - batch_start
            running_accuracy += accuracy
            out_classes.append(out_class)
            labels.append(label)
            all_premises.append(premises)
            all_hypotheses.append(hypotheses)
            similarity_matrices.append(similarity_matrix)
            count = count + batch_size
            if testing: break

    batch_time /= len(dataloader)
    total_time = time.time() - time_start
    running_accuracy /= (len(dataloader))

    return batch_time, total_time, running_accuracy, out_classes, labels, similarity_matrices, probs, all_premises, all_hypotheses

def validate(model, dataloader, embeddings, criterion, batch_size, testing):
    """
    Compute the loss and accuracy of a model on some validation dataset.

    Args:
        model: A torch module for which the loss and accuracy must be
            computed.
        dataloader: A DataLoader object to iterate over the validation data.
        criterion: A loss criterion to use for computing the loss.
        epoch: The number of the epoch for which validation is performed.
        device: The device on which the model is located.

    Returns:
        epoch_time: The total time to compute the loss and accuracy on the
            entire validation set.
        epoch_loss: The loss computed on the entire validation set.
        epoch_accuracy: The accuracy computed on the entire validation set.
    """
    # Switch to evaluate mode.
    model.eval()
    device = model.device
    epoch_start = time.time()
    running_loss = 0.0
    running_accuracy = 0.0
    # Deactivate autograd for evaluation.
    with torch.no_grad():
        count = 0
        for batch in dataloader:
            # Move input and output data to the GPU if one is used.
            premises = batch["premises"]#.to(device)
            premises_lengths = torch.LongTensor(batch["premises_lengths"]).to(device)
            hypotheses = batch["hypotheses"]#.to(device)
            hypotheses_lengths = torch.LongTensor(batch["hypotheses_lengths"]).to(device)
            labels = torch.LongTensor(batch["labels"]).to(device)
            premise_polarities = get_tensor_of_tensor(batch["premise_polarities"], batch['max_premise_length']).to(
                device)
            hypothesis_polarities = get_tensor_of_tensor(batch["hypothesis_polarities"],
                                                         batch['max_hypothesis_length']).to(device)
            batch_embeddings = embeddings.get_batch(count)

            logits, probs, _ = model(premises,
                                  premises_lengths,
                                  hypotheses,
                                  hypotheses_lengths,
                                  premise_polarities,
                                  hypothesis_polarities,
                                  batch_embeddings,
                                  batch['max_premise_length'],
                                  batch['max_hypothesis_length'])
            
            loss = criterion(logits, labels)
            running_loss += loss.item()
            accuracy, out_class = correct_predictions(probs, labels)
            
            running_accuracy += accuracy
            count = count + batch_size
            if testing: break

    epoch_time = time.time() - epoch_start
    epoch_loss = running_loss / len(dataloader)
    epoch_accuracy = running_accuracy / (len(dataloader))
    return epoch_time, epoch_loss, epoch_accuracy

def main(test_file,
         test_embeddings_file,
	     pretrained_file,
         target_file,
         dataset,
         embedding_dim,
	     distmult=False,
         distmultPath=None,
         distmultEmbeddingDim=None,
         hidden_size=300,
         num_classes=3,
         epochs=64,
         batch_size=32,
         lr=0.0004,
         patience=5,
         max_grad_norm=10.0,
         testing=True,
         multipassiterations=1,
         lstm=False,
         weightedattention=False,
         sentiment=False
         ):
    """
    Test the ESIM model with pretrained weights on some dataset.

    Args:
        test_file: The path to a file containing preprocessed NLI data.
        pretrained_file: The path to a checkpoint produced by the
            'train_model' script.
        vocab_size: The number of words in the vocabulary of the model
            being tested.
        embedding_dim: The size of the embeddings in the model.
        hidden_size: The size of the hidden layers in the model. Must match
            the size used during training. Defaults to 300.
        num_classes: The number of classes in the output of the model. Must
            match the value used during training. Defaults to 3.
        batch_size: The size of the batches used for testing. Defaults to 32.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")

    checkpoint = torch.load(pretrained_file, map_location="cpu")

    # Retrieving model parameters from checkpoint.
    #vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
    #embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
    #hidden_size = checkpoint["model"]["_projection.0.weight"].size(0)
    #num_classes = checkpoint["model"]["_classification.4.weight"].size(0)

    print("\t* Loading validation data...")
    with open(test_file, "rb") as pkl:
        test_data = MedNLIDataset(pickle.load(pkl), batch_size=batch_size)

    # valid_loader = valid_data.batch(batch_size)#DataLoader(valid_data, shuffle=False, batch_size=batch_size)
    print("\t* Loading validation embeddings...")
    with open(test_embeddings_file, "rb") as pkl:
        bleh = pickle.load(pkl)
        test_embeddings = MedNLIEmbedding(bleh, batch_size=batch_size)

    print("\t* Building model...")
    model = ESIM(
                 embedding_dim,
                 hidden_size,
                 dataset=dataset,
                 distmult=distmult,
                 distmultPath=distmultPath,
                 distmultEmbeddingDim=distmultEmbeddingDim,
                 num_classes=num_classes,
                 multipassiterations=multipassiterations,
                 lstm=lstm,
                 weightedattention=weightedattention,
                 testing=testing,
                 sentiment=sentiment,
                 device=device).to(device)
    print("\t* Model initialized...")
    for idx in range(multipassiterations):
        model.initialize_layers(idx)
    print(model)
    print("\t* Model build...")
    model.load_state_dict(checkpoint["model"])

    print(20 * "=",
          " Testing ESIM model on device: {} ".format(device),
          20 * "=")
    batch_time, total_time, accuracy, out_classes, labels, similarity_matrices, probs, all_premises, all_hypotheses = _test(model, test_data, test_embeddings, batch_size, testing)

    with open(target_file, 'w') as f:
        data_points = len(test_data)
        f.write(
            "premises_raw" + "\t" + "premises_umls" + "\t" + "hypotheses_raw" + "\t" + "hypotheses_umls" + "\t" + "output" + "\t" + "gold_label" + "\n")
        count = 0
        for idx1, item1 in enumerate(all_premises):
            for idx2, item2 in enumerate(all_premises[idx1]):
                if out_classes[idx1][idx2] == 0:
                    opt1 = "entailment"
                elif out_classes[idx1][idx2] == 2:
                    opt1 = "contradiction"
                elif out_classes[idx1][idx2] == 1:
                    opt1 = "neutral"
                if labels[idx1][idx2] == 0:
                    opt2 = "entailment"
                elif labels[idx1][idx2] == 2:
                    opt2 = "contradiction"
                elif labels[idx1][idx2] == 1:
                    opt2 = "neutral"
                f.write(" ".join(all_premises[idx1][idx2][0]) + "\t" + " ".join(
                    all_premises[idx1][idx2][1]) + "\t" + " ".join(all_hypotheses[idx1][idx2][0]) + "\t" + " ".join(
                    all_hypotheses[idx1][idx2][1]) + "\t" + opt1 + "\t" + opt2 + "\n")
                # Uncomment to save similarity matrices
                # f.write(" ".join(all_premises[idx1][idx2][0]) + "\t" + " ".join(all_hypotheses[idx1][idx2][0]) + "\t" + opt1 + "\t" + opt2 + "\n")
                # torch.save(similarity_matrices[idx1][idx2],
                #            "/".join(target_file.split("/")[:-1]) + "/" + str(count) + ".tensor")
                count += 1

    print("-> Average batch processing time: {:.4f}s, total test time:\
 {:.4f}s, accuracy: {:.4f}%".format(batch_time, total_time, (accuracy*100)))
    return out_classes, labels, similarity_matrices, probs

if __name__ == "__main__":

    default_config = "../../config/testing/mednli_testing.json"

    parser = argparse.ArgumentParser(description="Test the ESIM model on\
 some dataset")
    parser.add_argument("--config",
                        default=default_config,
                        help="Path to a json configuration file")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Path to a checkpoint file to resume training")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), 'r') as config_file:
        config = json.load(config_file)

    a,b,c,d=main(
         os.path.normpath(os.path.join(script_dir, config["test_data"])),
         os.path.normpath(os.path.join(script_dir, config["test_embeddings"])),
         config["checkpoint"],
	     os.path.normpath(os.path.join(script_dir, config["target_file"])),
         config["dataset"],
         config["embedding_dim"],
         config["distmult"],
         config["distmultPath"],
         config["distmultEmbeddingDim"],
         config["hidden_size"],
         config["num_classes"],
         config["epochs"],
         config["batch_size"],
         config["lr"],
         config["patience"],
         config["max_gradient_norm"],
         config["testing"],
         config["multipassiterations"],
         config["lstm"],
         config["weightedattention"],
        config["sentiment"]
    )

