"""
Preprocess the MedNLI dataset and word embeddings to be used by the ESIM model.
"""
# Soumya Sharma, 2019.

import os
import sys
sys.path.insert(0, "../../")
import pickle
import argparse
import fnmatch
import json

from esim.data import Preprocessor

def preprocess_SNLI_data(inputdir,
                         targetdir,
                         lowercase=False,
                         ignore_punctuation=False,
                         num_words=None,
                         stopwords=[],
                         labeldict={}):
    """
    Preprocess the data from the MedNLI corpus so it can be used by the
    ESIM model.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        inputdir: The path to the directory containing the NLI corpus.
        embeddings_file: The path to the file containing the pretrained
            word vectors that must be used to build the embedding matrix.
        targetdir: The path to the directory where the preprocessed data
            must be saved.
        lowercase: Boolean value indicating whether to lowercase the premises
            and hypotheseses in the input data. Defautls to False.
        ignore_punctuation: Boolean value indicating whether to remove
            punctuation from the input data. Defaults to False.
        num_words: Integer value indicating the size of the vocabulary to use
            for the word embeddings. If set to None, all words are kept.
            Defaults to None.
        stopwords: A list of words that must be ignored when preprocessing
            the data. Defaults to an empty list.
            If set to None, eos tokens aren't used. Defaults to None.
    """
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Retrieve the train, dev and test data files from the dataset directory.
    train_file = ""
    dev_file = ""
    test_file = ""
    for file in os.listdir(inputdir):
        if fnmatch.fnmatch(file, "train*"):
            train_file = file
        elif fnmatch.fnmatch(file, "dev*"):
            dev_file = file
        elif fnmatch.fnmatch(file, "test*"):
            test_file = file

    # -------------------- Train data preprocessing -------------------- #
    preprocessor = Preprocessor(lowercase=lowercase,
                                ignore_punctuation=ignore_punctuation,
                                num_words=num_words,
                                stopwords=stopwords,
                                labeldict=labeldict)

    print(20*"=", " Preprocessing train set ", 20*"=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, train_file))

    print("\t* Saving result...")
    with open(os.path.join(targetdir, "train_data.pkl"), "wb") as pkl_file:
        pickle.dump(data, pkl_file)

    # -------------------- Tokenization preprocessing -------------------- #
    print(20 * "=", " Preprocessing tokenization", 20 * "=")
    print("\t* Creating tokenization and saving it...")
    preprocessor.create_tokenizations(data,
                                      os.path.join(targetdir, "train_elmo.pkl"),
                                      os.path.join(targetdir, "train_bert.pkl"))

    # -------------------- Validation data preprocessing -------------------- #
    print(20*"=", " Preprocessing dev set ", 20*"=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, dev_file))

    print("\t* Saving result...")
    with open(os.path.join(targetdir, "dev_data.pkl"), "wb") as pkl_file:
        pickle.dump(data, pkl_file)


    # -------------------- Tokenization preprocessing -------------------- #
    print(20 * "=", " Preprocessing tokenization", 20 * "=")
    print("\t* Creating tokenization and saving it...")
    preprocessor.create_tokenizations(data,
                                      os.path.join(targetdir, "dev_elmo.pkl"),
                                      os.path.join(targetdir, "dev_bert.pkl"))

    # -------------------- Test data preprocessing -------------------- #
    print(20*"=", " Preprocessing test set ", 20*"=")
    print("\t* Reading data...")
    data = preprocessor.read_data(os.path.join(inputdir, test_file))

    print("\t* Saving result...")
    with open(os.path.join(targetdir, "test_data.pkl"), "wb") as pkl_file:
        pickle.dump(data, pkl_file)


    # -------------------- Tokenization preprocessing -------------------- #
    print(20 * "=", " Preprocessing tokenization", 20 * "=")
    print("\t* Creating tokenization and saving it...")
    preprocessor.create_tokenizations(data,
                                      os.path.join(targetdir, "test_elmo.pkl"),
                                      os.path.join(targetdir, "test_bert.pkl"))


if __name__ == "__main__":
    default_config = "../../config/preprocessing/mednli_preprocessing.json"

    parser = argparse.ArgumentParser(description="Preprocess the MedNLI dataset")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to a configuration file for preprocessing MedNLI"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), "r") as cfg_file:
        config = json.load(cfg_file)

    preprocess_SNLI_data(
        os.path.normpath(os.path.join(script_dir, config["data_dir"])),
        os.path.normpath(os.path.join(script_dir, config["target_dir"])),
        lowercase=config["lowercase"],
        ignore_punctuation=config["ignore_punctuation"],
        num_words=config["num_words"],
        stopwords=config["stopwords"],
        labeldict=config["labeldict"]
    )
