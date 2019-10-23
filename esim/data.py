"""
Preprocessor and dataset definition for MedNLI.
"""
# Modified by Soumya Sharma, 2019
# Aurelien Coet, 2018.

import string
import torch
import numpy as np
import pickle

from collections import Counter
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer

class Tokenizer():
    def __init__(self):
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def get_bert_tokenization(self, sentence):
        tokenization = self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(sentence))
        return tokenization, len(tokenization)

    def get_batched_bert_tokenization(self, sentences):
        tokenized_text = []
        tokenized_length = []
        for sentence in sentences:
            # sentence[0] to pick the non-distmult representation of sentence. Refer to read_data of PreProcessor
            text, length = self.get_bert_tokenization(sentence[0])
            tokenized_text.append(text)
            tokenized_length.append(length)
        return tokenized_text, tokenized_length

    def get_elmo_tokenization(self, sentence):
        tokenization = [w for w in sentence.rstrip().split()]
        return tokenization, len(tokenization)

    def get_batched_elmo_tokenization(self, sentences):
        tokenized_text = []
        tokenized_length = []
        for sentence in sentences:
            # sentence[0] to pick the non-distmult representation of sentence. Refer to read_data of PreProcessor
            # text, length = self.get_elmo_tokenization(sentence[0])
            text, length = sentence[0], len(sentence[0])
            tokenized_text.append(text)
            tokenized_length.append(length)
        return tokenized_text, tokenized_length

class Preprocessor(object):
    """
    Preprocessor class for MedNLI datasets.

    The class can be used to read MedNLI datasets, build worddicts for them
    and transform their premises, hypotheses and labels into lists of
    integer indices.
    """

    def __init__(self,
                 lowercase=False,
                 ignore_punctuation=False,
                 num_words=None,
                 stopwords=[],
                 labeldict={}):
        """
        Args:
            lowercase: A boolean indicating whether the words in the datasets
                being preprocessed must be lowercased or not. Defaults to
                False.
            ignore_punctuation: A boolean indicating whether punctuation must
                be ignored or not in the datasets preprocessed by the object.
            num_words: An integer indicating the number of words to use in the
                worddict of the object. If set to None, all the words in the
                data are kept. Defaults to None.
            stopwords: A list of words that must be ignored when building the
                worddict for a dataset. Defaults to an empty list.
        """
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.labeldict = labeldict

    def read_data(self, filepath):
        """
        Read the premises, hypotheses and labels from some NLI dataset's
        file and return them in a dictionary. The file should be in the same
        form as SNLI's .txt files.

        Args:
            filepath: The path to a file containing some premises, hypotheses
                and labels that must be read. The file should be formatted in
                the same way as the SNLI (and MultiNLI) dataset.

        Returns:
            A dictionary containing three lists, one for the premises, one for
            the hypotheses, and one for the labels in the input data.
        """
        with open(filepath, "r", encoding="utf8") as input_data:
            ids, premises, hypotheses, premises_length, hypotheses_length, premise_polarities, hypothesis_polarities, labels = [], [], [], [], [], [], [], []

            for idx, line in enumerate(input_data):
                print(f"Reading instance {idx}", end='\r')
                line = eval(line)
                hypothesis = eval(line['hypothesis'])
                premise = eval(line['premise'])
                label = line['label']

                # Align the nltk premise and the umls premise
                premise, premise_umls, premise_polarity = self.aligner(premise)
                hyp, hyp_umls, hypothesis_polarity = self.aligner(hypothesis)

                ids.append(idx)
                premises.append((premise, premise_umls))
                hypotheses.append((hyp, hyp_umls))
                labels.append(self.labeldict[label])
                premises_length.append(len(premise))
                hypotheses_length.append(len(hyp))
                premise_polarities.append(premise_polarity)
                hypothesis_polarities.append(hypothesis_polarity)

            return {"ids": ids,
                    "premises": premises,
                    "hypotheses": hypotheses,
                    "labels": labels,
                    "premises_lengths": premises_length,
                    "hypotheses_lengths": hypotheses_length,
                    "premise_polarities": premise_polarities,
                    "hypothesis_polarities": hypothesis_polarities,
                    "max_premise_length": max(premises_length),
                    "max_hypothesis_length": max(hypotheses_length)}

    def aligner(self, h):
        ht = h['text']

        normal_ht = ht.split(" ")
        return normal_ht, normal_ht, [0]*len(normal_ht)

    # def pair_umls(self, h):

    #     ht = h['text']

    #     normal_ht = ht.split(" ")

    #     mods = []
    #     for ph in h['phrases']:
    #         pht = ph["text"]
    #         for mapping in ph["Mappings"]:
    #             rep = mapping['CandidatePreferred'].replace(" ", "_")
    #             polarity = mapping["Negated"]
    #             for mw, cpi in zip(mapping['MatchedWords'], mapping['ConceptPIs']):
    #                 l = int(cpi['StartPos'])
    #                 r = l + int(cpi['Length'])
    #                 mods.append((l, r, rep, polarity))

    #     final_text = ''
    #     last_access = 0
    #     mods = sorted(mods, key=lambda x: x[0])
    #     #     print(mods)
    #     for mod in mods:
    #         l, r, rep, pol = mod
    #         if l < last_access:
    #             continue
    #         if l > last_access:
    #             final_text += ht[last_access:l]
    #             # polarities.extend([0]*len(ht[last_access:l].strip().split()))
    #         last_access = r
    #         w = ht[l:r]
    #         LW = len(w.split())
    #         final_text += ' '.join([rep] * LW)
    #         # polarities.extend([pol] * LW)
    #         # for _ in range(len(w.split())):
    #         #    print(f"Replace: '{ht[l:r]}' by '{rep}'")
    #     final_text += ht[last_access:]

    #     WL, DML = ht.split(" "), final_text.split(" ")

    #     polarities = [0] * len(DML)
    #     for j in range(len(mods)):
    #         for idx, item in enumerate(DML):
    #             if mods[j][2] in item:
    #                 DML[idx] = mods[j][2]
    #                 polarities[idx] = int(mods[j][3])

    #     # if 1 in polarities:
    #     #     print("hello!")
    #     assert len(WL) == len(DML)
    #     assert len(WL) == len(polarities)
    #     return WL, DML, polarities

    def create_tokenizations(self, data, elmo_file, bert_file):
        tokenizer = Tokenizer()
        tokenized_premises, tokenized_premises_lengths = tokenizer.get_batched_elmo_tokenization(data["premises"])
        tokenized_hypotheses, tokenized_hypotheses_lengths = tokenizer.get_batched_elmo_tokenization(data["hypotheses"])

        tokenized_data = {
            "ids": data["ids"],
            "premises": tokenized_premises,
            "premises_lengths": tokenized_premises_lengths,
            "hypotheses": tokenized_hypotheses,
            "hypotheses_lengths": tokenized_hypotheses_lengths,
            "labels": data["labels"],
            'max_premise_length': max(tokenized_premises_lengths),
            'max_hypothesis_length': max(tokenized_hypotheses_lengths)
        }

        print("\t* Saving result...")
        with open(elmo_file, "wb") as pkl_file:
            pickle.dump(tokenized_data, pkl_file)

        tokenized_premises, tokenized_premises_lengths = tokenizer.get_batched_bert_tokenization(data["premises"])
        tokenized_hypotheses, tokenized_hypotheses_lengths = tokenizer.get_batched_bert_tokenization(data["hypotheses"])
        
        tokenized_data = {
             "ids": data["ids"],
             "premises": tokenized_premises,
             "premises_lengths": tokenized_premises_lengths,
             "hypotheses": tokenized_hypotheses,
             "hypotheses_lengths": tokenized_hypotheses_lengths,
             "labels": data["labels"],
             "max_premise_length": max(tokenized_premises_lengths),
             "max_hypothesis_length": max(tokenized_hypotheses_lengths)
        }
        
        print("\t* Saving result...")
        with open(bert_file, "wb") as pkl_file:
            pickle.dump(tokenized_data, pkl_file)

        return

class MedNLIEmbedding(Dataset):
    """
    Dataset class for MedNLI Embeddings.

    The class can be used to read preprocessed datasets where the premises,
    hypotheses and labels have been transformed to their embedding matrices.
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """
    def __init__(self,
                 data,
                 batch_size=64):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            batch_size: An integer indicating the batch size the data should 
                be processed in. Defaults to 64
        """
        self.batch_size = batch_size
        self.ndx = 0

        self.num_sequences = len(data["premises"])
        self.data = data

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {"id": self.data["ids"][index],
                "premise": self.data["premises"][index],
                "premise_length": min(self.data['premises_lengths'][index],
                                      self.data.max_premise_length),
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.data['hypotheses_lengths'][index],
                                         self.data.max_hypothesis_length),
                "label": self.data["labels"][index],
                "max_premise_length": self.data["max_premise_length"],
                "max_hypothesis_length": self.data["max_hypothesis_length"]
                }

    def __iter__(self):
        return self

    def __next__(self):
        if self.ndx >= self.num_sequences:
            self.ndx = 0
            raise StopIteration
        else:
            ndx = self.ndx
            batch_size = self.batch_size
            l = self.num_sequences
            self.ndx = self.ndx + self.batch_size
            return {
                "ids": self.data["ids"][ndx:min(ndx + batch_size, l)],
                "premises": self.data["premises"][ndx:min(ndx + batch_size, l)],
                "premises_lengths": self.data['premises_lengths'][ndx:min(ndx + batch_size, l)],
                "hypotheses": self.data["hypotheses"][ndx:min(ndx + batch_size, l)],
                "hypotheses_lengths": self.data['hypotheses_lengths'][ndx:min(ndx + batch_size, l)],
                "labels": self.data["labels"][ndx:min(ndx + batch_size, l)],
                "max_premise_length": self.data["max_premise_length"],
                "max_hypothesis_length": self.data["max_hypothesis_length"]
            }

    def get_batch(self, ndx):
        if ndx >= self.num_sequences:
            ndx = 0
            raise StopIteration
        else:
            batch_size = self.batch_size
            l = self.num_sequences
            return {
                "ids": self.data["ids"][ndx:min(ndx + batch_size, l)],
                "premises": self.data["premises"][ndx:min(ndx + batch_size, l)],
                "premises_lengths": self.data['premises_lengths'][ndx:min(ndx + batch_size, l)],
                "hypotheses": self.data["hypotheses"][ndx:min(ndx + batch_size, l)],
                "hypotheses_lengths": self.data['hypotheses_lengths'][ndx:min(ndx + batch_size, l)],
                "labels": self.data["labels"][ndx:min(ndx + batch_size, l)],
                "max_premise_length": self.data["max_premise_length"],
                "max_hypothesis_length": self.data["max_hypothesis_length"]
            }

class MedNLIDataset(Dataset):
    """
    Dataset class for MedNLI datasets.

    The class can be used to read raw datasets where the premises,
    hypotheses and labels have been pickled.
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """
    def __init__(self,
                 data,
                 batch_size=64):
        """
        Args:
            data: A dictionary containing the preprocessed premises,
                hypotheses and labels of some dataset.
            batch_size: An integer indicating the batch size the data should 
                be processed in. Defaults to 64
        """
        self.batch_size = batch_size
        self.ndx = 0

        self.num_sequences = len(data["premises"])
        self.data = data

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {"id": self.data["ids"][index],
                "premise": self.data["premises"][index],
                "premise_length": min(self.data['premises_lengths'][index],
                                      self.data.max_premise_length),
                "hypothesis": self.data["hypotheses"][index],
                "hypothesis_length": min(self.data['hypotheses_lengths'][index],
                                         self.data.max_hypothesis_length),
                "premise_polarities": self.data["premise_polarities"][index],
                "hypothesis_polarities": self.data["hypothesis_polarities"][index],
                "label": self.data["labels"][index],
                "max_premise_length": self.data["max_premise_length"],
                "max_hypothesis_length": self.data["max_hypothesis_length"]
                }

    def __iter__(self):
        return self

    def __next__(self):
        if self.ndx >= self.num_sequences:
            self.ndx = 0
            raise StopIteration
        else:
            ndx = self.ndx
            batch_size = self.batch_size
            l = self.num_sequences
            self.ndx = self.ndx + self.batch_size
            return {
                "ids": self.data["ids"][ndx:min(ndx + batch_size, l)],
                "premises": self.data["premises"][ndx:min(ndx + batch_size, l)],
                "premises_lengths": self.data['premises_lengths'][ndx:min(ndx + batch_size, l)],
                "hypotheses": self.data["hypotheses"][ndx:min(ndx + batch_size, l)],
                "hypotheses_lengths": self.data['hypotheses_lengths'][ndx:min(ndx + batch_size, l)],
                "premise_polarities": self.data['premise_polarities'][ndx:min(ndx + batch_size, l)],
                "hypothesis_polarities": self.data['hypothesis_polarities'][ndx:min(ndx + batch_size, l)],
                "labels": self.data["labels"][ndx:min(ndx + batch_size, l)],
                "max_premise_length": self.data["max_premise_length"],
                "max_hypothesis_length": self.data["max_hypothesis_length"]
            }
