"""
Definition of the ESIM model.
"""
# Modified by Soumya Sharma, 2019.
# Aurelien Coet, 2018.

import torch
import torch.nn as nn

from .layers import RNNDropout, Seq2SeqEncoder, SoftmaxAttention, WeightedAttention
from .utils import get_mask, replace_masked

from allennlp.modules.elmo import Elmo, batch_to_ids
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from pymagnitude import *
import re
import numpy as np

class ElmoClass():
    def __init__(self, device):
        self.device = device
        bioelmo_options_file = "/home/soumyasharma/datafiles/biomed_elmo_options.json"
        bioelmo_weight_file = "/home/soumyasharma/datafiles/biomed_elmo_weights.hdf5"
        # Compute two different representation for each token.
        # Each representation is a linear weighted combination for the
        # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
        self.model = Elmo(bioelmo_options_file, bioelmo_weight_file, 2, dropout=0)
        self.model = self.model.to(self.device)

    def get_embeddings(self, data, max_length, embedding_dim):
        character_ids = batch_to_ids(data).to(self.device)
        elmo_output = self.model(character_ids)
        batch_size = len(data)
        embedding = torch.zeros(batch_size, max_length, embedding_dim, dtype=torch.float).to(self.device)
        mask = torch.ones(batch_size, max_length, dtype=torch.float).to(self.device)
        for idx, temp in enumerate(elmo_output['elmo_representations'][0]):
            embedding[idx][:len(data[idx])] = temp[:len(data[idx])]
            mask[idx][len(data[idx]):] = 0.0

        return embedding, mask

class BertClass():
    def __init__(self, device):
        self.device = device
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.to(self.device)
        self.model.eval()

    def get_embeddings(self, batched_data, max_length, embedding_dim):
        batch_size = len(batched_data)
        mask = torch.ones(batch_size, max_length, dtype=torch.float).to(self.device)
        indexed_tokens = []
        for idx, data in enumerate(batched_data):
            indexed_tokens.append(data + [110 for x in range(max_length - len(data))])
            mask[idx][len(data):] = 0.0

        tokens_tensor = torch.tensor(indexed_tokens).to(self.device)

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, attention_mask=mask, output_all_encoded_layers=False)
        return encoded_layers, mask

class FastTextClass():
    def __init__(self, device, ft_mag_path):
        self.ft_vectors = Magnitude(ft_mag_path)
        _r = re.compile(r'[()!@#$%^&-.\/]')
        self.clear_fn = lambda w: _r.sub('', w)
        self.device = device

    def get_embeddings(self, batched_data, max_length):
        raw_ = [[self.clear_fn(w) for w in x[0]] for x in batched_data]
        ft_output = self.ft_vectors.query(raw_, pad_to_length=max_length)

        return torch.FloatTensor(ft_output).to(self.device)

class GloveClass():
    def __init__(self, device, glove_path):
        print("Loading Glove Model")
        f = open(glove_path, 'r')
        model = {}
        for line in f:
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
        print("Done.", len(model), " words loaded!")
        self.model = model
        _r = re.compile(r'[()!@#$%^&-.\/]')
        self.clear_fn = lambda w: _r.sub('', w)
        self.device = device

    def get_embeddings(self, batched_data, max_length, embedding_dim):
        raw_ = [[self.clear_fn(w) for w in x[0]] for x in batched_data]

        batch_size = len(batched_data)

        dm_embedding = torch.zeros(batch_size, max_length, embedding_dim, dtype=torch.float).to(self.device)
        for i, s in enumerate(raw_):
            for j, w in enumerate(s):
                if w in self.model:
                    dm_embedding[i, j, :] = torch.from_numpy(self.model[w]).to(self.device)

        return dm_embedding

class DistmultClass():
    def __init__(self, device, dm_path):
        self.device = device
        self.dm_mat = {}
        with open(dm_path, 'r') as f:
            for i, train_e in enumerate(f):
                print(f'Reading Distmult: {i}', end='\r')
                w, v = train_e.strip().split('\t', 1)
                self.dm_mat[w] = np.array(eval(v))
        _r = re.compile(r'[()!@#$%^&-.\/]')
        
    def get_embeddings(self, batched_data, max_length, embedding_dim):
        umls_ = [x[1] for x in batched_data]

        batch_size = len(batched_data)

        dm_embedding = torch.zeros(batch_size, max_length, embedding_dim, dtype=torch.float).to(self.device)
        mask = torch.ones(batch_size, max_length, dtype=torch.float).to(self.device)
        for i, s in enumerate(umls_):
            for j, w in enumerate(s):
                w = w.replace('_', ' ')
                if w in self.dm_mat:
                    dm_embedding[i, j, :] = torch.from_numpy(self.dm_mat[w]).to(self.device)
            mask[i][len(s):] = 0.0

        return dm_embedding, mask

class ESIM(nn.Module):
    """
    Implementation of the ESIM model presented in the paper "Enhanced LSTM for
    Natural Language Inference" by Chen et al.
    """

    def __init__(self,
                 embedding_dim,
                 hidden_size,
                 embeddingString,
                 padding_idx=0,
                 distmult=0,
                 distmultPath=None,
                 distmultEmbeddingDim=None,
                 dropout=0.0,
                 num_classes=3,
                 multipassiterations=1,
                 lstm=True,
                 weightedattention=False,
                 testing=True,
                 sentiment=False,
                 device="cpu"):
        """
        Args:
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddingString: A string to indicate which embedding to use
            # embeddings: A tensor of size (vocab_size, embedding_dim) containing
            #     pretrained word embeddings. If None, word embeddings are
            #     initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the premises and
                hypotheses passed as input to the model. Defaults to 0.
            distmult: int value {0,1,2} where 0 means don't include. 
                1 means only include distmult. 2 for merge with generic embeddings.
            distmultPath: Path where distmult embeddings for dataset are stored. 
            distmultEmbeddingDim: dimension of distmult embeddings
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            multipassiterations: Number of iterations for multipass attention. 
                Defaults to 1
            lstm: Boolean value to include the lstm layer in ESIM model.
            weightedattention: Boolean value to include weighted attention in ESIM model.
            testing: Boolean value to test the pipeline. Each iteration would only be run once.
            sentiment: Boolean value to include sentiment vector for input of ESIM model.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(ESIM, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.embeddingString = embeddingString
        self.dropout = dropout
        self.num_classes = num_classes
        self.distmult = distmult
        self.distmultPath = distmultPath
        self.distmultEmbeddingDim = distmultEmbeddingDim
        self.multipassiterations = multipassiterations
        self.lstm = lstm
        self.testing = testing
        self.weightedattention = weightedattention
        self.sentiment = sentiment
        self.device = device

        fastTextPath = "/home/soumyasharma/Fasttext/wiki-news-300d-1M.magnitude"
        glovePath = "/home/soumyasharma/datafiles/glove.6B.300d.txt"

        if self.embeddingString == "elmo": self._embeddings = ElmoClass(self.device)
        elif self.embeddingString == "bert": self._embeddings = BertClass(self.device)
        elif self.embeddingString == "fasttext": self._embeddings = FastTextClass(self.device, fastTextPath)
        elif self.embeddingString == "glove": self._embeddings = GloveClass(self.device, glovePath)
        else: raise RuntimeError("embeddingString not specified")

        if self.distmult > 0: self._distmultembeddings = DistmultClass(self.device, distmultPath)
        if self.sentiment: self.embedding_dim += 1
        if self.distmult == 1: self.embedding_dim = self.distmultEmbeddingDim
        elif self.distmult == 2: self.embedding_dim = self.embedding_dim + self.distmultEmbeddingDim

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)

        if self.lstm:
            self._encoding = Seq2SeqEncoder(nn.LSTM,
                                        self.embedding_dim,
                                        self.hidden_size,
                                        bidirectional=True)

        if self.lstm:
            self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(2*4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))
        else:
            self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Linear(4*self.hidden_size,
                                                       self.hidden_size),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def initialize_layers(self, itr):
        if self.weightedattention:
            if self.lstm:
                setattr(self, "_attention" + str(itr), WeightedAttention(2*self.hidden_size).cuda())
            else:
                setattr(self, "_attention" + str(itr), WeightedAttention(self.embedding_dim).cuda())
        else: setattr(self, "_attention" + str(itr), SoftmaxAttention().cuda())

        if self.lstm:
            setattr(self, "_projection" + str(itr), nn.Sequential(nn.Linear(4*2*self.hidden_size,
                                                    self.hidden_size).cuda(),
                                                    nn.ReLU()))
            setattr(self, "_composition" + str(itr), Seq2SeqEncoder(nn.LSTM,
                                           self.hidden_size,
                                           self.hidden_size,
                                           bidirectional=True).cuda())
        else:
            setattr(self, "_projection" + str(itr), nn.Sequential(nn.Linear(4*self.embedding_dim,
                                                   self.hidden_size).cuda(),
                                                   nn.ReLU()))

    def forward(self,
                premises,
                premises_lengths,
                hypotheses,
                hypotheses_lengths,
                premise_polarities,
                hypothesis_polarities,
                batch_embeddings,
                embeddingString,
                max_premise_length,
                max_hypotheses_length):
        """
        This needs padded instance of premises
        Args:
            premises: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, premises_length).
            premises_lengths: A 1D tensor containing the lengths of the
                premises in 'premises'.
            hypothesis: A batch of varaible length sequences of word indices
                representing hypotheses. The batch is assumed to be of size
                (batch, hypotheses_length).
            hypotheses_lengths: A 1D tensor containing the lengths of the
                hypotheses in 'hypotheses'.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """

        if self.distmult != 1:
            if self.distmult == 2:
                dim = self.embedding_dim - self.distmultEmbeddingDim
            else:
                dim = self.embedding_dim
            if self.sentiment:
                dim -= 1

            if self.embeddingString=="elmo" or self.embeddingString=="bert":
                embedded_premises, premises_mask = self._embeddings.get_embeddings(batch_embeddings["premises"],
                                                                                   max_premise_length,
                                                                                   dim,
                                                                                   )
                embedded_hypotheses, hypotheses_mask = self._embeddings.get_embeddings(batch_embeddings["hypotheses"],
                                                                                       max_hypotheses_length,
                                                                                       dim,
                                                                                        )
            if self.embeddingString=="fasttext":
                embedded_premises = self._embeddings.get_embeddings(premises, max_premise_length)
                embedded_hypotheses = self._embeddings.get_embeddings(hypotheses, max_hypotheses_length)
            if self.embeddingString=="glove":
                embedded_premises = self._gloveembeddings.get_embeddings(premises, max_premise_length, self.embedding_dim - self.distmultEmbeddingDim)
                embedded_hypotheses = self._gloveembeddings.get_embeddings(hypotheses, max_hypotheses_length, self.embedding_dim - self.distmultEmbeddingDim)

        if self.distmult > 0:
            distmult_embedded_premises, distmult_premises_mask = self._distmultembeddings.get_embeddings(premises,
                                                                                                         max_premise_length,
                                                                                                         self.distmultEmbeddingDim)
            distmult_embedded_hypotheses, distmult_hypotheses_mask = self._distmultembeddings.get_embeddings(hypotheses,
                                                                                                             max_hypotheses_length,
                                                                                                             self.distmultEmbeddingDim)
            premises_mask = distmult_premises_mask
            hypotheses_mask = distmult_hypotheses_mask

        if self.distmult == 1:
            embedded_premises = distmult_embedded_premises
            embedded_hypotheses = distmult_embedded_hypotheses
            premises_mask = distmult_premises_mask
            hypotheses_mask = distmult_hypotheses_mask
        elif self.distmult == 2:
            embedded_premises = torch.cat((embedded_premises, distmult_embedded_premises), dim=2)
            embedded_hypotheses = torch.cat((embedded_hypotheses, distmult_embedded_hypotheses), dim=2)

        if self.sentiment: embedded_premises = torch.cat((embedded_premises, premise_polarities.unsqueeze(2)), dim=2)

        if self.sentiment: embedded_hypotheses = torch.cat((embedded_hypotheses, hypothesis_polarities.unsqueeze(2)), dim=2)

        if self.dropout:
            embedded_premises = self._rnn_dropout(embedded_premises)
            embedded_hypotheses = self._rnn_dropout(embedded_hypotheses)

        if self.lstm:
            encoded_premises = self._encoding(embedded_premises,
                                              premises_lengths, max_premise_length)
            encoded_hypotheses = self._encoding(embedded_hypotheses,
                                                hypotheses_lengths, max_hypotheses_length)
        else:
            encoded_premises = embedded_premises
            encoded_hypotheses = embedded_hypotheses

        for idx in range(self.multipassiterations):
            if idx != 0:
                encoded_premises = v_ai
                encoded_hypotheses = v_bj

            attention = getattr(self, '_attention' + str(idx))

            attended_premises, attended_hypotheses , similarity_matrix = \
                attention(encoded_premises, premises_mask,
                          encoded_hypotheses, hypotheses_mask)

            enhanced_premises = torch.cat([encoded_premises,
                                           attended_premises,
                                           encoded_premises - attended_premises,
                                           encoded_premises * attended_premises],
                                          dim=-1).to(self.device)
            enhanced_hypotheses = torch.cat([encoded_hypotheses,
                                             attended_hypotheses,
                                             encoded_hypotheses -
                                             attended_hypotheses,
                                             encoded_hypotheses *
                                             attended_hypotheses],
                                            dim=-1).to(self.device)

            projection = getattr(self, '_projection' + str(idx))

            projected_premises = projection(enhanced_premises)
            projected_hypotheses = projection(enhanced_hypotheses)

            if self.dropout:
                projected_premises = self._rnn_dropout(projected_premises)
                projected_hypotheses = self._rnn_dropout(projected_hypotheses)

            if self.lstm:
                composition = getattr(self, '_composition' + str(idx))
                v_ai = composition(projected_premises, premises_lengths, max_premise_length)
                v_bj = composition(projected_hypotheses, hypotheses_lengths, max_hypotheses_length)
            else:
                v_ai = projected_premises
                v_bj = projected_hypotheses

        v_a_avg = torch.sum(v_ai * premises_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(premises_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * hypotheses_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(hypotheses_mask, dim=1, keepdim=True)

        v_a_max, _ = replace_masked(v_ai, premises_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, hypotheses_mask, -1e7).max(dim=1)

        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        logits = self._classification(v)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities, similarity_matrix

def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0
