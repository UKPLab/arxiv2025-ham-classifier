import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from gensim.models import KeyedVectors
from .utils import KWArgsMixin


class Embedder(nn.Module, KWArgsMixin):
    def __init__(self, emb_dim=10, weights_path=None, dict_path=None, vocab_size=None, freeze=True, padding='zeros'):
        '''
        Creates an Embedder object from word2vec embeddings or dictionary
        
        emb_dim: dimension of the word2vec embeddings. Ignored if weights_path is specified
        weights_path: path to the word2vec weights file
        dict_path: path to the word2vec dictionary file
        vocab_size: number of words to load from the word2vec dictionary
        freeze: whether to freeze the word2vec embeddings. Ignored if weights_path is specified
        '''
        super().__init__()
        if weights_path is not None and dict_path is not None:
            raise ValueError('Cannot specify both weights_path and dict_path')
        elif weights_path is None and dict_path is None:
            self._key_to_index = {'<unk>': 0}
            self.index_to_key = ['<unk>']
            self.embedding = nn.Embedding(num_embeddings=len(self._key_to_index), embedding_dim=emb_dim, padding_idx=0)
            self.resizer = nn.Identity()
        elif dict_path is not None:  # Load only word2vec dictionary
            self._key_to_index = self._build_vocabulary(dict_path, limit=vocab_size)
            self.index_to_key = list(self._key_to_index.keys())
            self.vocabulary_size = len(self._key_to_index)
            self.embedding = nn.Embedding(num_embeddings=self.vocabulary_size, embedding_dim=emb_dim, padding_idx=0)
            self.resizer = nn.Identity()
        else:   # Load GoogleNews word2vec embeddings with gensim
            w2v = KeyedVectors.load_word2vec_format(weights_path, binary=True, limit=vocab_size)
            self._key_to_index = w2v.key_to_index
            self.index_to_key = w2v.index_to_key
            self.vocabulary_size = len(self._key_to_index)
            self.w2v_dim = w2v.vector_size
            emb_dim = self.w2v_dim
            vectors = torch.FloatTensor(w2v.vectors)

            # TODO: apply alternative solution where unk vector is taken from original word2vec
            # If there is no <unk> token, add it along with its vector
            # Embedding value is the same of last word in the vocabulary
            if '<unk>' not in self._key_to_index:
                self._key_to_index['<unk>'] = len(self._key_to_index)
                self.index_to_key.append('<unk>')
                self.vocabulary_size += 1
                vectors = torch.cat((vectors, vectors[-1].unsqueeze(0)), dim=0)
            
            assert '<unk>' in self._key_to_index, 'No <unk> token in vocabulary'
            self.embedding = nn.Embedding.from_pretrained(vectors, freeze=freeze)
        self.emb_dim = emb_dim
        self.padding = padding
        KWArgsMixin.__init__(self, emb_dim=emb_dim, weights_path=weights_path, dict_path=dict_path, vocab_size=vocab_size, freeze=freeze) # Saves kwargs

    def _build_vocabulary(self, path, limit=None):
        # Use word2vec vocabulary
        return KeyedVectors.load_word2vec_format(path, binary=True, limit=limit).key_to_index

    def forward(self, text):
        # Converts text into a list of indices
        # Assumes text is tokenized already
        if isinstance(text, str):
            text = [text]
        # If the word is in vocabulary, use its index
        # If the word is not in vocabulary, use the index of <unk>
        # If <unk> is not in vocabulary, use 0
        text = [sentence.lower() for sentence in text] # Lowercase all words
        indices = [[self._key_to_index[word] if word in self._key_to_index else self._key_to_index['<unk>'] if '<unk>' in self._key_to_index else 0 for word in sentence.split()] for sentence in text]
        indices = [torch.LongTensor(sentence).to(device=self.embedding.weight.device) for sentence in indices]
        seq_lengths = [len(sentence) for sentence in indices]
        seq_lengths = torch.tensor(seq_lengths)
        if self.padding == 'zeros':
            torch_embedding = [self.embedding(idx) for idx in indices]
            torch_embedding = nn.utils.rnn.pad_sequence(torch_embedding, batch_first=True)

            # indices = nn.utils.rnn.pad_sequence(indices, batch_first=True)
            # torch_embedding = self.embedding(indices)
        else:
            torch_embedding = [self.embedding(sentence) for sentence in indices]
        # torch_embedding = nn.functional.softmax(torch_embedding, dim=2)
        # Tensor of size (n sentences, n words, embedding_dim)
        return torch_embedding, seq_lengths  

    # TODO: give alternative definition based on MSE
    def mrr_score(self, inputs, text_labels, reduction='mean'):
        if isinstance(inputs, str) or isinstance(inputs, list):
            inputs = self.forward(inputs)
        # x is a tensor of size (n_sentences, embedding_dim)
        assert len(inputs.shape) == 2, 'x must be a tensor of size (n_sentences, embedding_dim)'
        assert isinstance(text_labels, list), 'text_labels must be a list of strings'
        # Compute norms of all vectors
        norms = torch.norm(self.embedding.weight.data, dim=1)
        # Compute norms of all vectors
        x_norms = torch.norm(inputs, dim=1)
        # Compute dot product between all vectors and x
        dot_products = torch.matmul(self.embedding.weight.data, inputs.T)
        # Compute cosine similarity between all vectors and x
        cos_sim = dot_products / (norms[:, None] * x_norms[None, :])
        # Compute ranks of all vectors
        labels = self.key_to_index(text_labels)
        ranks = torch.argsort(cos_sim, dim=0, descending=True) # Ranks contiene gli indici organizzati per colonne
        ranks = ranks[labels, torch.arange(ranks.shape[1])]
        if reduction == 'none':
            return 1 / (ranks+1)
        elif reduction == 'mean':
            return torch.mean(1 / (ranks+1))
        elif reduction == 'sum':
            return torch.sum(1 / (ranks+1))
        else:
            raise ValueError('Invalid reduction')         
        
    def vec2word(self, vec, k=1, filter_words=None):
        if len(vec.shape) != 1:
            vec = torch.squeeze(vec)
        # Finds the closest k words to the given vector(s) in terms of cosine similarity
        similarity = torch.matmul(self.embedding.weight.data, vec)
        similarity = similarity / (torch.norm(self.embedding.weight.data, dim=1) * torch.norm(vec))
        tmp_index_to_key = self.index_to_key.copy()
        if filter_words is not None:
            tmp_index_to_key = {i : word for i, word in enumerate(tmp_index_to_key) if word in filter_words}
            return [tmp_index_to_key[i] for i in torch.argsort(similarity, dim=0, descending=True).detach().cpu().numpy() if i in tmp_index_to_key][:k]
        else:
            return [tmp_index_to_key[i] for i in torch.argsort(similarity, dim=0, descending=True)[:k].detach().cpu().numpy()]

    def key_to_index(self, key: list | str):
        if isinstance(key, list):
            return [self._key_to_index[k] if k in self._key_to_index else self._key_to_index['<unk>'] for k in key]
        else:
            return self._key_to_index[key] if key in self._key_to_index else self._key_to_index['<unk>']
