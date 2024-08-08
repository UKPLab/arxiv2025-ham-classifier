import torch
import torch.nn as nn

from .circuit import Circuit, device
from .utils import KWArgsMixin, UpdateMixin


class BagOfWordsClassifier(nn.Module, KWArgsMixin):
    '''
    Bag of words baseline classifier
    '''
    def __init__(self, emb_dim):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, 1),
            nn.Sigmoid()
        )
        KWArgsMixin.__init__(self, emb_dim=emb_dim)
    
    def forward(self, input, seq_lengths):
        seq_lengths = seq_lengths.to(device=input.device)
        sent_emb = torch.sum(input, dim=1) / seq_lengths.view(-1, 1)
        pred = self.classifier(sent_emb).squeeze() # Squeeze to remove last dimension in binary classification tasks

        return pred, sent_emb
    
    def get_n_params(self):
        n_all_params = sum(p.numel() for p in self.parameters())
        n_params = {'n_all_params': n_all_params}
        return n_params


class RecurrentClassifier(nn.Module, KWArgsMixin):
    '''
    Simple RNN/LSTM classifier
    '''
    def __init__(self, emb_dim, hidden_dim, rnn_layers, architecture):
        super().__init__()
        if architecture == 'rnn':
            self.combiner = nn.RNN(input_size=emb_dim, hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)
        elif architecture == 'lstm':
            self.combiner = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)
        else:
            raise ValueError(f'Unknown architecture: {architecture}')
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        KWArgsMixin.__init__(self, emb_dim=emb_dim, hidden_dim=hidden_dim, 
                             rnn_layers=rnn_layers, architecture=architecture)
    
    def forward(self, input, seq_lengths):
        # (batch_size, max_seq_len, in_dim), (batch_size)
        assert input.size(0) == seq_lengths.size(0), 'Batch size mismatch'

        # Sort the input by decreasing sequence length, required by pack_padded_sequence
        sort_seq_length, perm_idx = seq_lengths.sort(descending=True)
        # Set sort_seq_length to 1 if the sequence length is 0
        sort_seq_length[sort_seq_length <= 0] = 1
        input = input[perm_idx]

        packed_seq_batch = nn.utils.rnn.pack_padded_sequence(input, lengths=sort_seq_length, batch_first=True)
        output, _ = self.combiner(packed_seq_batch)
        padded_output, output_lens = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # Calculate the indices of the last valid elements
        last_valid_indices = (sort_seq_length - 1).to(device=padded_output.device)

        # Use torch.gather to extract the last valid elements
        last_valid_elements = torch.gather(padded_output, 1, last_valid_indices.view(-1, 1, 1).expand(-1, 1, padded_output.size(2)))
        rnn_output = last_valid_elements.view(padded_output.size(0), padded_output.size(2))

        assert last_valid_elements.size(0) == input.size(0), 'Batch size mismatch'
        pred = self.classifier(rnn_output).squeeze() # Squeeze to remove last dimension in binary classification tasks

        # Unsort the output
        _, unperm_idx = perm_idx.sort()
        pred = pred[unperm_idx]
        rnn_output = rnn_output[unperm_idx]

        return pred, rnn_output
    
    def get_n_params(self):
        n_combiner_params = sum(p.numel() for p in self.combiner.parameters())
        n_classifier_params = sum(p.numel() for p in self.classifier.parameters())
        n_all_params = sum(p.numel() for p in self.parameters())
        n_params = {'n_combiner_params': n_combiner_params, 'n_classifier_params': n_classifier_params, 
                    'n_all_params': n_all_params}
        return n_params


class QuantumCircuitClassifier(nn.Module, KWArgsMixin, UpdateMixin):
    '''
    Quantum circuit classifier based on tomography
    '''

    def __init__(self, emb_dim, clas_type,
                 bias, pos_enc,
                 max_len=300, *args, **kwargs) -> None:
        '''
        emb_dim: size of the embedding
        clas_type: 'tomography' or 'hamiltonian'
        bias: 'vector', or 'none'
        pos_enc: 'learned' or 'none'
        max_len: maximum sentence length
        '''
        super().__init__()
        self.emb_size = emb_dim
        self.clas_type = clas_type
        self.bias = bias
        self.max_len = max_len
        self.pos_enc = pos_enc
        self.n_wires = (emb_dim - 1).bit_length() # Next power of 2 of log(emb_size)
        self.circuit = Circuit(n_wires=self.n_wires, *args, **kwargs)
        if clas_type == 'tomography':
            self.classifier = nn.Sequential(
                nn.Linear(2**self.n_wires, 1),
                nn.BatchNorm1d(1),
                nn.Sigmoid()
            )
        elif clas_type == 'hamiltonian':
            self.ham_param = nn.Parameter(torch.rand((2**self.n_wires, 2**self.n_wires)), ).type(torch.complex64)
            self.head = nn.Sequential(
                nn.BatchNorm1d(1),
                nn.Sigmoid()
            )

        if bias == 'vector':
            self.bias_param = nn.Parameter(torch.rand((emb_dim, 1)), )
        elif bias == 'none':
            self.bias_param = None
        if pos_enc == 'learned':
            self.pos_param = nn.Parameter(torch.rand((max_len, emb_dim))).type(torch.complex64)
        elif pos_enc == 'none':
            self.pos_param = None

        KWArgsMixin.__init__(self, emb_dim=emb_dim, bias=bias, clas_type=clas_type, pos_enc=pos_enc, max_len=max_len, 
                             **kwargs)
        self.update()

    def update(self):
        self.circuit.update()

    def to(self, device):
        super().to(device)
        if self.bias == 'vector':
            self.bias_param = self.bias_param.to(device)
        if self.pos_enc == 'learned':
            self.pos_param = self.pos_param.to(device)
        if self.clas_type == 'hamiltonian':
            self.ham_param = self.ham_param.to(device)
        return self

    def forward(self, x, seq_lengths):
        '''
        x: (batch_size, sent_len, emb_dim)
        lengths: (batch_size)

        Returns:
        (batch_size), (batch_size, emb_dim)
        '''
        seq_lengths = seq_lengths
        x = x.type(torch.complex64)
        
        # Add vector bias
        if self.bias == 'vector':
            batch_bias = [self.bias_param.view(1, -1).repeat(l, 1) for l in seq_lengths]
            batch_bias = nn.utils.rnn.pad_sequence(batch_bias, batch_first=True)
            batch_bias = batch_bias.to(device)
            x += batch_bias

        # Add positional encoding
        if self.pos_enc == 'learned':
            pos_enc = [self.pos_param[:l] for l in seq_lengths]
            pos_enc = nn.utils.rnn.pad_sequence(pos_enc, batch_first=True)
            x += pos_enc

        x = torch.nn.functional.pad(x, (0, 2**self.n_wires - self.emb_size))
        x = x.mean(dim=1) # (batch_size, emb_dim)
        x = x / torch.norm(x, dim=1).view(-1, 1)

        # Apply self.circuit to sentence
        circ_out = self.circuit(x)

        if self.clas_type == 'tomography':
            # Compute amplitudes
            circ_out = (circ_out * circ_out.conj()).real
            pred = self.classifier(circ_out).squeeze()
        elif self.clas_type == 'hamiltonian':
            hamiltonian = self.ham_param.triu() + self.ham_param.triu(1).H
            pred = torch.einsum('bi,ij,jb -> b', circ_out, hamiltonian, circ_out.H).real
            pred = self.head(pred.view(-1,1)).squeeze()

        return pred, circ_out

    def get_n_params(self):
        if self.bias == 'vector':
            bias_params = self.bias_param.numel()
        elif self.bias == 'none':
            bias_params = 0
        else:
            raise ValueError(f'Unknown bias {self.bias}')

        if self.pos_enc == 'learned':
            pos_enc_params = self.pos_param.numel()
        elif self.pos_enc == 'none':
            pos_enc_params = 0
        else:
            raise ValueError(f'Unknown positional encoding {self.pos_enc}')

        if self.clas_type == 'tomography':
            clas_params = sum(p.numel() for p in self.classifier.parameters())
        elif self.clas_type == 'hamiltonian':
            clas_params2 = self.ham_param.numel() 
            clas_params = int(clas_params2**0.5) * (int(clas_params2**0.5) - 1) // 2
            clas_params += sum(p.numel() for p in self.head.parameters())
        else:
            raise ValueError(f'Unknown classifier type {self.clas_type}')

        circ_params = sum(p.numel() for p in self.circuit.parameters() if p.requires_grad)
        all_params = bias_params + circ_params + clas_params + pos_enc_params
        n_params = {'n_bias_params': bias_params, 'n_circ_params': circ_params,
                    'n_pos_enc_params': pos_enc_params, 'n_all_params': all_params,
                    'n_clas_params': clas_params}
        return n_params
    

class MLPClassifier(nn.Module, KWArgsMixin):
    '''
    Multi-layer perceptron classifier
    '''
    def __init__(self, emb_dim, n_layers, hidden_dim):
        super().__init__()

        if n_layers < 2:
            raise ValueError('n_layers must be at least 2')
        self.classifier = nn.Sequential()
        self.classifier.add_module('input', nn.Linear(emb_dim, hidden_dim))
        self.classifier.add_module('relu_in', nn.ReLU())
        for i in range(n_layers-2):
            self.classifier.add_module(f'hidden_{i}', nn.Linear(hidden_dim, hidden_dim))
            self.classifier.add_module(f'relu_{i}', nn.ReLU())
        self.classifier.add_module('output', nn.Linear(hidden_dim, 1))
        self.classifier.add_module('sigmoid_out', nn.Sigmoid())

        KWArgsMixin.__init__(self, emb_dim=emb_dim, n_layers=n_layers, hidden_dim=hidden_dim)
    
    def forward(self, input, seq_lengths):
        seq_lengths = seq_lengths.to(device=input.device)
        sent_emb = torch.sum(input, dim=1) / seq_lengths.view(-1, 1)
        pred = self.classifier(sent_emb).squeeze() # Squeeze to remove last dimension in binary classification tasks

        return pred, sent_emb
    
    def get_n_params(self):
        n_all_params = sum(p.numel() for p in self.parameters())
        n_params = {'n_all_params': n_all_params}
        return n_params
    

