import itertools
import torch
import torch.nn as nn

from .circuit import  Circuit, I, X, Y, Z, device
from .utils import KWArgsMixin, UpdateMixin


class HamiltonianClassifier(nn.Module, KWArgsMixin, UpdateMixin):
    '''
    Simulated classification based on a quantum Hamiltonian
    '''

    def __init__(self, emb_dim, circ_in, bias, batch_norm, n_paulis=None,
                 pauli_strings=None, strategy='full', pauli_weight=None, pos_enc=None, n_wires=None,
                 max_len=1024, n_classes=2, *args, **kwargs) -> None:
        '''
        emb_dim: size of the embedding
        hamiltonian: 'pure' or 'mixed'
        circ_in: 'sentence' or 'zeros'
        bias: 'matrix', 'vector', 'diag', 'single' or 'none'
        batch_norm: bool
        n_paulis: number of Pauli measurements
        strategy: 'full' or 'simplified'
        pos_enc: 'learned' or 'none'
        n_wires: number of wires for the circuit
        max_len: maximum sentence length
        '''
        super().__init__()

        self.emb_dim = emb_dim
        self.circ_in = circ_in
        self.bias = bias
        self.strategy = strategy
        self.max_len = max_len
        self.pos_enc = pos_enc
        self.n_classes = n_classes
        self.pauli_weight = pauli_weight

        if n_classes != 2 and strategy != 'simplified':
            raise ValueError('Only binary classification is supported for full strategy')

        if batch_norm:
            batch_norm_dim = 1 if n_classes == 2 else self.n_classes
            self.batch_norm = nn.BatchNorm1d(batch_norm_dim)
        else:
            self.batch_norm = None
        
        if strategy == 'full':
            self.n_wires = (emb_dim - 1).bit_length() # Next power of 2 of log(emb_size)
            self.circuit = Circuit(n_wires=self.n_wires, *args, **kwargs)
            if pos_enc == 'learned':
                self.pos_param = nn.Parameter(torch.rand((max_len))).type(torch.complex64)
            if n_wires is not None:
                print('Warning: n_wires is ignored when strategy is full')
            if n_paulis is not None:
                print('Warning: n_paulis is ignored when strategy is full')
            if pauli_weight is not None:
                print('Warning: pauli_weight is ignored when strategy is full')
        elif strategy == 'simplified':
            assert n_wires is not None, 'n_wires must be defined for simplified strategy'
            self.n_wires = n_wires
            self.circuit = Circuit(n_wires=self.n_wires, *args, **kwargs)

            if pos_enc is not None:
                print('Warning: pos_enc is ignored when strategy is simplified')
            
            # Define the Pauli measurements randomly
            all_paulis = [I(1), X(), Y(), Z()]
            if pauli_strings is not None:
                print('Warning: pauli_weights is ignored when pauli_strings is defined')
                assert n_paulis == len(pauli_strings), 'Number of Pauli strings must match n_paulis'
                assert pauli_strings.shape == (n_paulis, self.n_wires), 'Pauli strings must be of shape (n_paulis, n_wires)'
                assert pauli_strings.min() >= 0 and pauli_strings.max() <= len(all_paulis), 'Pauli strings must be in [0, len(all_paulis)]'
                self.pauli_strings = pauli_strings
            else:
                if pauli_weight is None:
                    pauli_strings = torch.randint(0, len(all_paulis), (n_paulis, self.n_wires))
                    self.pauli_strings = pauli_strings
                elif pauli_weight == 'max':
                    self.pauli_strings = self._generate_pauli_strings(self.n_wires, self.n_wires)
                    self.n_paulis = self.pauli_strings.shape[0]
                elif pauli_weight == 'half':
                    self.pauli_strings = self._generate_pauli_strings((self.n_wires + 1) // 2, self.n_wires)
                    self.n_paulis = self.pauli_strings.shape[0]
                elif isinstance(pauli_weight, int):
                    self.pauli_strings = self._generate_pauli_strings(pauli_weight, self.n_wires)
                    self.n_paulis = self.pauli_strings.shape[0]
                else:
                    raise ValueError(f'Invalid pauli_weight {pauli_weight}')
                                    

            measurements = []
            for row in self.pauli_strings:
                pauli = all_paulis[row[0]]
                for p in row[1:]:
                    pauli = torch.kron(pauli, all_paulis[p])
                measurements.append(pauli)
            self.measurements = torch.stack(measurements).to(device)
            if self.n_classes == 2:
                self.measurement_map = nn.Linear(emb_dim, self.n_paulis)
            else:
                self.measurement_map = nn.ModuleList([nn.Linear(emb_dim, self.n_paulis) for _ in range(n_classes)])

        if bias == 'matrix':
            self.bias_param = nn.Parameter(torch.rand((2**self.n_wires, 2**self.n_wires)), )
        if bias == 'vector':
            self.bias_param = nn.Parameter(torch.rand((emb_dim, 1)), )
        elif bias == 'diag':
            self.bias_param = nn.Parameter(torch.rand((2**self.n_wires, 1)), )
        elif bias == 'single':
            self.bias_param = nn.Parameter(torch.rand((1, 1)), )

        KWArgsMixin.__init__(self, emb_dim=emb_dim, circ_in=circ_in, bias=bias, batch_norm=batch_norm, n_paulis=self.n_paulis,
                                pauli_strings=self.pauli_strings, strategy=strategy, pauli_weight=pauli_weight, pos_enc=pos_enc, 
                                n_wires=n_wires, max_len=max_len, **kwargs)
        self.update()

    def update(self):
        self.circuit.update()

    def to(self, device):
        super().to(device)
        if self.batch_norm:
            self.batch_norm = self.batch_norm.to(device)
        if self.bias == 'matrix' or self.bias == 'vector':
            self.bias_param = self.bias_param.to(device)
        if self.pos_enc == 'learned' and self.strategy == 'full':
            self.pos_param = self.pos_param.to(device)
        return self

    def _generate_pauli_strings(self, pauli_weights, n_wires):
        assert pauli_weights <= n_wires, f'Pauli weight {pauli_weights} too large for number of wires {n_wires}'
        assert pauli_weights > 0, 'Pauli weights must be greater than 0'
        pauli_indices = [1,2,3]

        # Generate all combinations of Pauli strings with given weight     
        combinations = []    
        for i in itertools.combinations_with_replacement(pauli_indices, pauli_weights):
            i = list(i)
            i = i + [0] * (n_wires - len(i))
            combinations += list(itertools.permutations(i, n_wires))
        combinations = list(set(combinations))

        combinations = torch.tensor(combinations)
        return combinations

    def hamiltonian(self, x, seq_lengths):
        '''
        x: (batch_size, sent_len, emb_dim)
        lengths: (batch_size)

        Returns:
        (batch_size, 2**n_wires, 2**n_wires)
        '''
        if self.strategy == 'full':
            return self._hamiltonian_full(x, seq_lengths)
        elif self.strategy == 'simplified':
            return self._hamiltonian_sim(x, seq_lengths)
        else:
            raise ValueError(f'Unknown strategy {self.strategy}')

    def _hamiltonian_full(self, x, seq_lengths):
        x = x.type(torch.complex64).to(device)
        seq_lengths = seq_lengths.to(device)
        
        # Add vector bias
        if self.bias == 'vector':
            batch_bias = [self.bias_param.view(1, -1).repeat(l, 1) for l in seq_lengths]
            batch_bias = nn.utils.rnn.pad_sequence(batch_bias, batch_first=True)
            x += batch_bias

        # Build hamiltonians
        # Outer product from (batch_size, sent_len, emb_dim) to (batch_size, emb_size, emb_size)
        if self.pos_enc == 'learned':
            pos_enc = self.pos_param[:x.shape[1]].type(torch.complex64)
            x = torch.einsum('s, bsi, bsj -> bij', pos_enc, x, x) / seq_lengths.view(-1, 1, 1)
        elif self.pos_enc == 'none':
            x = torch.einsum('bsi, bsj -> bij', x, x) / seq_lengths.view(-1, 1, 1)
        else:
            raise ValueError(f'Unknown positional encoding {self.pos_enc}')

        # Pad emb_size to next power of 2 (batch_size, 2**n_wires, 2**n_wires)
        x = torch.nn.functional.pad(
            x, (0, 2**self.n_wires - self.emb_dim, 0, 2**self.n_wires - self.emb_dim))

        # Add bias
        if self.bias == 'matrix': # Full matrix
            h0 = self.bias_param.triu() + self.bias_param.triu(1).H
        elif self.bias == 'vector': # Done before
            h0 = torch.zeros_like(x[0])
        elif self.bias == 'diag': # Diagonal matrix
            h0 = torch.diag(self.bias_param.view(-1))
        elif self.bias == 'single': # Constant * Identity
            h0 = self.bias_param * I(self.n_wires)
        elif self.bias == 'none': # No bias
            h0 = torch.zeros_like(x[0])
        else:
            raise ValueError(f'Unknown bias {self.bias}')
        x = x + h0
        x = torch.nn.functional.normalize(x,dim=0)
        return x
    
    def _hamiltonian_sim(self, x, seq_lengths):
        x = x.to(device)
        seq_lengths = seq_lengths.to(device)
        
        # Add vector bias
        if self.bias == 'vector':
            batch_bias = [self.bias_param.view(1, -1).repeat(l, 1) for l in seq_lengths]
            batch_bias = nn.utils.rnn.pad_sequence(batch_bias, batch_first=True)
            x += batch_bias
        
        if self.n_classes == 2:
            x = self.measurement_map(x).type(torch.complex64)

            x = torch.einsum('bsp, pij -> bij', x, self.measurements) / seq_lengths.view(-1, 1, 1)

            # Add bias
            if self.bias == 'matrix': # Full matrix
                h0 = self.bias_param.triu() + self.bias_param.triu(1).H
            elif self.bias == 'vector': # Done before
                h0 = torch.zeros_like(x[0])
            elif self.bias == 'diag': # Diagonal matrix
                h0 = torch.diag(self.bias_param.view(-1))
            elif self.bias == 'single': # Constant * Identity
                h0 = self.bias_param * I(self.n_wires)
            elif self.bias == 'none': # No bias
                h0 = torch.zeros_like(x[0])
            else:
                raise ValueError(f'Unknown bias {self.bias}')
            x = x + h0
            x = torch.nn.functional.normalize(x,dim=0)
        else:
            x = [m(x).type(torch.complex64) for m in self.measurement_map]
            x = torch.stack(x, dim=1)
            seq_lengths_inv = 1 / seq_lengths
            x = torch.einsum('bcsp, b -> bcsp', x, seq_lengths_inv)
            x = torch.einsum('bcsp, pij -> bcij', x, self.measurements)

        return x

    def state(self, x):
        x = x.type(torch.complex64).to(device)
        if self.circ_in == 'sentence': # Mean of sentence
            sent = x.mean(dim=1).reshape(-1, self.emb_dim) # (batch_size, emb_dim)
            sent = torch.nn.functional.pad(sent, (0, 2**self.n_wires - self.emb_dim))
            norms = torch.norm(sent, dim=1).view(-1, 1)
            if torch.any(norms <= 0):
                fill_in = torch.zeros_like(sent[(norms <= 0).squeeze()])
                fill_in[:,0] = 1
                sent[(norms <= 0).squeeze()] = fill_in
                norms = torch.norm(sent, dim=1).view(-1, 1)
            sent = sent / norms
            sent = self.circuit(sent)
            return sent
        elif self.circ_in == 'zeros': # Zero state
            sent = torch.zeros(2**self.n_wires, dtype=torch.complex64).to(device)
            sent[0] = 1
            sent = self.circuit(sent.view(1, -1))
            sent = sent.repeat(x.shape[0], 1)
            return sent
        elif self.circ_in == 'hadamard':
            sent = torch.ones(2**self.n_wires, dtype=torch.complex64).to(device)
            sent = sent / torch.norm(sent).view(-1, 1)
            sent = self.circuit(sent.view(1, -1))
            sent = sent.repeat(x.shape[0], 1)
            return sent
        else:
            raise ValueError(f'Unknown circuit input {self.circ_in}')


    def expval(self, ham, state):
        if self.n_classes == 2:
            if len(state.shape) == 1:
                state = state.view(1, -1)
            if len(ham.shape) == 2:
                ham = ham.view(1, *ham.shape)

            ev = torch.einsum('bi,bij,jb -> b', state, ham, state.H).real

            if self.batch_norm:
                ev = self.batch_norm(ev.view(-1, 1)).view(-1)
            ev = nn.functional.sigmoid(ev)
        else:
            if len(state.shape) == 1:
                state = state.view(1, -1)
            ev = torch.einsum('bi,bcij,jb -> bc', state, ham, state.H).real
            if self.batch_norm:
                ev = self.batch_norm(ev)
            ev = nn.functional.softmax(ev, dim=1)
        return ev

    def forward(self, x, seq_lengths):
        '''
        x: (batch_size, sent_len, emb_dim)
        lengths: (batch_size)

        Returns:
        (batch_size), (batch_size, emb_dim)
        '''
        x = x.to(device)
        seq_lengths = seq_lengths.to(device)
        seq_lengths[seq_lengths == 0] = 1

        assert x.shape[2] == self.emb_dim, f'Expected emb_dim {self.emb_dim}, got {x.shape[2]}'

        # If the sentence is too long, truncate it
        if x.shape[1] > self.max_len:
            x = x[:, :self.max_len]
            seq_lengths = torch.minimum(seq_lengths, torch.tensor(self.max_len).to(device))


        state = self.state(x) # Get state/sent emb
        ham = self.hamiltonian(x, seq_lengths) # Get hamiltonian
        x = self.expval(ham, state) # Combine to get expval

        return x, state

    def get_n_params(self):
        if self.bias == 'matrix':
            bias_params2 = self.bias_param.numel() 
            bias_params = int(bias_params2**0.5) * (int(bias_params2**0.5) - 1) // 2
        elif self.bias == 'vector' or self.bias == 'diag':
            bias_params = self.bias_param.numel()
        elif self.bias == 'single':
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

        if self.strategy == 'simplified':
            measurements_params = self.pauli_strings.numel()
        else:
            measurements_params = 0

        if self.batch_norm:
            batch_norm_params = sum(p.numel() for p in self.batch_norm.parameters() if p.requires_grad)
        else:
            batch_norm_params = 0

        circ_params = sum(p.numel() for p in self.circuit.parameters() if p.requires_grad)
        all_params = circ_params + bias_params + pos_enc_params + measurements_params + batch_norm_params
        n_params = {'n_bias_params': bias_params, 'n_circ_params': circ_params, 'n_measurements_params': measurements_params,
                    'n_batch_norm_params': batch_norm_params, 'n_pos_enc_params': pos_enc_params, 'n_all_params': all_params,}
        return n_params
