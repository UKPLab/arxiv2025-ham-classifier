import torch
import torch.nn as nn
from .utils import KWArgsMixin, UpdateMixin
from .circuit import Circuit, I, device

class HamiltonianClassifier(nn.Module, KWArgsMixin, UpdateMixin):
    '''
    Simulated classification based on a quantum Hamiltonian
    '''

    def __init__(self, emb_dim, circ_in, 
                 bias, pos_enc, batch_norm,
                 max_len=300, *args, **kwargs) -> None:
        '''
        emb_dim: size of the embedding
        hamiltonian: 'pure' or 'mixed'
        circ_in: 'sentence' or 'zeros'
        bias: 'matrix', 'vector', 'diag', 'single' or None
        pos_enc: 'learned' or None
        batch_norm: bool
        max_len: maximum sentence length
        '''
        super().__init__()
        self.emb_size = emb_dim
        self.circ_in = circ_in
        self.bias = bias
        self.max_len = max_len
        self.pos_enc = pos_enc
        self.n_wires = (emb_dim - 1).bit_length() # Next power of 2 of log(emb_size)
        self.circuit = Circuit(n_wires=self.n_wires, *args, **kwargs)
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(1)
        else:
            self.batch_norm = None
        
        if bias == 'matrix':
            self.bias_param = nn.Parameter(torch.rand((2**self.n_wires, 2**self.n_wires)), )
        if bias == 'vector':
            self.bias_param = nn.Parameter(torch.rand((emb_dim, 1)), )
        elif bias == 'diag':
            self.bias_param = nn.Parameter(torch.rand((2**self.n_wires, 1)), )
        elif bias == 'single':
            self.bias_param = nn.Parameter(torch.rand((1, 1)), )
        if pos_enc == 'learned':
            self.pos_param = nn.Parameter(torch.rand((max_len))).type(torch.complex64)

        KWArgsMixin.__init__(self, emb_dim=emb_dim, circ_in=circ_in, 
                 bias=bias, pos_enc=pos_enc, batch_norm=batch_norm,
                 max_len=max_len, **kwargs)
        self.update()

    def update(self):
        self.circuit.update()

    def to(self, device):
        super().to(device)
        if self.batch_norm:
            self.batch_norm = self.batch_norm.to(device)
        if self.bias == 'matrix' or self.bias == 'vector':
            self.bias_param = self.bias_param.to(device)
        if self.pos_enc == 'learned':
            self.pos_param = self.pos_param.to(device)
        return self

    def forward(self, x, seq_lengths):
        '''
        x: (batch_size, sent_len, emb_dim)
        lengths: (batch_size)

        Returns:
        (batch_size), (batch_size, emb_dim)
        '''
        x = x.type(torch.complex64)
        seq_lengths = seq_lengths.to(device)
        
        if self.circ_in == 'sentence': # Mean of sentence
            s = x.mean(dim=1).reshape(-1, self.emb_size) # (batch_size, emb_dim)
            s = torch.nn.functional.pad(s, (0, 2**self.n_wires - self.emb_size))
            s = s / torch.norm(s, dim=1).view(-1, 1)
        elif self.circ_in == 'zeros': # Zero state
            s = torch.zeros((x.shape[0], 2**self.n_wires), dtype=torch.complex64).to(device)
            s[:, 0] = 1
        else:
            raise ValueError(f'Unknown circuit input {self.circ_in}')
        
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
        elif self.pos_enc == None:
            x = torch.einsum('bsi, bsj -> bij', x, x) / seq_lengths.view(-1, 1, 1)
        else:
            raise ValueError(f'Unknown positional encoding {self.pos_enc}')

        # Pad emb_size to next power of 2 (batch_size, 2**n_wires, 2**n_wires)
        x = torch.nn.functional.pad(
            x, (0, 2**self.n_wires - self.emb_size, 0, 2**self.n_wires - self.emb_size))

        # Add bias
        if self.bias == 'matrix': # Full matrix
            h0 = self.bias_param.triu() + self.bias_param.triu(1).H
        elif self.bias == 'vector': # Done before
            h0 = torch.zeros_like(x[0])
        elif self.bias == 'diag': # Diagonal matrix
            h0 = torch.diag(self.bias_param.view(-1))
        elif self.bias == 'single': # Constant * Identity
            h0 = self.bias_param * I(self.n_wires)
        elif self.bias == None: # No bias
            h0 = torch.zeros_like(x[0])
        else:
            raise ValueError(f'Unknown bias {self.bias}')
        x = x + h0
        x = torch.nn.functional.normalize(x,dim=0)

        # Apply self.circuit to sentence
        circ_out = self.circuit(s)
        x = torch.einsum('bi,bij,jb -> b', circ_out, x, circ_out.H).real

        if self.batch_norm:
            x = self.batch_norm(x.view(-1, 1)).view(-1)
        x = nn.functional.sigmoid(x)

        return x, circ_out

    def get_n_params(self):
        if self.bias == 'matrix':
            bias_params2 = self.bias_param.numel() 
            bias_params = int(bias_params2**0.5) * (int(bias_params2**0.5) - 1) // 2
        elif self.bias == 'vector' or self.bias == 'diag':
            bias_params = self.bias_param.numel()
        elif self.bias == 'single':
            bias_params = self.bias_param.numel()
        else:
            raise ValueError(f'Unknown bias {self.bias}')

        if self.pos_enc == 'learned':
            pos_enc_params = self.pos_param.numel()
        elif self.pos_enc == None:
            pos_enc_params = 0
        else:
            raise ValueError(f'Unknown positional encoding {self.pos_enc}')

        if self.batch_norm:
            batch_norm_params = sum(p.numel() for p in self.batch_norm.parameters() if p.requires_grad)
        else:
            batch_norm_params = 0

        circ_params = sum(p.numel() for p in self.circuit.parameters() if p.requires_grad)
        all_params = circ_params + bias_params + pos_enc_params + batch_norm_params
        n_params = {'n_bias_params': bias_params, 'n_circ_params': circ_params,
                    'n_batch_norm_params': batch_norm_params, 'n_pos_enc_params': pos_enc_params, 'n_all_params': all_params,}
        return n_params


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training test
    emb_dim = 300
    x = torch.rand((256, 10, emb_dim)).type(torch.complex64).to(device)
    lengths = torch.randint(1, 10, (256,)).to(device)
    model = HamiltonianClassifier(emb_dim=emb_dim, gates=[
                                  'rx', 'ry', 'rz'], circ_in='zeros', 
                                    batch_norm=True,
                                  pos_enc='learned', bias='single', n_reps=1)
    model.to(device)
    print(model(x, lengths))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for i in tqdm(range(2000)):
        optimizer.zero_grad()
        output, _ = model(x, lengths)
        loss = torch.mean(output).real
        print(loss)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = param.grad.data.to(param.data.device)
        # Print gradient norm
        print(f'Gradient norm: {torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters()]))}')
        optimizer.step()
        model.update()

    # Layer matrix test
    # emb_dim = 5
    # device = 'cpu'
    # circ = Circuit(n_wires=emb_dim, gates=['crx_all_to_all'], n_reps=1)
    # print(circ.matrix())

    # import pennylane as qml

    # dev = qml.device("default.qubit", wires=emb_dim)

    # @qml.qnode(dev)
    # def circuit():
    #     # qml.broadcast(qml.CNOT, wires=range(emb_dim), pattern="all_to_all")
    #     # qml.broadcast(qml.RY, wires=range(emb_dim), pattern="single", parameters=circ.layers[1].theta)
    #     # qml.broadcast(qml.CZ, wires=range(emb_dim), pattern="ring")
    #     # qml.broadcast(qml.CNOT, wires=range(emb_dim), pattern="ring")
    #     wdiff = []
    #     for i in range(emb_dim):
    #         wdiff += [0] + [-(i+1)]*emb_dim
    #     wdiff += [0]
    #     for i in range(emb_dim):
    #         for j in range(emb_dim):
    #             if i != j:
    #                 w_index = (i * emb_dim + j) - \
    #                     (i * emb_dim + j) // (emb_dim + 1) - 1
    #                 qml.CRX(circ.layers[0].theta[w_index], wires=[i, j])

    #     return qml.state()
    # print(qml.matrix(circuit)())
    # qml.draw_mpl(circuit)()
    # plt.show()
