import torch
import torch.nn as nn

class HamiltonianClassifier(nn.Module):
    '''
    Simulated classification based on a quantum Hamiltonian
    '''
    def __init__(self, emb_dim, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.emb_size = emb_dim
        # Next power of 2 of log(emb_size)
        self.n_wires = (emb_dim - 1).bit_length()
        self.circuit = Circuit(self.n_wires)

    def forward(self, x):
        '''
        x: (batch_size, sent_len, emb_dim)

        Returns:
        (batch_size)
        '''
        x = x.type(torch.complex64)
        s = x.mean(dim=1).reshape(-1, self.emb_size)
        s = torch.nn.functional.pad(s, (0, 2**self.n_wires - self.emb_size))
        # Outer product from (batch_size, sent_len, emb_dim) to (batch_size, emb_size, emb_size)
        x = torch.einsum('bsi, bsj -> bij', x, x) / x.shape[1]
        # Pad emb_size to next power of 2 (batch_size, 2**n_wires, 2**n_wires)
        x = torch.nn.functional.pad(x, (0, 2**self.n_wires - self.emb_size, 0, 2**self.n_wires - self.emb_size))
        # Apply self.circuit to sentence
        sent_emb = self.circuit(s)
        x = torch.einsum('bi,bij,jb -> b', sent_emb, x, sent_emb.H).real
        x = torch.clamp(x, min=0, max=1) # Avoids numerical errors
        # Assert everything is between 0 and 1
        assert (x >= 0).all() and (x <= 1).all(), f'Expected x between 0 and 1, got {x}'
        return x, sent_emb

    def get_n_params(self):
        all_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        n_params = {'n_all_params': all_params}
        return n_params
    
def identity(n_wires):
    '''
    Builds the identity matrix of size 2**n_wires
    '''
    return torch.eye(2**n_wires, dtype=torch.complex64).to('cuda')

def rot_x(theta):
    '''
    Builds the rotation matrix around X axis
    '''
    theta = theta.view(1,1)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    return torch.cat([torch.cat([cos_theta, -1j * sin_theta], dim=1),
                      torch.cat([-1j * sin_theta, cos_theta], dim=1)], dim=0)

def x():
    '''
    Builds the Pauli-X gate
    '''
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64)

def rot_y(theta):
    '''
    Builds the rotation matrix around Y axis
    '''
    theta = theta.view(1,1)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    rot_y = torch.cat([torch.cat([cos_theta, -sin_theta], dim=1),
                      torch.cat([sin_theta, cos_theta], dim=1)], dim=0).type(torch.complex64)
    return rot_y

def cz():
    '''
    Builds the CZ gate
    '''
    CZ = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]).type(torch.complex64)
    return CZ.to('cuda')

def distant_cz(control, target, n_wires):
    '''
    Builds a CZ gate for distant qubits
    Only works for first and last qubit
    '''
    assert control < n_wires, f'Expected control < n_wires, got {control} and {n_wires}'
    assert target < n_wires, f'Expected target < n_wires, got {target} and {n_wires}'
    U = identity(n_wires)
    ketbra1 = torch.tensor([[0, 0], [0, 1]]).type(torch.complex64).to('cuda')
    sub = 2 * ketbra1
    for _ in range(control+1, target):
        sub = torch.kron(sub, identity(1))
    sub = torch.kron(sub, ketbra1)
    return (U - sub)


def hadamard():
    '''
    Builds the Hadamard gate
    '''
    H = torch.tensor([[1, 1], [1, -1]]).type(torch.complex64) / torch.tensor(2.0).sqrt().type(torch.complex64)
    return H

def layer_gate(n_wires, gate):
    '''
    Builds the gate of size 2**n_wires
    Accepts a single gate or a list of gates
    '''
    if isinstance(gate, list):
        assert len(gate) == n_wires, f'Expected {n_wires} gates, got {len(gate)}'
        U = gate[0]
        for g in gate[1:]:
            U = torch.kron(U, g)
    else:
        U = gate
        for _ in range(n_wires - 1):
            U = torch.kron(U, gate)
    return U.type(torch.complex64)
    
def ring_gate(n_wires, gate):
    '''
    Builds the ring of gates of size 2**n_wires
    gate is assumed to be 2 wire gate
    '''
    assert n_wires > 2, f'Expected n_wires > 2, got {n_wires}'
    U = torch.kron(gate, identity(n_wires - 2))
    for i in range(1, n_wires - 2 + 1):
        tmp = torch.kron(identity(i), gate)
        U = torch.kron(tmp, identity(n_wires-2-i)) @ U
    return U.type(torch.complex64)

class RXLayer(nn.Module):
    '''
    Rotation around X axis 
    '''
    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires
        self.theta = nn.Parameter(torch.rand((self.n_wires, 1)))

    def forward(self, x):
        gates = [rot_x(t) for t in self.theta]
        return layer_gate(self.n_wires, gates) @ x
    
class RYLayer(nn.Module):
    '''
    Rotation around Y axis
    '''
    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires
        self.theta = nn.Parameter(torch.rand((self.n_wires, 1)))

    def forward(self, x):
        gates = [rot_y(t) for t in self.theta]
        return layer_gate(self.n_wires, gates) @ x
    
class CZRing(nn.Module):
    '''
    Ring of CZ gates
    '''
    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires

    def forward(self, x):
        x = ring_gate(self.n_wires, cz()) @ x
        x = distant_cz(0, self.n_wires - 1, self.n_wires) @ x
        return x
    
class Circuit(nn.Module):
    '''
    Simulated quantum circuit
    '''
    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires
        self.layers = nn.ModuleList([
            RXLayer(self.n_wires), 
            RYLayer(self.n_wires), 
            CZRing(self.n_wires)]*3)
    
    def forward(self, x):
        x = x.T
        for layer in self.layers:
            x = layer(x)
        return x.T


if __name__ == '__main__':
    emb_dim = 300
    x = torch.rand((256, 10, emb_dim)).type(torch.complex64).to('cuda')
    model = HamiltonianClassifier(emb_dim=emb_dim).to('cuda')
    print(model(x))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for i in range(2000):
        optimizer.zero_grad()
        output, _ = model(x)
        loss = torch.mean((output - 1) ** 2).real
        print(loss)
        loss.backward()
        optimizer.step()