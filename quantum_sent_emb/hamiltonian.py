import torch
import torch.nn as nn
from quantum_sent_emb import KWArgsMixin, UpdateMixin

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def I(n_wires):
    '''
    Builds the identity matrix of size 2**n_wires
    '''
    return torch.eye(2**n_wires, dtype=torch.complex64).to(device)


def RX(theta):
    '''
    Builds the rotation matrix around X axis
    '''
    theta = theta.view(1, 1)
    cos_theta = torch.cos(theta/2)
    sin_theta = torch.sin(theta/2)
    return torch.cat([torch.cat([cos_theta, -1j * sin_theta], dim=1),
                      torch.cat([-1j * sin_theta, cos_theta], dim=1)], dim=0).to(device)


def X():
    '''
    Builds the Pauli-X gate
    '''
    return torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64).to(device)


def RY(theta):
    '''
    Builds the rotation matrix around Y axis
    '''
    theta = theta.view(1, 1)
    cos_theta = torch.cos(theta/2)
    sin_theta = torch.sin(theta/2)
    rot_y = torch.cat([torch.cat([cos_theta, -sin_theta], dim=1),
                      torch.cat([sin_theta, cos_theta], dim=1)], dim=0).type(torch.complex64)
    return rot_y.to(device)


def RZ(theta):
    '''
    Builds the rotation matrix around Z axis
    '''
    theta = theta.view(1, 1)
    neg_exp = torch.exp(-1j * theta / 2)
    pos_exp = torch.exp(1j * theta / 2)
    torch.zeros_like(theta)
    rot_z = torch.cat([torch.cat([pos_exp, torch.zeros_like(theta)], dim=1),
                       torch.cat([torch.zeros_like(theta), neg_exp], dim=1)], dim=0)
    return rot_z.to(device)


def CZ():
    '''
    Builds the CZ gate
    '''
    CZ = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [
                      0, 0, 0, -1]]).type(torch.complex64)
    return CZ.to(device)


def CNOT():
    '''
    Builds the CNOT gate
    '''
    CNOT = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [
                        0, 0, 1, 0]]).type(torch.complex64)
    return CNOT.to(device)


def distant_CZ(control, target, n_wires):
    '''
    Builds a CZ gate for distant qubits
    '''
    return _distant_control(control, target, n_wires, Z())


def distant_CNOT(control, target, n_wires):
    '''
    Builds a CNOT gate for distant qubits
    '''
    return _distant_control(control, target, n_wires, X())


def _distant_control(control, target, n_wires, gate):
    '''
    Generalizes non-adjacent control gates

    See:
    https://quantumcomputing.stackexchange.com/questions/9180/how-do-i-write-the-matrix-for-a-cz-gate-operating-on-nonadjacent-qubits
    https://quantumcomputing.stackexchange.com/questions/4252/how-to-derive-the-cnot-matrix-for-a-3-qubit-system-where-the-control-target-qu/4254#4254
    '''
    U0 = torch.tensor([1], device=device)
    U1 = torch.tensor([1], device=device)
    assert control < n_wires, f'Expected control < n_wires, got {
        control} and {n_wires}'
    assert target < n_wires, f'Expected target < n_wires, got {
        target} and {n_wires}'
    assert control != target, f'Expected control != target, got {
        control} and {target}'
    for i in range(n_wires):
        if i == control:
            U0 = torch.kron(U0, torch.tensor([[1, 0], [0, 0]], device=device))
            U1 = torch.kron(U1, torch.tensor([[0, 0], [0, 1]], device=device))
        elif i == target:
            U0 = torch.kron(U0, I(1))
            U1 = torch.kron(U1, gate)
        else:
            U0 = torch.kron(U0, I(1))
            U1 = torch.kron(U1, I(1))
    return U0 + U1


def H():
    '''
    Builds the Hadamard gate
    '''
    H = torch.tensor([[1, 1], [1, -1]]).type(torch.complex64) / \
        torch.tensor(2.0).sqrt().type(torch.complex64)
    return H.to(device)


def Z():
    '''
    Builds the Pauli-Z gate
    '''
    Z = torch.tensor([[1, 0], [0, -1]]).type(torch.complex64)
    return Z.to(device)


def layer_gate(n_wires, gate):
    '''
    Builds the gate of size 2**n_wires
    Accepts a single gate or a list of gates
    '''
    if isinstance(gate, list):
        assert len(gate) == n_wires, f'Expected {
            n_wires} gates, got {len(gate)}'
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
    Applies _distant_control to each pair in ring

    If multiple gates are passed, applies each gate to each pair
    If a single gate is passed, applies the same gate to each pair

    Note: different than qml.broadcast for n_wires = 2
    qml only applies one gate, this applies two

    Note: gate should be a 1-qubit gate, not a 2-qubit gate,
    the control adds the second qubit    

    n_wires: number of qubits
    gate: 1-qubit gate or list of 1-qubit gates
    '''
    if not isinstance(gate, list):
        gate = [gate] * n_wires
    else:
        assert len(gate) == n_wires, f'Expected {
            n_wires} gates, got {len(gate)}'

    U = I(n_wires)
    for i in range(n_wires):
        U = _distant_control(i, (i+1) % n_wires, n_wires, gate[i]) @ U

    return U


def a2a_gate(n_wires, gate):
    '''
    Builds the all-to-all gate of size 2**n_wires
    Applies _distant_control to each pair

    If multiple gates are passed, applies each gate to each pair
    If a single gate is passed, applies the same gate to each pair

    Note: different than qml.broadcast
    qml only applies controls A->B, this applies also B->A

    Note: gate should be a 1-qubit gate, not a 2-qubit gate,
    the control adds the second qubit    

    n_wires: number of qubits
    gate: 1-qubit gate or list of 1-qubit gates
    '''
    if not isinstance(gate, list):
        gate = [gate] * n_wires * (n_wires - 1)
    else:
        assert len(gate) == n_wires * (n_wires - 1), \
            f'Expected {n_wires * (n_wires - 1)} gates, got {len(gate)}'

    U = I(n_wires)
    for i in range(n_wires):
        for j in range(n_wires):
            if i != j:
                # Unrolls matrix ignoring diagonal
                w_index = (i * n_wires + j) - \
                    (i * n_wires + j) // (n_wires + 1) - 1
                U = _distant_control(i, j, n_wires, gate[w_index]) @ U

    return U


class RXLayer(nn.Module, UpdateMixin):
    '''
    Rotation around X axis 
    '''

    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires
        self.theta = nn.Parameter(torch.rand((self.n_wires, 1)))
        self.gate = None
        self.update()

    def update(self):
        del self.gate
        gates = [RX(t) for t in self.theta]
        self.gate = layer_gate(self.n_wires, gates).to(device)

    def forward(self, x):
        return self.gate @ x


class RYLayer(nn.Module, UpdateMixin):
    '''
    Rotation around Y axis
    '''

    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires
        self.theta = nn.Parameter(torch.rand((self.n_wires, 1)))
        self.gate = None
        self.update()

    def update(self):
        del self.gate
        gates = [RY(t) for t in self.theta]
        self.gate = layer_gate(self.n_wires, gates).to(device)

    def forward(self, x):
        return self.gate @ x


class RZLayer(nn.Module, UpdateMixin):
    '''
    Rotation around Z axis
    '''

    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires
        self.theta = nn.Parameter(torch.rand((self.n_wires, 1)))
        self.gate = None

    def update(self):
        del self.gate
        gates = [RZ(t) for t in self.theta]
        self.gate = layer_gate(self.n_wires, gates).to(device)

    def forward(self, x):
        return self.gate @ x


class CZRing(nn.Module, UpdateMixin):
    '''
    Ring of CZ gates
    '''

    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires
        self.gate = ring_gate(self.n_wires, Z())

    def forward(self, x):
        return self.gate @ x


class CNOTRing(nn.Module, UpdateMixin):
    '''
    Ring of CNOT gates
    '''

    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires
        self.gate = ring_gate(self.n_wires, X())

    def forward(self, x):
        return self.gate @ x


class CRXRing(nn.Module, UpdateMixin):
    '''
    Ring of CRX gates
    '''

    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires
        self.theta = nn.Parameter(torch.rand((self.n_wires, 1)))
        self.gate = None
        self.update()

    def update(self):
        del self.gate
        gates = [RX(t) for t in self.theta]
        self.gate = ring_gate(self.n_wires, gates)

    def forward(self, x):
        return self.gate @ x


class CRZRing(nn.Module, UpdateMixin):
    '''
    Ring of CRZ gates
    '''

    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires
        self.theta = nn.Parameter(torch.rand((self.n_wires, 1)))
        self.gate = None
        self.update()

    def update(self):
        del self.gate
        gates = [RZ(t) for t in self.theta]
        self.gate = ring_gate(self.n_wires, gates)

    def forward(self, x):
        return self.gate @ x


class CRXAllToAll(nn.Module, UpdateMixin):
    '''
    All-to-all CRX gates
    '''

    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires
        n_params = self.n_wires * (self.n_wires - 1)
        self.theta = nn.Parameter(torch.rand((n_params, 1)))
        self.gate = None
        self.update()

    def update(self):
        del self.gate
        gates = [RX(t) for t in self.theta]
        self.gate = a2a_gate(self.n_wires, gates)

    def forward(self, x):
        return self.gate @ x


class CRZAllToAll(nn.Module, UpdateMixin):
    '''
    All-to-all CRZ gates
    '''

    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires
        n_params = self.n_wires * (self.n_wires - 1)
        self.theta = nn.Parameter(torch.rand((n_params, 1)))
        self.gate = None
        self.update()

    def update(self):
        del self.gate
        gates = [RZ(t) for t in self.theta]
        self.gate = a2a_gate(self.n_wires, gates)

    def forward(self, x):
        return self.gate @ x


class Circuit(nn.Module, UpdateMixin):
    '''
    Simulated quantum circuit
    '''

    def __init__(self, n_wires, gates, n_reps, *args, **kwargs) -> None:
        super().__init__()
        self.n_wires = n_wires
        # 1-qubit gates are applied to each qubit
        # 2-qubit gates are applied in ring configuration
        c2g = {'rx': RXLayer, 'ry': RYLayer, 'rz': RZLayer, 'cz_ring': CZRing,
               'cnot_ring': CNOTRing, 'crx_ring': CRXRing, 'crz_ring': CRZRing,
               'crx_all_to_all': CRXAllToAll, 'crz_all_to_all': CRZAllToAll}
        self.layers = []
        for gate in gates:
            self.layers.append(c2g[gate](n_wires=n_wires))
        self.layers = nn.ModuleList(self.layers * n_reps)

    def forward(self, x):
        x = x.T
        for layer in self.layers:
            x = layer(x)
        return x.T

    def matrix(self):
        '''
        Returns the matrix representation of the circuit
        '''
        x = torch.eye(2**self.n_wires, dtype=torch.complex64).to(device)
        for layer in self.layers:
            x = layer(x)
        return x

    def update(self):
        for layer in self.layers:
            layer.update()


class HamiltonianClassifier(nn.Module, KWArgsMixin, UpdateMixin):
    '''
    Simulated classification based on a quantum Hamiltonian
    '''

    def __init__(self, emb_dim, hamiltonian, *args, **kwargs) -> None:
        '''
        emb_dim: size of the embedding
        hamiltonian: 'pure' or 'mixed'
        '''
        super().__init__()
        self.emb_size = emb_dim
        self.hamiltonian = hamiltonian
        # Next power of 2 of log(emb_size)
        self.n_wires = (emb_dim - 1).bit_length()
        self.circuit = Circuit(n_wires=self.n_wires, *args, **kwargs)
        KWArgsMixin.__init__(self, emb_dim=emb_dim, **kwargs)
        self.update()

    def update(self):
        self.circuit.update()

    def forward(self, x, lengths):
        '''
        x: (batch_size, sent_len, emb_dim)
        lengths: (batch_size)

        Returns:
        (batch_size), (batch_size, emb_dim)
        '''
        x = x.type(torch.complex64)
        s = x.mean(dim=1).reshape(-1, self.emb_size) # (batch_size, emb_dim)
        s = torch.nn.functional.pad(s, (0, 2**self.n_wires - self.emb_size))
        # Normalize s
        s = s / torch.norm(s, dim=1).view(-1, 1)

        # Outer product from (batch_size, sent_len, emb_dim) to (batch_size, emb_size, emb_size)
        if self.hamiltonian == 'pure':
            # This measures pure states
            x = torch.einsum('bsi, bsj -> bij', x, x) / lengths.view(-1, 1, 1)
        elif self.hamiltonian == 'mixed':
            # This measures mixed states
            x = torch.sum(x, dim=1)
            x = torch.einsum('bi,bj -> bij', x, x) / lengths.view(-1, 1, 1)
        else:
            raise ValueError(f'Unknown Hamiltonian {self.hamiltonian}')

        # Pad emb_size to next power of 2 (batch_size, 2**n_wires, 2**n_wires)
        x = torch.nn.functional.pad(
            x, (0, 2**self.n_wires - self.emb_size, 0, 2**self.n_wires - self.emb_size))
        x = torch.nn.functional.normalize(x,dim=0)

        # TODO: remove this
        # import matplotlib.pyplot as plt

        # plt.plot(torch.linalg.eig(x[0]).eigenvalues.real.cpu().numpy(), 'o')
        # plt.plot(torch.linalg.eig(x[1]).eigenvalues.real.cpu().numpy(), 'o')
        # plt.plot(torch.linalg.eig(x[2]).eigenvalues.real.cpu().numpy(), 'o')
        # plt.plot(torch.linalg.eig(x[3]).eigenvalues.real.cpu().numpy(), 'o')
        # plt.show()



        # Apply self.circuit to sentence
        sent_emb = self.circuit(s)
        x = torch.einsum('bi,bij,jb -> b', sent_emb, x, sent_emb.H).real
        x = nn.functional.sigmoid(x)
        return x, sent_emb

    def get_n_params(self):
        all_params = sum(p.numel()
                         for p in self.parameters() if p.requires_grad)
        n_params = {'n_all_params': all_params}
        return n_params


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training test
    emb_dim = 300
    x = torch.rand((256, 10, emb_dim)).type(torch.complex64).to(device)
    lengths = torch.randint(1, 10, (256,)).to(device)
    model = HamiltonianClassifier(emb_dim=emb_dim, gates=[
                                  'rx', 'ry', 'rz'], hamiltonian='pure', n_reps=2)
    model.to(device)
    print(model(x, lengths))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for i in range(2000):
        optimizer.zero_grad()
        output, _ = model(x, lengths)
        loss = torch.mean(output).real
        print(loss)
        loss.backward()
        # Print gradient norm
        print(f'Gradient norm: {torch.norm(torch.stack(
            [torch.norm(p.grad) for p in model.parameters()]))}')
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = param.grad.data.to(param.data.device)
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
