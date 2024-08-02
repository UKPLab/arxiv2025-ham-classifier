import concurrent.futures

import torch
import torch.nn as nn
from qiskit.quantum_info import SparsePauliOp
from tqdm import tqdm

from .utils import UpdateMixin

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# def pauli_decompose(matrix):
#     dec = SparsePauliOp.from_operator(matrix)
#     return dec


def decompose_hamiltonians(hamiltonians, sorting=True):
    if hamiltonians[0].shape[0] < 64:
        # decs = [pauli_decompose(h) for h in hamiltonians]
        decs = [SparsePauliOp.from_operator(h) for h in tqdm(hamiltonians)]
    else:
        # Multiprocessing
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # decs = list(tqdm(executor.map(pauli_decompose, hamiltonians), total=len(hamiltonians)))
            decs = list(tqdm(executor.map(SparsePauliOp.from_operator, hamiltonians), total=len(hamiltonians)))

    decs = [d.to_list() for d in decs]
    # Sort based on coefficients
    if sorting:
        decs = [sorted(d, key=lambda x: abs(x[1]), reverse=True) for d in decs]
    return decs


def pauli2matrix(string_list):
    if len(string_list) == 0:
        return torch.eye(1).type(torch.complex64).to(device)
    matrix = SparsePauliOp.from_list(string_list).to_matrix()
    matrix = torch.tensor(matrix).type(torch.complex64)
    return matrix


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


def Y():
    '''
    Builds the Pauli-Y gate
    '''
    return torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64).to(device)


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
    assert control < n_wires, f'Expected control < n_wires, got {control} and {n_wires}'
    assert target < n_wires, f'Expected target < n_wires, got {target} and {n_wires}'
    assert control != target, f'Expected control != target, got {control} and {target}'
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

def pauli_Z_observable(n_wires, target):
    '''
    Builds the Pauli-Z observable
    '''
    U = torch.tensor([1], device=device)
    for i in range(n_wires):
        if i == target:
            U = torch.kron(U, Z())
        else:
            U = torch.kron(U, I(1))
    return U.to(device)


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
        assert len(gate) == n_wires, f'Expected {n_wires} gates, got {len(gate)}'

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


def angle_embedding(x, n_wires):
    '''
    Creates an angle embedded state
    Returns the state vector
    '''
    assert x.shape[1] == n_wires, f'Expected {n_wires} features, got {x.shape[1]}'
    cosines = torch.cos(x)
    sines = -1j * torch.sin(x)
    
    out = torch.ones(x.shape[0], 1, dtype=torch.complex64).to(device) 
    for i in range(x.shape[1]):
        angle = torch.stack([cosines[:, i], sines[:, i]], dim=1)
        out = torch.stack([torch.kron(o, a) for o,a in zip(out, angle)], dim=0)
        
    assert out.shape[1] == 2**n_wires, f'Expected 2**{n_wires} features, got {out.shape[1]}'
    return out 

class ILayer(nn.Module, UpdateMixin):
    '''
    Identity layer
    '''

    def __init__(self, n_wires, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_wires = n_wires
        self.gate = I(self.n_wires)

    def forward(self, x):
        return self.gate @ x

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
        c2g = {'i': ILayer, 'rx': RXLayer, 'ry': RYLayer, 'rz': RZLayer, 'cz_ring': CZRing,
               'cnot_ring': CNOTRing, 'crx_ring': CRXRing, 'crz_ring': CRZRing,
               'crx_all_to_all': CRXAllToAll, 'crz_all_to_all': CRZAllToAll}
        self.layers = []
        for gate in gates:
            self.layers.append(c2g[gate](n_wires=n_wires))
        self.layers = nn.ModuleList(self.layers * n_reps)
        self.update()

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


class PauliCircuit(Circuit):
    '''
    Simulates a circuit with Pauli measurements
    '''
    def __init__(self, n_wires, gates, n_reps, pauli, *args, **kwargs) -> None:
        self.pauli = pauli
        self.n_wires = n_wires
        super().__init__(n_wires, gates, n_reps, *args, **kwargs)
        
        if pauli == 'z':
            self.paulis = [pauli_Z_observable(n_wires, i) for i in range(n_wires)]
            self.paulis = torch.stack(self.paulis)
        

    def forward(self, x):
        x = super().forward(x)
        x = x.T
        return torch.einsum('bi,wij,jb->bw', x.H, self.paulis, x).real
        # (batch, 2**wires) x (wires, 2**wires, 2**wires) x (2**wires, batch) = (batch, wires)

    def update(self):
        super().update()
        if self.pauli == 'z':
            self.paulis = [pauli_Z_observable(self.n_wires, i) for i in range(self.n_wires)]
            self.paulis = torch.stack(self.paulis)


class QLSTM(nn.Module, UpdateMixin):
    '''
    Quantum LSTM as implemented in https://github.com/rdisipio/qlstm
    Uses internally defined PauliCircuit instead of PennyLane
    '''
    def __init__(self, 
                input_size, 
                hidden_size, 
                n_qubits=4,
                n_qlayers=1,
                gates=['rx', 'ry', 'rz'],
                batch_first=True,
                return_sequences=False, 
                return_state=False,
                backend="default.qubit.torch"):
        super(QLSTM, self).__init__()
        self.n_inputs = input_size
        self.hidden_size = hidden_size
        self.concat_size = self.n_inputs + self.hidden_size
        self.n_qubits = n_qubits
        self.n_qlayers = n_qlayers
        self.backend = backend  # "default.qubit", "qiskit.basicaer", "qiskit.ibm"
        self.gates = gates
        self.batch_first = batch_first
        self.return_sequences = return_sequences
        self.return_state = return_state

        self.clayer_in = torch.nn.Linear(self.concat_size, n_qubits)
        self.VQC = {
            'forget': PauliCircuit(n_wires=n_qubits, gates=self.gates, 
                                                 n_reps=n_qlayers, pauli='z'),
            'input': PauliCircuit(n_wires=n_qubits, gates=self.gates, 
                                                 n_reps=n_qlayers, pauli='z'),
            'update': PauliCircuit(n_wires=n_qubits, gates=self.gates, 
                                                 n_reps=n_qlayers, pauli='z'),
            'output': PauliCircuit(n_wires=n_qubits, gates=self.gates, 
                                                 n_reps=n_qlayers, pauli='z'),
        }

        self.clayer_out = torch.nn.Linear(self.n_qubits, self.hidden_size)
        self.update()

    def forward(self, x, seq_lengths, init_states=None):
        '''
        x.shape is (batch_size, seq_length, feature_size)
        recurrent_activation -> sigmoid
        activation -> tanh
        '''
        if self.batch_first is True:
            batch_size, max_seq_length, features_size = x.size()
        else:
            max_seq_length, batch_size, features_size = x.size()

        hidden_seq = []
        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(device)  # hidden state (output)
            c_t = torch.zeros(batch_size, self.hidden_size).to(device)  # cell state
        else:
            # for now we ignore the fact that in PyTorch you can stack multiple RNNs
            # so we take only the first elements of the init_states tuple init_states[0][0], init_states[1][0]
            h_t, c_t = init_states
            h_t = h_t[0]
            c_t = c_t[0]

        for t in range(max_seq_length):
            # get features from the t-th element in seq, for all entries in the batch
            x_t = x[:, t, :]
            
            # Concatenate input and hidden state
            v_t = torch.cat((h_t, x_t), dim=1)

            # match qubit dimension
            y_t = angle_embedding(self.clayer_in(v_t), n_wires=self.n_qubits)

            f_t = torch.sigmoid(self.clayer_out(self.VQC['forget'](y_t)))  # forget block
            i_t = torch.sigmoid(self.clayer_out(self.VQC['input'](y_t)))  # input block
            g_t = torch.tanh(self.clayer_out(self.VQC['update'](y_t)))  # update block
            o_t = torch.sigmoid(self.clayer_out(self.VQC['output'](y_t))) # output block

            c_t = (f_t * c_t) + (i_t * g_t)
            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()

        # Calculate the indices of the last valid elements
        last_valid_indices = (seq_lengths - 1).to(device=device)

        # Use torch.gather to extract the last valid elements
        last_valid_elements = torch.gather(hidden_seq, 1, last_valid_indices.view(-1, 1, 1).expand(-1, 1, hidden_seq.size(2)))
        qlstm_output = last_valid_elements.view(hidden_seq.size(0), hidden_seq.size(2))

        assert last_valid_elements.size(0) == input.size(0), 'Batch size mismatch'
        
                
        # return hidden_seq, (h_t, c_t)

    def to(self, device):
        super().to(device)
        for vqc in self.VQC.values():
            vqc.to(device)
        return self

    def update(self):
        for vqc in self.VQC.values():
            vqc.update()
    
    def get_n_params(self):
        n_all_params = sum(p.numel() for p in self.parameters())
        n_params_forget = sum(p.numel() for p in self.VQC['forget'].parameters())
        n_params_input = sum(p.numel() for p in self.VQC['input'].parameters())
        n_params_update = sum(p.numel() for p in self.VQC['update'].parameters())
        n_params_output = sum(p.numel() for p in self.VQC['output'].parameters())
        n_params = {'n_all_params': n_all_params, 'n_params_forget': n_params_forget, 
                    'n_params_input': n_params_input, 'n_params_update': n_params_update, 
                    'n_params_output': n_params_output}
        return n_params