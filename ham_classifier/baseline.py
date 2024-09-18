import torch
import torch.nn as nn
import pennylane as qml
from .circuit import Circuit, device, QLSTMCell
from .utils import KWArgsMixin, UpdateMixin


class BagOfWordsClassifier(nn.Module, KWArgsMixin):
    '''
    Bag of words baseline classifier
    '''
    def __init__(self, emb_dim, n_classes):
        super().__init__()
        self.n_classes = n_classes
        if n_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(emb_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(emb_dim, n_classes),
                nn.Softmax(dim=1)
            )
        KWArgsMixin.__init__(self, emb_dim=emb_dim)
    
    def forward(self, input, seq_lengths):
        seq_lengths = seq_lengths.to(device=input.device)
        seq_lengths[seq_lengths == 0] = 1
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
    def __init__(self, emb_dim, hidden_dim, rnn_layers, architecture, n_classes):
        super().__init__()
        self.n_classes = n_classes
        if architecture == 'rnn':
            self.combiner = nn.RNN(input_size=emb_dim, hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)
        elif architecture == 'lstm':
            self.combiner = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=rnn_layers, batch_first=True)
        else:
            raise ValueError(f'Unknown architecture: {architecture}')
        if n_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, n_classes),
                nn.Softmax(dim=1)
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
                 bias, pos_enc, n_classes,
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
        self.n_classes = n_classes
        self.circuit = Circuit(n_wires=self.n_wires, *args, **kwargs)
        if clas_type == 'tomography':
            if n_classes == 2:
                self.classifier = nn.Sequential(
                    nn.Linear(2**self.n_wires, 1),
                    nn.BatchNorm1d(1),
                    nn.Sigmoid()
                )
            else:
                self.classifier = nn.Sequential(
                    nn.Linear(2**self.n_wires, n_classes),
                    nn.BatchNorm1d(n_classes),
                    nn.Softmax(dim=1)
                )
        elif clas_type == 'hamiltonian':
            if n_classes != 2:
                raise ValueError('Hamiltonian classifier only supports binary classification')
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

        if torch.any(seq_lengths <= 0):
            fill_in = torch.zeros_like(x[seq_lengths <= 0])
            fill_in[:,0] = 1
            x[seq_lengths <= 0] = fill_in
            seq_lengths[seq_lengths <= 0] = 1
        norms = torch.norm(x, dim=1).view(-1, 1)
        x = x / norms

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
    

#self, emb_dim, hidden_dim, n_wires, n_layers, gates, n_classes
class QLSTMClassifier(nn.Module, KWArgsMixin, UpdateMixin):
    '''
    Quantum LSTM classifier
    Heavily inspired by the QLSTM implementation in https://github.com/rdisipio/qlstm
    '''
    def __init__(self, emb_dim, hidden_dim, n_wires, n_layers, gates, n_classes):
        super(QLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        
        # Single LSTM cell
        self.qlstm_cell = QLSTMCell(emb_dim, hidden_dim, n_wires, n_layers, gates)
        
        if n_classes == 2:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, n_classes),
                nn.Softmax(dim=1)
            )        
        KWArgsMixin.__init__(self, emb_dim=emb_dim, hidden_dim=hidden_dim, n_wires=n_wires, n_layers=n_layers, gates=gates)
    
    def forward(self, x, seq_lengths):
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden state and cell state to zeros
        hidden = (torch.zeros(batch_size, self.hidden_dim).to(x.device),
                    torch.zeros(batch_size, self.hidden_dim).to(x.device))
        
        outputs = []
        for t in range(seq_len):
            input_t = x[:, t, :]  # Get input at time step t
            h, C = self.qlstm_cell(input_t, hidden)  # Get the new hidden state and cell state
            hidden = (h, C)  # Update hidden and cell states
            outputs.append(h.unsqueeze(1))  # Collect the output for each time step
        
        # Concatenate outputs for all time steps
        outputs = torch.cat(outputs, dim=1)
        
        # # Use the output of the last time step for classification
        # final_output = outputs[:, -1, :]  # Take the output of the last time step
        
        # # Pass the final output through the classifier
        # normalized_output = self.classifier(final_output)
        
        # Use the last valid output for classification
        seq_lengths = seq_lengths.to(device=x.device)
        seq_lengths[seq_lengths == 0] = 1
        last_valid_indices = seq_lengths - 1
        last_valid_indices = last_valid_indices.view(-1, 1).expand(-1, self.hidden_dim)
        last_valid_indices = last_valid_indices.unsqueeze(1)
        normalized_output = torch.gather(outputs, 1, last_valid_indices).squeeze()
        normalized_output = self.classifier(normalized_output)

        return normalized_output.squeeze(), hidden
    
    def update(self):
        for gate in self.qlstm_cell.VQC.values():
            gate.update()

    def get_n_params(self):
        qlstm_params = self.qlstm_cell.get_n_params() 
        n_qlstm_params = sum(p.numel() for p in self.qlstm_cell.parameters())
        n_classifier_params = sum(p.numel() for p in self.classifier.parameters())
        n_all_params = sum(p.numel() for p in self.parameters())
        n_params = {'n_qlstm_params': n_qlstm_params, 'n_classifier_params': n_classifier_params, 
                    'n_all_params': n_all_params}
        qlstm_params.update(n_params)
        return qlstm_params


class MLPClassifier(nn.Module, KWArgsMixin):
    '''
    Multi-layer perceptron classifier
    '''
    def __init__(self, emb_dim, n_layers, hidden_dim, n_classes):
        super().__init__()
        self.n_classes = n_classes
        if n_layers < 2:
            raise ValueError('n_layers must be at least 2')
        self.classifier = nn.Sequential()
        self.classifier.add_module('input', nn.Linear(emb_dim, hidden_dim))
        self.classifier.add_module('relu_in', nn.ReLU())
        for i in range(n_layers-2):
            self.classifier.add_module(f'hidden_{i}', nn.Linear(hidden_dim, hidden_dim))
            self.classifier.add_module(f'relu_{i}', nn.ReLU())
        if n_classes == 2:
            self.classifier.add_module('output', nn.Linear(hidden_dim, 1))
            self.classifier.add_module('sigmoid_out', nn.Sigmoid())
        else:
            self.classifier.add_module('output', nn.Linear(hidden_dim, n_classes))
            self.classifier.add_module('softmax_out', nn.Softmax(dim=1))

        KWArgsMixin.__init__(self, emb_dim=emb_dim, n_layers=n_layers, hidden_dim=hidden_dim)
    
    def forward(self, input, seq_lengths):
        seq_lengths = seq_lengths.to(device=input.device)
        seq_lengths[seq_lengths == 0] = 1
        sent_emb = torch.sum(input, dim=1) / seq_lengths.view(-1, 1)
        pred = self.classifier(sent_emb).squeeze() # Squeeze to remove last dimension in binary classification tasks

        return pred, sent_emb
    
    def get_n_params(self):
        n_all_params = sum(p.numel() for p in self.parameters())
        n_params = {'n_all_params': n_all_params}
        return n_params
    

class CNNClassifier(nn.Module, KWArgsMixin):
    '''
    Convolutional Neural Network classifier
    '''
    def __init__(self, in_channels, n_layers, emb_dim, out_channels, n_classes, kernel_size=3):
        super().__init__()
        self.n_classes = n_classes
        if n_layers < 2:
            raise ValueError('n_layers must be at least 2')
        
        self.features = nn.Sequential()
        
        # Add the first convolutional layer
        self.features.add_module('conv1', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1))
        self.features.add_module('relu1', nn.ReLU())
        # self.features.add_module('pool1', nn.MaxPool2d(pool_size))
        
        # Add hidden convolutional layers
        for i in range(n_layers - 2):
            self.features.add_module(f'conv_hidden_{i}', nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1))
            self.features.add_module(f'relu_hidden_{i}', nn.ReLU())
            # self.features.add_module(f'pool_hidden_{i}', nn.MaxPool2d(pool_size))
        
        # Flatten the output from the convolutional layers for fully connected layers
        self.flatten = nn.Flatten()
        flatten_dim = self._get_flatten_dim(in_channels, emb_dim)

        # Add final fully connected layers
        self.classifier = nn.Sequential()
        if n_classes == 2:
            self.classifier.add_module('fc', nn.Linear(flatten_dim, 1))
            self.classifier.add_module('sigmoid_out', nn.Sigmoid())
        else:
            self.classifier.add_module('fc', nn.Linear(flatten_dim, n_classes))
            self.classifier.add_module('softmax_out', nn.Softmax(dim=1))
        KWArgsMixin.__init__(self, in_channels=in_channels, n_layers=n_layers, out_channels=out_channels, 
                                n_classes=n_classes, kernel_size=kernel_size)

    def _get_flatten_dim(self, in_channels, emb_dim):
        # Create a dummy input tensor with batch size = 1 and in_channels, and a standard size (e.g., 32x32)
        dummy_input = torch.randn(1, in_channels, emb_dim, emb_dim)
        
        # Pass through the convolutional layers to get the output shape
        conv_out = self.features(dummy_input)
        
        # Flatten the output and get the number of features
        flatten_dim = conv_out.view(1, -1).size(1)
        
        return flatten_dim

    def forward(self, input, seq_lengths):
        # Pass through the convolutional layers
        conv_out = self.features(input)
        # Flatten the feature map
        flattened = self.flatten(conv_out)
        # Pass through the fully connected classifier
        pred = self.classifier(flattened).squeeze()

        return pred, flattened

    def get_n_params(self):
        # Count the number of parameters in the convolutional layers
        conv_params = sum(p.numel() for p in self.features.parameters())
        
        # Count the number of parameters in the fully connected classifier layers
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        # Total number of parameters
        n_all_params = conv_params + classifier_params
        
        # Return as a dictionary with breakdown
        n_params = {
            'n_conv_params': conv_params,
            'n_classifier_params': classifier_params,
            'n_all_params': n_all_params
        }
        
        return n_params


class QCNNClassifier(torch.nn.Module, KWArgsMixin):
    '''
    Quantum Convolutional Neural Network Classifier.
    Inspired by the QCNN implementation in https://pennylane.ai/qml/demos/tutorial_learning_few_data/
    '''

    def __init__(self, n_wires, n_layers, ent_layers, n_classes):
        super().__init__()
        self.n_wires = n_wires
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.ent_layers = ent_layers
        
        weight_shapes, final_wires = self._qcnn_sizes(n_wires, n_layers, ent_layers)
        self.qml_device = qml.device("default.qubit", wires=n_wires)
        
        @qml.qnode(self.qml_device, interface='torch')
        def conv_net(inputs, weights, last_layer_weights):
            """Define the QCNN circuit
            Args:
                weights (np.array): Parameters of the convolution and pool layers.
                last_layer_weights (np.array): Parameters of the last dense layer.
                features (np.array): Input data to be embedded using AmplitudEmbedding."""

            layers = weights.shape[1]
            wires = list(range(n_wires))

            # inputs the state input_state
            qml.AmplitudeEmbedding(features=inputs, wires=wires, pad_with=0.5, normalize=True)

            # adds convolutional and pooling layers
            for j in range(layers):
                self._conv_and_pooling(weights[:, j], wires, skip_first_layer=(not j == 0))
                wires = wires[::2]

            self._dense_layer(last_layer_weights, wires)

            # Perform expval on the remaining wires
            return [qml.expval(qml.PauliZ(w)) for w in final_wires]
        
        self.conv_net = qml.qnn.TorchLayer(conv_net, weight_shapes)
        if n_classes == 2:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(len(final_wires), 1),
                torch.nn.Sigmoid()
            )
        elif n_classes > 2:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(len(final_wires), n_classes),
                torch.nn.Softmax(dim=-1)
            )
        else:
            raise ValueError("The number of classes must be greater than 1!")

        KWArgsMixin.__init__(self, n_wires=n_wires, n_layers=n_layers, ent_layers=ent_layers, n_classes=n_classes)

    def _qcnn_sizes(self, n_wires, n_layers, ent_layers):
        wires = list(range(n_wires))
        for i in range(n_layers):
            wires = wires[::2]

        n_final_wires = len(wires)
        assert n_final_wires > 1, "The number of final wires is too low!"
        return {"weights": (18, n_layers), "last_layer_weights": (ent_layers, n_final_wires, 3)}, wires

    def _convolutional_layer(self, weights, wires, skip_first_layer=True):
        """Adds a convolutional layer to a circuit.
        Args:
            weights (np.array): 1D array with 15 weights of the parametrized gates.
            wires (list[int]): Wires where the convolutional layer acts on.
            skip_first_layer (bool): Skips the first two U3 gates of a layer.
        """
        n_wires = len(wires)
        assert n_wires >= 3, "this circuit is too small!"

        for p in [0, 1]:
            for indx, w in enumerate(wires):
                if indx % 2 == p and indx < n_wires - 1:
                    if indx % 2 == 0 and not skip_first_layer:
                        qml.U3(*weights[:3], wires=[w])
                        qml.U3(*weights[3:6], wires=[wires[indx + 1]])
                    qml.IsingXX(weights[6], wires=[w, wires[indx + 1]])
                    qml.IsingYY(weights[7], wires=[w, wires[indx + 1]])
                    qml.IsingZZ(weights[8], wires=[w, wires[indx + 1]])
                    qml.U3(*weights[9:12], wires=[w])
                    qml.U3(*weights[12:], wires=[wires[indx + 1]])


    def _pooling_layer(self, weights, wires):
        """Adds a pooling layer to a circuit.
        Args:
            weights (np.array): Array with the weights of the conditional U3 gate.
            wires (list[int]): List of wires to apply the pooling layer on.
        """
        n_wires = len(wires)
        assert len(wires) >= 2, "this circuit is too small!"

        for indx, w in enumerate(wires):
            if indx % 2 == 1 and indx < n_wires:
                m_outcome = qml.measure(w)
                qml.cond(m_outcome, qml.U3)(*weights, wires=wires[indx - 1])


    def _conv_and_pooling(self, kernel_weights, n_wires, skip_first_layer=True):
        """Apply both the convolutional and pooling layer."""
        self._convolutional_layer(kernel_weights[:15], n_wires, skip_first_layer=skip_first_layer)
        self._pooling_layer(kernel_weights[15:], n_wires)


    def _dense_layer(self, weights, wires):
        """Apply an arbitrary unitary gate to a specified set of wires."""
        qml.StronglyEntanglingLayers(weights, wires)

    def forward(self, input, _):
        input = input.squeeze()
        assert len(input) <= 2 ** self.n_wires, "The input is too large for the number of wires!"
        output = self.conv_net(input)
        return self.classifier(output).squeeze(), output
    
    def get_n_params(self):
        _, final_wires = self._qcnn_sizes(self.n_wires, self.n_layers, self.ent_layers)
        conv_params = 15 * self.n_layers
        pool_params = 3 * self.n_layers
        dense_params = 3 * self.ent_layers * len(final_wires)
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        n_all_params = conv_params + pool_params + dense_params + classifier_params
        return {"n_conv_params": conv_params, "n_pool_params": pool_params,
                "n_dense_params": dense_params, "n_classifier_params": classifier_params,
                "n_all_params": n_all_params}