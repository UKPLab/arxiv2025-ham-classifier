import torch
import torch.nn as nn
from .utils import KWArgsMixin

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
        KWArgsMixin.__init__(self, in_dim=emb_dim, emb_dim=hidden_dim, rnn_num_layers=rnn_layers)
    
    def forward(self, input, seq_lengths):
        # (batch_size, max_seq_len, in_dim), (batch_size)
        assert input.size(0) == seq_lengths.size(0), 'Batch size mismatch'

        # Sort the input by decreasing sequence length, required by pack_padded_sequence
        sort_seq_length, perm_idx = seq_lengths.sort(descending=True)
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
    
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from tqdm import tqdm

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Training test
    emb_dim = 5
    batch_size = 4
    max_seq_len = 3
    x = torch.rand((batch_size, max_seq_len, emb_dim)).to(device)
    lengths = torch.randint(1, max_seq_len, (batch_size,)) # Should be cpu
    model = RecurrentClassifier(emb_dim, emb_dim, 1, architecture='rnn').to(device)

    print(model(x, lengths))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    for i in tqdm(range(2000)):
        optimizer.zero_grad()
        output, _ = model(x, lengths)
        loss = torch.mean(output).real
        # print(loss)
        loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data = param.grad.data.to(param.data.device)
        # Print gradient norm
        # print(f'Gradient norm: {torch.norm(torch.stack([torch.norm(p.grad) for p in model.parameters()]))}')
        optimizer.step()