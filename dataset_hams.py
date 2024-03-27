import pandas as pd
from quantum_sent_emb import Embedder
import torch
from datasets import load_dataset
import pennylane as qml
from tqdm import tqdm
import concurrent.futures
import matplotlib.pyplot as plt
import seaborn as sns


def pauli_decompose(hamiltonian):
    return qml.pauli_decompose(hamiltonian)

# Decompose hamiltonias using pennylane
def decompose_hamiltonians(hamiltonians):
    # Multiprocessing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        decs = list(tqdm(executor.map(pauli_decompose, hamiltonians), total=len(hamiltonians)))
    # decs[0].items()
    return decs

def pad_to_next_power_of_two(tensor):
    # Get the size of the tensor along the emb_dim dimension
    emb_dim_size = tensor.size(-1)
    
    # Calculate the next power of two for the emb_dim size
    next_power_of_two = 1 << (emb_dim_size - 1).bit_length()
    
    # Calculate the amount of padding needed
    padding_needed = next_power_of_two - emb_dim_size
    
    # Pad the tensor along the emb_dim dimension
    padded_tensor = torch.nn.functional.pad(tensor, (0, padding_needed))
    
    return padded_tensor

# Returns a dict with a coeffs tensor and a gates tensor
def dec_to_dict(decompositions):
    coeffs = [dec.coeffs for dec in decompositions] # list of tensors (n_samples) x n_hams
    gates = torch.tensor([[ [ n2c[gate] for gate in op.name] for op in dec.ops] for dec in decompositions]) # tensor (n_hams, n_samples, n_wires)
    # gates = torch.tensor([[ [ gate for gate in op.name] for op in dec.ops] for dec in decompositions]) # tensor (n_hams, n_samples, n_wires)
    gates = gates.flatten(0,1) # tensor (n_hams*n_samples, n_wires)

    return {'coeffs':coeffs, 'gates':gates}

def plot_hams(ham_dict, name):
    cutoff = 1e-4
    coeffs = ham_dict['coeffs']
    coeffs = torch.cat(coeffs, dim=0)
    gates = ham_dict['gates']
    n_wires = gates.size(-1)
    wire_count = torch.tensor([i for i in range(n_wires)]).repeat(gates.size(0),1) # tensor (n_hams*n_samples, n_wires)
    gates_wire = torch.stack([gates, wire_count]).permute((1,2,0)).flatten(0,1)
    gates_nonI = (gates != ord('I')).sum(dim=1)
    gates_nonI_cutoff = gates_nonI[coeffs.abs() > cutoff]
    print(f'Percentage of non-identity gates: {gates_nonI.sum()/gates.numel()}')
    print(f'Percentage of non-identity gates (cutoff={cutoff}): {gates_nonI_cutoff.sum()/gates.numel()}')

    # Plot log histogram of coeffs using seaborn
    sns.histplot(coeffs.log().flatten(), bins=100)
    plt.title(f'Log Histogram of {name} Hamiltonians Coefficients')
    plt.savefig(f'plots/{name.lower().replace(' ','_')}_coeff.png')
    plt.close()

    # Plot histogram of gates separate by wire using seaborn
    df = pd.DataFrame({'gate':gates_wire[:,0], 'qubit':gates_wire[:,1]})
    df.gate = df.gate.apply(lambda x: chr(x))
    sns.countplot(df, x='qubit', hue='gate', stat='percent')
    plt.title(f'Percentage of {name} gates per Pauli string')
    plt.savefig(f'plots/{name.lower().replace(' ','_')}_gates.png')
    plt.close()

    # Plot log histogram of nonid pauli strings
    # Bins should be int numbers from 0 to n_wires
    sns.histplot(gates_nonI, bins=n_wires, discrete=True, stat='percent')
    plt.title(f'Percent of {name} non-identity gates')
    plt.savefig(f'plots/{name.lower().replace(' ','_')}_nonid.png')
    plt.close()

    # Plot log histogram of nonid pauli strings with cutoff
    # Bins should be int numbers from 0 to n_wires
    sns.histplot(gates_nonI_cutoff, bins=n_wires, discrete=True, stat='percent')
    plt.title(f'Histogram of {name} non-identity gates (cutoff={cutoff})')
    plt.savefig(f'plots/{name.lower().replace(' ','_')}_nonid_cutoff.png')
    plt.close()

    # Plot histogram of coefficient sum grouped by number of nonid gates
    df = pd.DataFrame({'coeff':coeffs**2, 'qubit':gates_nonI}).groupby('qubit').sum()
    sns.barplot(df, x='qubit', y='coeff')
    plt.title('Coefficient squared sums per number of non-identity Pauli gates')
    plt.xlabel('Number of non-identity Pauli gates')
    plt.ylabel('Sum coefficients$^2$')
    plt.savefig(f'plots/{name.lower().replace(' ','_')}_nonid_sumcoeff.png')
    plt.close()

n2c = {
    'Identity':ord('I'),
    'PauliX':ord('X'),
    'PauliY':ord('Y'),
    'PauliZ':ord('Z'),
       }

if __name__ == '__main__':
    n_samples = 1 # n samples per label

    # Initialize Embedder
    print('Initializing Embedder...')
    embedder = Embedder(weights_path='embeddings/word2vec.300d.bin.gz', vocab_size=10000, padding=False)

    # Load stanford sentiment treebank dataset
    dataset = load_dataset("sst2")

    # Load 1000 random sentences
    sentences = dataset['train']['sentence'][:1000]

    # Select 100 positive and 100 negative sentences
    print('Selecting sentences...')
    pos_sentences = [s for s, l in zip(sentences, dataset['train']['label']) if l == 1][:n_samples]
    neg_sentences = [s for s, l in zip(sentences, dataset['train']['label']) if l == 0][:n_samples]

    # Embed sentences list(sent_length x emb_dim) of size n_samples
    print('Embedding sentences...')
    pos_embeddings = embedder(pos_sentences)
    neg_embeddings = embedder(neg_sentences)

    # Takes the first 32 dimensions of the embeddings
    pos_embeddings = [e[:,:20] for e in pos_embeddings]
    neg_embeddings = [e[:,:20] for e in neg_embeddings]

    # Pad embeddings to next power of 2 so from list(sent_length x emb_dim) of size n_samples to list(sent_length x emb_dim_padded) of size n_samples
    print('Padding embeddings...')
    pos_embeddings = [pad_to_next_power_of_two(e) for e in pos_embeddings]
    neg_embeddings = [pad_to_next_power_of_two(e) for e in neg_embeddings]

    # Sentence hamiltonians as sum of pure states
    print('Computing pure hamiltonians...')
    pos_hamiltonians = [torch.einsum('jk,jl->jkl', e, e) for e in pos_embeddings]
    neg_hamiltonians = [torch.einsum('jk,jl->jkl', e, e) for e in neg_embeddings]
    pos_word_hamiltonians = torch.cat(pos_hamiltonians, dim=0) # Just combine in a tensor
    neg_word_hamiltonians = torch.cat(neg_hamiltonians, dim=0)
    pos_pure_hamiltonians = torch.stack([torch.sum(e, dim=0) for e in pos_hamiltonians])
    neg_pure_hamiltonians = torch.stack([torch.sum(e, dim=0) for e in neg_hamiltonians])
    
    # Sentence hamiltonians as sum of mixed states
    print('Computing mixed hamiltonians')
    pos_mean_emb = [torch.mean(e,dim=0) for e in pos_embeddings]
    neg_mean_emb = [torch.mean(e,dim=0) for e in neg_embeddings]
    pos_mean_emb = [torch.einsum('k,l -> kl', e, e) for e in pos_mean_emb]
    neg_mean_emb = [torch.einsum('k,l -> kl', e, e) for e in neg_mean_emb]
    pos_mix_hamiltonians = torch.cat(pos_mean_emb, dim=0) # Just combine in a tensor
    neg_mix_hamiltonians = torch.cat(neg_mean_emb, dim=0)

    # Decompose hamiltonians using pennylane
    # print('Decomposing Positive Words Hamilto nians...')
    # pos_word_dec = decompose_hamiltonians(pos_word_hamiltonians)
    # print('Decomposing Negative Words Hamiltonians...')
    # neg_word_dec = decompose_hamiltonians(neg_word_hamiltonians)
    print('Decomposing Positive Pure Hamiltonians...')
    pos_pure_dec = decompose_hamiltonians(pos_pure_hamiltonians)
    print('Decomposing Negative Pure Hamiltonians...')
    neg_pure_dec = decompose_hamiltonians(neg_pure_hamiltonians)

    print('Decomposing Positive Mixed Hamiltonians...')
    pos_mix_dec = decompose_hamiltonians(pos_pure_hamiltonians)
    print('Decomposing Negative Mixed Hamiltonians...')
    neg_mix_dec = decompose_hamiltonians(neg_pure_hamiltonians)

    # Total
    # total_word_dec = pos_word_dec + neg_word_dec
    total_pure_dec = pos_pure_dec + neg_pure_dec
    total_mix_dec = pos_mix_dec + neg_mix_dec

    # Plot histograms
    # plot_hams(dec_to_dict(pos_word_dec), 'Positive Words')
    # plot_hams(dec_to_dict(neg_word_dec), 'Negative Words')
    plot_hams(dec_to_dict(pos_pure_dec), 'Positive Pure Sentences')
    plot_hams(dec_to_dict(neg_pure_dec), 'Negative Pure Sentences')
    plot_hams(dec_to_dict(pos_mix_dec), 'Positive Mixed Sentences')
    plot_hams(dec_to_dict(neg_mix_dec), 'Negative Mixed Sentences')
    # plot_hams(dec_to_dict(total_word_dec), 'Total Words')
    plot_hams(dec_to_dict(total_pure_dec), 'Total Pure Sentences')
    plot_hams(dec_to_dict(total_mix_dec), 'Total Mixed Sentences')
    

    # Create test hamiltonians at random
    # print('Creating random test vectors...')
    # test_vectors = torch.randn((n_samples, 20))
    # test_vectors = pad_to_next_power_of_two(test_vectors)
    # test_hamiltonians = torch.einsum('jk,jl->jkl', test_vectors, test_vectors)

    # # Decompose test hamiltonians using pennylane
    # print('Decomposing test hamiltonians...')
    # test_ham_dec = decompose_hamiltonians(test_hamiltonians)

    # # Plot histograms
    # plot_hams(dec_to_dict(test_ham_dec), 'Test')
    # All plots have the same structure of word embeddings!


    print('Done!')