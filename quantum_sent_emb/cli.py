import os
import wandb
import argparse
from quantum_sent_emb.wandb import build_train

def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m quantum_sent_emb` and `$ quantum_sent_emb `.

    Example:
    ```
    python -m quantum_sent_emb --arch ham_mean
    ```
    """
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default=None, required=True, help='Architecture to train. Options: ham_mean, ham_weight, baseline')
    parser.add_argument('--emb_path', type=str, default='./embeddings/word2vec.300d.bin.gz', help='Path to word2vec embeddings')
    args = parser.parse_args()
    arch = args.arch
    emb_path = args.emb_path


    wandb.login()

    sweep_config = {
        'method': 'random',
        'name' : f'composition_{arch}_sweep', # Set this to a unique name
        }
    metric = {
        'name': 'loss',
        'goal': 'minimize'
        }

    sweep_config['metric'] = metric

    global_params = {
        'optimizer': {
            'values': ['adam', 'sgd']
            },
        'learning_rate': {
            'values': [1e-2, 1e-3, 1e-4, 1e-5]
            },
        'batch_size': {
            'values': [32,64,128,256]
            },
        'epochs': {
            'value': 30
            },
        'emb_dim': {
            'values': [300] 
            },
        'gates': {
            'values': [['ry', 'rz', 'cnot', 'ry','rz'], # Proposed in qiskit's EfficientSU2
                       # TODO: Circuit 6 of Sim et al 2019 (requires all to all entanglement)
                       # TODO: Circuit 5 of Sim et al 2019 (requires all to all entanglement)
                       ['ry', 'crz', 'ry', 'crz'], # Circuit 13 of Sim et al 2019
                       ['ry', 'crx', 'ry', 'crx'], # Circuit 14 of Sim et al 2019
                       ['rx', 'ry','rz'], # Control circuit without entanglement
                       ]
            },
        'n_reps': {
            'values': [4, 8, 16]
            },
        'vocab_size' : {
            'values': [None]
            },

        }
    
    sweep_config['parameters'] = global_params

    sweep_id = wandb.sweep(sweep_config, project="quantum-sent-emb-v0")

    model_dir = './models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # torch_baseline, torch_quantum, lambeq
    train = build_train(arch, model_dir, emb_path)

    # Train the network
    wandb.agent(sweep_id, train, count=50)
