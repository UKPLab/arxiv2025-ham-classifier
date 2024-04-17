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
    python -m quantum_sent_emb --arch ham
    ```
    """
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default=None, required=True, help='Architecture to train. Options: ham, baseline')
    parser.add_argument('--emb_path', type=str, default='./embeddings/word2vec.300d.bin.gz', help='Path to word2vec embeddings')
    args = parser.parse_args()
    arch = args.arch
    emb_path = args.emb_path


    wandb.login()

    sweep_config = {
        'method': 'random',
        'name' : f'sentemb_{arch}_sweep',
        }
    metric = {
        'name': 'loss',
        'goal': 'minimize'
        }

    sweep_config['metric'] = metric

    global_params = { 
        'optimizer': {
            'value': 'adam'
            },
        'learning_rate': {
            'values': [1e-2, 1e-3, 1e-4]
            },
        'batch_size': {
            'values': [64,128,256]
            },
        'epochs': {
            'value': 30
            },
        'emb_dim': {
            'value': 300 
            },
        'circ_in': {
            'values': ['sentence','zeros']
            },
        'bias': {
            'values': ['matrix', 'diag', 'single',None]
            },
        'pos_enc': {
            'values': ['learned',None]
            },
        'batch_norm': {
            'values': [True, False]
            },
        'gates': {
            'values': [#['ry', 'rz', 'cnot_ring', 'ry','rz'], # Proposed in qiskit's EfficientSU2
                       ['rx', 'rz', 'crx_all_to_all', 'rx', 'rz'], # Circuit 6 of Sim et al 2019
                       #['rx', 'rz', 'crz_all_to_all', 'rx', 'rz'], # Circuit 5 of Sim et al 2019
                       #['ry', 'crz_ring', 'ry', 'crz_ring'], # Circuit 13 of Sim et al 2019
                       ['ry', 'crx_ring', 'ry', 'crx_ring'], # Circuit 14 of Sim et al 2019
                       ['rx', 'ry','rz'], # Control circuit without entanglement
                       #['i'] # Control empty circuit 
                       ]
            },
        'n_reps': {
            'values': [8, 16, 32]
            },
        'vocab_size' : {
            'value': None
            },

        }
    
    sweep_config['parameters'] = global_params

    sweep_id = wandb.sweep(sweep_config, project="quantum-sent-emb-v1")

    model_dir = './models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # torch_baseline, torch_quantum, lambeq
    train = build_train(arch, model_dir, emb_path, patience=5)

    # Train the network
    wandb.agent(sweep_id, train, count=100)
