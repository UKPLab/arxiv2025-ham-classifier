import os
import wandb
import argparse

from quantum_sent_emb.wandb import build_train

def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m quantum_sent_emb` and `$ quantum_sent_emb `.

    This is your program's entry point.

    You can change this function to do whatever you want.
    Examples:
        * Run a test suite
        * Run a server
        * Do some other stuff
        * Run a command line application (Click, Typer, ArgParse)
        * List all available tasks
        * Run an application (Flask, FastAPI, Django, etc.)
    """
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default=None, required=True, help='Type of model to train. Options: torch_baseline, torch_quantum, lambeq')
    parser.add_argument('--model_dir', type=str, default='./models/', help='Path to model directory')
    parser.add_argument('--emb_path', type=str, default='./embeddings/word2vec.300d.bin.gz', help='Path to word2vec embeddings')
    args = parser.parse_args()
    type = args.type
    model_dir = args.model_dir
    emb_path = args.emb_path


    wandb.login()
     # Set from command line to either torch_baseline, torch_quantum or lambeq

    sweep_config = {
        'method': 'random',
        'name' : f'composition_{type}_sweep', # Set this to a unique name
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
            'values': [8,16,32,64]
            },
        'epochs': {
            'value': 30
            },
        'embedding_dim': {
            'values': [16, 32, 64, 128, 256]#, 300, 512] 
            },
        'limit' : {
            'values': [10000, 50000, 100000, 500000]
            },
        'n_qubits': {
            'values': [4, 8, 16]
            },
        }
    
    sweep_config['parameters'] = global_params

    sweep_id = wandb.sweep(sweep_config, project="quantum-sent-emb-v0")

    model_dir = './models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # torch_baseline, torch_quantum, lambeq
    train = build_train(type=type, dataset_dir=dataset_dir, model_dir=model_dir, emb_path=emb_path)

    # Train the network
    wandb.agent(sweep_id, train, count=50)
