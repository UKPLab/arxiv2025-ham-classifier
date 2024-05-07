import os
import random
import wandb
import argparse
from .experiment import build_train, infer
from .configurations import *

def wandb_sweep(arch, emb_path, sweep_seed, test, patience, model_dir = './models/'):
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

    global_params = sweep_global
    if arch == 'ham':
        ham_params = sweep_ham
        global_params.update(ham_params)
    elif arch == 'circ':
        circ_params = sweep_circ
        global_params.update(circ_params)
    elif arch == 'rnn' or arch == 'lstm':
        rnn_params = sweep_rnn
        global_params.update(rnn_params)
    elif arch == 'bow':
        pass # Nothing to do
    else:
        raise ValueError(f'Architecture {arch} not recognized.')

    if sweep_seed:
        global_params.update({'seed': {'values': [random.randrange(1000) for _ in range(10)]}})
    else:
        global_params.update({'seed': {'value': 42}})

    sweep_config['parameters'] = global_params

    sweep_id = wandb.sweep(sweep_config, project="quantum-sent-emb-v1")

    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # torch_baseline, torch_quantum, lambeq
    train = build_train(arch=arch, model_dir=model_dir, emb_path=emb_path, test=test, patience=patience)

    # Train the network
    wandb.agent(sweep_id, train, count=25)


def wandb_run(arch, emb_path, sweep_seed, test, model_dir = './models/'):
    wandb.login()

    sweep_config = {
        'method': 'grid',
        'name' : f'sentemb_{arch}_run',
        }
    metric = {
        'name': 'loss',
        'goal': 'minimize'
        }

    sweep_config['metric'] = metric

    if arch == 'ham':
        global_params = run_ham
    elif arch == 'circ':
        global_params = run_circ
    elif arch == 'rnn':
        global_params = run_rnn
    elif arch == 'lstm':
        global_params = run_lstm
    elif arch == 'bow':
        global_params = run_bow
    else:
        raise ValueError(f'Architecture {arch} not recognized.')

    if sweep_seed:
        global_params.update({'seed': {'values': [random.randrange(1000) for _ in range(10)]}})
    else:
        global_params.update({'seed': {'value': 42}})

    sweep_config['parameters'] = global_params

    sweep_id = wandb.sweep(sweep_config, project="quantum-sent-emb-v2")

    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # torch_baseline, torch_quantum, lambeq
    train = build_train(arch=arch, model_dir=model_dir, emb_path=emb_path, test=test, patience=None)

    # Train the network
    wandb.agent(sweep_id, train)


def inference(model_name, emb_path, test, model_dir = './models/'):
    # If no file exist with model_name, crash
    if not os.path.exists(model_dir + model_name):
        raise ValueError(f'Model {model_name} not found in {model_dir}')
    
    infer(model_name, model_dir, emb_path, test)

# TODO: write definitive version of this
def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m quantum_sent_emb` and `$ quantum_sent_emb `.

    To run sweep:
    ```
    python -m quantum_sent_emb --arch ham --mode sweep
    ```

    To run inference:
    ```
    python -m quantum_sent_emb --arch ham --mode inference
    ```

    To run a single model over many seeds:
    ```
    python -m quantum_sent_emb --arch ham --mode run --sweep_seed
    ```
    """
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='Mode to run. Options: sweep, run, inference')
    parser.add_argument('--arch', type=str, default=None, help='Architecture to train. Options: ham, baseline')
    parser.add_argument('--emb_path', type=str, default='./embeddings/word2vec.300d.bin.gz', help='Path to word2vec embeddings')
    parser.add_argument('--sweep_seed', action='store_true', help='Enables multiple runs with different seeds.')
    parser.add_argument('--test', action='store_true', help='Use original sst2 splits.')
    parser.add_argument('--model_dir', type=str, default='./models/', help='Directory to save models')
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    args = parser.parse_args()
    mode = args.mode
    arch = args.arch
    emb_path = args.emb_path
    sweep_seed = args.sweep_seed
    test = args.test
    model_name = args.model_name
    patience = args.patience

    if mode == 'sweep':
        wandb_sweep(arch, emb_path, sweep_seed, test, patience=patience)
    elif mode == 'inference':
        inference(model_name, emb_path, test)
    elif mode == 'run':
        wandb_run(arch, emb_path, sweep_seed, test)
