import argparse
import os
import random

import wandb

from .experiment import build_train, infer
from .utils import read_config


def wandb_sweep(arch, dataset, emb_path, sweep_seed, test, patience, save_test_predictions, model_dir = './models/', count=50):
    wandb.login()

    sweep_config = {
        'method': 'random',
        'name' : f'{dataset}_{arch}_sweep',
        }
    metric = {
        'name': 'loss',
        'goal': 'minimize'
        }

    sweep_config['metric'] = metric

    global_params = read_config('configs/sweep_global.json')
    if dataset in ['sst2', 'imdb', 'agnews']:
        global_params.update({'emb_dim': {'value': 300}})
    elif dataset in ['mnist2', 'fashion']:
        if arch == 'cnn':
            global_params.update({'emb_dim': {'value': 28}})
        else:
            global_params.update({'emb_dim': {'value': 784}})
    elif dataset in ['cifar10', 'cifar2']:
        global_params.update({'emb_dim': {'value': 1024}})


    if arch == 'ham':
        ham_params = read_config('configs/sweep_ham.json')
        global_params.update(ham_params)
    elif arch == 'circ':
        circ_params = read_config('configs/sweep_circ.json')
        global_params.update(circ_params)
    elif arch == 'qlstm':
        qlstm_params = read_config('configs/sweep_qlstm.json')
        global_params.update(qlstm_params)
    elif arch == 'rnn' or arch == 'lstm':
        rnn_params = read_config('configs/sweep_rnn.json')
        global_params.update(rnn_params)
    elif arch == 'bow':
        pass # Nothing to do
    elif arch == 'mlp':
        mlp_params = read_config('configs/sweep_mlp.json')
        global_params.update(mlp_params)
    elif arch == 'cnn':
        cnn_params = read_config('configs/sweep_cnn.json')
        global_params.update(cnn_params)
    elif arch == 'qcnn':
        qcnn_params = read_config('configs/sweep_qcnn.json')
        global_params.update(qcnn_params)
    elif arch == 'ham_peffbias':
        ham_params = read_config('configs/sweep_ham_peffbias.json')
        global_params.update(ham_params)
        arch = 'ham'
    elif arch == 'ham_sim':
        ham_params = read_config('configs/sweep_ham_sim.json')
        global_params.update(ham_params)
        arch = 'ham'
    else:
        raise ValueError(f'Architecture {arch} not recognized.')

    if sweep_seed:
        global_params.update({'seed': {'values': [random.randrange(1000) for _ in range(10)]}})
    else:
        global_params.update({'seed': {'value': 42}})

    sweep_config['parameters'] = global_params

    sweep_id = wandb.sweep(sweep_config, project="ham-clas-v1")

    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # torch_baseline, torch_quantum, lambeq
    train = build_train(arch=arch, dataset=dataset, model_dir=model_dir, emb_path=emb_path, 
                        test=test, patience=patience, save_test_predictions=save_test_predictions)

    # Train the network
    wandb.agent(sweep_id, train, count=count)


def wandb_run(arch, dataset, emb_path, sweep_seed, test, save_test_predictions,
              model_dir = './models/'):
    wandb.login()

    sweep_config = {
        'method': 'grid',
        'name' : f'{dataset}_{arch}_run',
        }
    metric = {
        'name': 'loss',
        'goal': 'minimize'
        }

    sweep_config['metric'] = metric

    global_params = {}

    if sweep_seed:
        global_params.update({'seed': {'values': [random.randrange(1000) for _ in range(10)]}})
    else:
        global_params.update({'seed': {'value': 42}})

    if arch == 'ham':
        global_params.update(read_config(f'configs/run_ham_{dataset}.json'))
    elif arch == 'ham_peffbias':
        global_params.update(read_config(f'configs/run_ham_peffbias_{dataset}.json'))
        arch = 'ham'
    elif arch == 'ham_sim':
        global_params.update(read_config(f'configs/run_ham_sim_{dataset}.json'))
        arch = 'ham'
    elif arch == 'circ':
        global_params.update(read_config(f'configs/run_circ_{dataset}.json'))
    elif arch == 'rnn':
        global_params.update(read_config(f'configs/run_rnn_{dataset}.json'))
    elif arch == 'lstm':
        global_params.update(read_config(f'configs/run_lstm_{dataset}.json'))
    elif arch == 'bow':
        global_params.update(read_config(f'configs/run_bow_{dataset}.json'))
    elif arch == 'mlp':
        global_params.update(read_config(f'configs/run_mlp_{dataset}.json'))
    elif arch == 'cnn':
        global_params.update(read_config(f'configs/run_cnn_{dataset}.json'))
    elif arch == 'qcnn':
        global_params.update(read_config(f'configs/run_qcnn_{dataset}.json'))
    elif arch == 'ablation_peffbias':
        global_params.update(read_config(f'configs/run_ablation_peffbias_{dataset}.json'))
        arch = 'ham'
    elif arch == 'ablation_nobias':
        global_params.update(read_config(f'configs/run_ablation_nobias_{dataset}.json'))
        arch = 'ham'
    elif arch == 'ablation_sentin':
        global_params.update(read_config(f'configs/run_ablation_sentin_{dataset}.json'))
        arch = 'ham'
    elif arch == 'ablation_circham':
        global_params.update(read_config(f'configs/run_ablation_circham_{dataset}.json'))
        arch = 'circ'
    elif arch == 'ablation_hamhad':
        global_params.update(read_config(f'configs/run_ablation_hamhad_{dataset}.json'))
        arch = 'ham'
    elif arch == 'exp_ham_sim_qubit':
        global_params.update(read_config(f'configs/run_exp_ham_sim_qubit_{dataset}.json'))
        arch = 'ham'
    elif arch == 'exp_ham_sim_pauli':
        global_params.update(read_config(f'configs/run_exp_ham_sim_pauli_{dataset}.json'))
        arch = 'ham'
    else:
        raise ValueError(f'Architecture {arch} not recognized.')

    sweep_config['parameters'] = global_params

    if test:
        project_name = "ham-clas-v3"
    else:
        project_name = "ham-clas-v2"
    sweep_id = wandb.sweep(sweep_config, project=project_name)

    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # torch_baseline, torch_quantum, lambeq
    train = build_train(arch=arch, dataset=dataset, model_dir=model_dir, emb_path=emb_path, 
                        test=test, patience=5, save_test_predictions=save_test_predictions)

    # Train the network
    wandb.agent(sweep_id, train)


def inference(arch, dataset, model_name, emb_path, test, model_dir = './models/'):
    # If no file exist with model_name, crash
    if not os.path.exists(model_dir + model_name):
        raise ValueError(f'Model {model_name} not found in {model_dir}')
    
    infer(arch, dataset, model_name, model_dir, emb_path, test)


# def inference_simplified(dataset, model_name, emb_path, model_dir = './models/'):
#     # If no file exist with model_name, crash
#     if not os.path.exists(model_dir + model_name):
#         raise ValueError(f'Model {model_name} not found in {model_dir}')
    
#     infer_simplified(dataset, model_name, model_dir, emb_path)


def main():  # pragma: no cover
    """
    The main function executes on commands:
    `python -m ham_classifier` and `$ ham_classifier `.

    To run sweep:
    ```
    python -m ham_classifier --arch ham --dataset sst2 --mode sweep
    ```

    To run inference:
    ```
    python -m ham_classifier --arch ham --dataset sst2 --mode inference --model_name <model_name>
    ```
    
    To run a single model over many seeds:
    ```
    python -m ham_classifier --arch ham --dataset sst2 --mode run --sweep_seed
    ```

    """
    # TODO: update help messages
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, help='Mode to run. Options: sweep, run, inference')
    parser.add_argument('--arch', type=str, default=None, help='Architecture to train. Options: ham, baseline')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset to use. Options: sst2, imdb')
    parser.add_argument('--emb_path', type=str, default='./embeddings/GoogleNews-vectors-negative300.bin.gz', help='Path to word2vec embeddings')
    parser.add_argument('--sweep_seed', action='store_true', help='Enables multiple runs with different seeds.')
    parser.add_argument('--test', action='store_true', help='Use original sst2 splits.')
    parser.add_argument('--save_test_predictions', action='store_true', help='Save test predictions.')
    parser.add_argument('--model_dir', type=str, default='./models/', help='Directory to save models')
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
    parser.add_argument('--count', type=int, default=50, help='Number of runs for sweep')
    args = parser.parse_args()
    mode = args.mode
    arch = args.arch
    dataset = args.dataset
    emb_path = args.emb_path
    sweep_seed = args.sweep_seed
    test = args.test
    save_test_predictions = args.save_test_predictions
    model_name = args.model_name
    patience = args.patience
    count = args.count

    if mode == 'sweep':
        wandb_sweep(arch, dataset, emb_path, sweep_seed, test, patience=patience, 
                    save_test_predictions=save_test_predictions, count=count)
    elif mode == 'inference':
        inference(arch, dataset, model_name, emb_path, test) 
    # elif mode == 'inference_simplified':
    #     inference_simplified(dataset, model_name, emb_path)
    elif mode == 'run':
        wandb_run(arch, dataset, emb_path, sweep_seed, test, save_test_predictions=save_test_predictions)
