# Define the training loop
import os
import time

import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision import datasets as tv_datasets

from datasets import concatenate_datasets, load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from .baseline import (BagOfWordsClassifier, MLPClassifier,
                       QuantumCircuitClassifier, RecurrentClassifier)
from .circuit import decompose_hamiltonians, pauli2matrix
from .dataloading import (CustomDataset, DecompositionDataset, CustomMNISTDataset,
                          decomposition_collate_fn)
from .embedding import NLTKEmbedder, FlattenEmbedder
from .hamiltonian import HamiltonianClassifier
from .utils import DotDict
import torch

# wandb requires programmatically setting some components starting from simple string hyperparameters
# The following functions achieve this

def load_model(model_name, model_dir, device):
    # Load model
    print('Loading model...')
    arch = model_name.split('_')[1]
    model_kwargs, model_state_dict = torch.load(os.path.join(model_dir, model_name))
    model_kwargs = DotDict(model_kwargs) # Ugly trick to access kwargs as attributes
    model = build_model(arch, model_kwargs)
    model.load_state_dict(model_state_dict)
    model.to(device)
    if hasattr(model, 'update'):
        model.update()
    print('Done.')
    return model, model_kwargs

def load_embedding(emb_path):
    # Load embedding
    print('Loading embedding...')
    embedding = NLTKEmbedder(weights_path = emb_path)
    print('Done.')
    return embedding


def batch_metrics(criteria, outputs, labels):
    '''
    Computes metrics for a batch of data
    '''
    loss = criteria(outputs, labels)
    pred = (outputs > 0.5)
    pos = (labels == 1)
    neg = (labels == 0)
    tp = torch.sum((pred == labels) & pos).item()
    tn = torch.sum((pred == labels) & neg).item()
    fp = torch.sum((pred != labels) & neg).item()
    fn = torch.sum((pred != labels) & pos).item()
    assert tp + tn + fp + fn == len(labels)
    return loss, tp, tn, fp, fn

def epoch_metrics(cumu_loss, cumu_tp, cumu_tn, cumu_fp, cumu_fn, len_dataset):
    '''
    Computes metrics for an epoch
    '''
    loss = cumu_loss / len_dataset
    acc = (cumu_tp + cumu_tn) / len_dataset
    f1 = 2 * cumu_tp / (2 * cumu_tp + cumu_fp + cumu_fn)
    return loss, acc, f1


def build_dataset(dataset, config, test, shuffle=True, eval_batch_size=256, batch_size=None):
    if batch_size is None:
        assert hasattr(config,'batch_size'), 'Batch size must be provided for torch dataset'
        batch_size = config.batch_size
    
    # Loads dataset from hf
    if dataset == 'sst2':
        datasets = load_dataset("sst2")

        if test == False:
            train_dataset = CustomDataset(datasets["train"], data_key='sentence', label_key='label')
            dev_dataset = CustomDataset(datasets["validation"], data_key='sentence', label_key='label')
            test_dataset = CustomDataset(datasets["test"], data_key='sentence', label_key='label')
        else:
            train_dataset = CustomDataset(datasets["train"], data_key='sentence', label_key='label')
            dev_dataset = CustomDataset(datasets["validation"], data_key='sentence', label_key='label')
            test_dataset = CustomDataset(datasets["test"], data_key='sentence', label_key='label')
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, dev_dataset])
    elif dataset == 'imdb':
        datasets = load_dataset("stanfordnlp/imdb")

        if test == False:
            train_dataset = CustomDataset(datasets["train"], data_key='text', label_key='label')
            dev_dataset = CustomDataset(datasets["test"], data_key='text', label_key='label')
            test_dataset = CustomDataset(datasets["unsupervised"], data_key='text', label_key='label')
        else:
            train_dataset = CustomDataset(datasets["train"], data_key='text', label_key='label')
            dev_dataset = CustomDataset(datasets["test"], data_key='text', label_key='label')
            test_dataset = CustomDataset(datasets["unsupervised"], data_key='text', label_key='label')
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, dev_dataset])
    elif dataset == 'mnist2':
        from torchvision import datasets, transforms
        import torch

        # Define a transform to convert the images to tensors and normalize them
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        # Load the MNIST dataset
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

        # Filter out the 0 and 1 digits for training and test datasets
        train_mask = (full_train_dataset.targets == 0) | (full_train_dataset.targets == 1)
        test_mask = (full_test_dataset.targets == 0) | (full_test_dataset.targets == 1)

        # Create the filtered subsets
        filtered_train_indices = torch.where(train_mask)[0]
        filtered_test_indices = torch.where(test_mask)[0]
        
        train_dataset = CustomMNISTDataset(full_train_dataset, filtered_train_indices)
        test_dataset = CustomMNISTDataset(full_test_dataset, filtered_test_indices)

        # Split the training set into train and dev
        train_size = int(0.8 * len(train_dataset))
        dev_size = len(train_dataset) - train_size
        train_dataset, dev_dataset = torch.utils.data.random_split(train_dataset, [train_size, dev_size])

    else:
        raise ValueError('Invalid dataset name')


    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    dev_loader = DataLoader(
        dev_dataset, batch_size=eval_batch_size, shuffle=shuffle, num_workers=8)
    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=8)
    return train_loader, dev_loader, test_loader


def build_model(arch, config):
    if arch == 'ham':
        assert hasattr(config,'emb_dim'), 'Embedding dimension must be provided for hamiltonian model'
        assert hasattr(config,'circ_in'), 'Type of circ input must be provided for hamiltonian model'
        assert hasattr(config,'bias'), 'Bias type must be provided for hamiltonian model'
        assert hasattr(config,'gates'), 'Gates must be provided for hamiltonian model'
        assert hasattr(config,'n_reps'), 'Number of repetitions must be provided for hamiltonian model'
        assert hasattr(config,'pos_enc'), 'Positional encoding must be provided for hamiltonian model'
        assert hasattr(config,'batch_norm'), 'Batch normalization must be provided for hamiltonian model'
        assert hasattr(config,'n_paulis'), 'Number of paulis must be provided for hamiltonian model'
        assert hasattr(config,'strategy'), 'Strategy must be provided for hamiltonian model'
        assert hasattr(config,'n_wires'), 'Number of wires must be provided for hamiltonian model'

        return HamiltonianClassifier(emb_dim=config.emb_dim, circ_in=config.circ_in, 
                                     bias=config.bias, gates=config.gates, n_reps=config.n_reps,
                                     pos_enc=config.pos_enc, batch_norm=config.batch_norm,
                                     n_paulis=config.n_paulis, strategy=config.strategy, n_wires=config.n_wires)
    elif arch == 'rnn' or arch == 'lstm':
        assert hasattr(config,'emb_dim'), 'Embedding dimension must be provided for recurrent model'
        assert hasattr(config,'hidden_dim'), 'Hidden dimension must be provided for recurrent model'
        assert hasattr(config,'rnn_layers'), 'Number of rnn layers must be provided for recurrent model'

        return RecurrentClassifier(emb_dim=config.emb_dim, hidden_dim=config.hidden_dim, 
                                   rnn_layers=config.rnn_layers, architecture=arch)
    elif arch == 'circ':
        assert hasattr(config,'emb_dim'), 'Embedding dimension must be provided for quantum circuit model'
        assert hasattr(config,'gates'), 'Gates must be provided for quantum circuit model'
        assert hasattr(config,'pos_enc'), 'Positional encoding must be provided for quantum circuit model'
        assert hasattr(config,'bias'), 'Bias type must be provided for quantum circuit model'
        assert hasattr(config,'n_reps'), 'Number of repetitions must be provided for quantum circuit model'
        assert hasattr(config,'clas_type'), 'Classifier type must be provided for quantum circuit model'

        return QuantumCircuitClassifier(emb_dim=config.emb_dim, clas_type=config.clas_type, gates=config.gates,
                                        pos_enc=config.pos_enc, bias=config.bias, n_reps=config.n_reps)
    elif arch == 'bow':
        assert hasattr(config,'emb_dim'), 'Embedding dimension must be provided for bag of words model'

        return BagOfWordsClassifier(emb_dim=config.emb_dim)
    elif arch == 'mlp':
        assert hasattr(config,'emb_dim'), 'Embedding dimension must be provided for mlp model'
        assert hasattr(config,'hidden_dim'), 'Hidden dimension must be provided for mlp model'
        assert hasattr(config,'n_layers'), 'Number of layers must be provided for mlp model'
        return MLPClassifier(emb_dim=config.emb_dim, hidden_dim=config.hidden_dim, n_layers=config.n_layers)
    else:
        raise ValueError('Invalid model architecture.')


def build_optimizer(model, config, momentum=0.9):
    assert hasattr(config,'optimizer'), 'Optimizer must be provided'
    assert hasattr(config,'learning_rate'), 'Learning rate must be provided'
    optimizer = config.optimizer
    learning_rate = config.learning_rate

    if optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate, momentum=momentum)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate)
    return optimizer


def build_parameters(arch, dataset, emb_path, device, test, config):
    '''
    Builds model, datasets and optimizer
    '''
    # Load embedding here
    if dataset == 'sst2' or dataset == 'imdb':
        embedding = NLTKEmbedder(weights_path = emb_path,  vocab_size=config.vocab_size)
        assert embedding.emb_dim == config.emb_dim, 'Embedding dimension mismatch'
    elif dataset == 'mnist2':
        embedding = FlattenEmbedder(device=device)


    # Load datasets
    all_datasets = build_dataset(dataset, config, test)
    # Build model
    model = build_model(arch, config)
    # Load embeddings
    model.to(device)

    optimizer = build_optimizer(
        model, config)

    return model, all_datasets, optimizer, embedding


def build_train(arch, dataset, model_dir, emb_path, test, patience=5):
    '''
    Builds a training function with just a config arg
    Necessary for wandb sweep
    '''
    def train(config=None):
        # Finds device to run on
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Running on {device}')

        # Initialize a new wandb run
        with wandb.init(config=config):#, Tensor.backend('pytorch'):
            config = wandb.config
            torch.manual_seed(config.seed)

            # Build model, datasets and optimizer
            print('Building model, datasets, optimizer and embedding...')
            model, all_datasets, optimizer, embedding = build_parameters(arch, dataset, emb_path, device=device, test=test, config=config)
            print('Done.')
            print(f'Now evaluating model: {model.kwargs}')
            
            n_params = model.get_n_params() # Dict containing n_params for every part 
            wandb.log(n_params)

            print('Sending model & embedding to device...')
            model = model.to(device)
            embedding = embedding.to(device)
            print('Done.')

            train_loader, dev_loader, test_loader = all_datasets

            # Define loss function and optimizer
            criterion = nn.BCELoss()

            print('Training...')
            total_time = 0
            train_time = 0
            dev_eval_time = 0
            best_dev_loss = float('inf')
            for epoch in range(config.epochs):
                print(f'Epoch {epoch+1}/{config.epochs}')
                # Save current time
                start_epoch = time.time()

                cumu_loss = cumu_tp = cumu_tn = cumu_fp = cumu_fn = 0
                for batch in tqdm(train_loader):
                    data = batch['data']
                    labels = batch['label'].type(torch.float).to(device)
                    # Zero the gradients
                    optimizer.zero_grad()

                    inputs, seq_lengths = embedding(data)

                    # Forward pass
                    outputs, _ = model(inputs, seq_lengths)
                    loss, tp, tn, fp, fn = batch_metrics(criterion, outputs, labels)
                    cumu_loss += loss.item()
                    cumu_tp += tp
                    cumu_tn += tn
                    cumu_fp += fp
                    cumu_fn += fn

                    # Backward pass and optimization
                    loss.backward()

                    # Ugly workaround to have grads on gpu
                    if hasattr(model, 'update'):                        
                        for param in model.parameters():
                            if param.grad is not None:
                                param.grad.data = param.grad.data.to(param.data.device)
                        optimizer.step()
                        model.update()
                    else:
                        optimizer.step()

                    # Log loss
                    wandb.log({"batch loss": loss.item()})
                print('Done.')

                # Compute metrics for the epoch
                train_loss, train_acc, train_f1 = epoch_metrics(cumu_loss, cumu_tp, cumu_tn, cumu_fp, cumu_fn, len(train_loader.dataset))
                print(f'Train loss: {train_loss}, Train accuracy: {train_acc}, Train F1: {train_f1}')

                # Log train runtime in minutes
                train_epoch = time.time()
                train_time += (train_epoch - start_epoch) / 60               
                    
                # Evaluate on dev set
                print('Evaluating on dev set...')
                cumu_loss = cumu_tp = cumu_tn = cumu_fp = cumu_fn = 0
                with torch.no_grad():
                    for batch in tqdm(dev_loader):
                        data = batch['data']
                        labels = batch['label'].type(torch.float).to(device)
                        inputs, seq_lengths = embedding(data)

                        # Forward pass
                        outputs, _ = model(inputs, seq_lengths)
                        loss, tp, tn, fp, fn = batch_metrics(criterion, outputs, labels)
                        cumu_loss += loss.item()
                        cumu_tp += tp
                        cumu_tn += tn
                        cumu_fp += fp
                        cumu_fn += fn

                
                # Compute metrics for the epoch
                dev_loss, dev_acc, dev_f1 = epoch_metrics(cumu_loss, cumu_tp, cumu_tn, cumu_fp, cumu_fn, len(dev_loader.dataset))
                print(f'Dev loss: {dev_loss}, Dev accuracy: {dev_acc}, Dev F1: {dev_f1}')
                print('Done.')

                # Log evaluation runtime in minutes
                eval_epoch = time.time()
                dev_eval_time += (eval_epoch - train_epoch) / 60
                
                # Log total runtime including evaluation
                total_time += (eval_epoch - start_epoch) / 60
                wandb.log({"epoch": epoch, 
                           "loss": train_loss, 
                           "dev loss": dev_loss,
                           "train accuracy": train_acc,
                           "dev accuracy": dev_acc,
                            "train F1": train_f1,
                            "dev F1": dev_f1,
                           "train epoch time": train_time, 
                           "dev eval epoch time": dev_eval_time,
                           "total epoch time": total_time})
                
                if patience is not None:
                    # Check if the current validation loss is better than the previous best loss
                    if dev_loss < best_dev_loss * 0.99:
                        best_dev_loss = dev_loss
                        counter = 0  # Reset the counter since there's improvement
                    else:
                        counter += 1  # Increment the counter as no improvement occurred
                        
                    # Check if the counter has exceeded the patience limit
                    if counter >= patience:
                        print("Early stopping: No improvement in validation loss for {} epochs.".format(patience))
                        break  # Exit the training loop

            if test:
                # Evaluate on test set
                print('Evaluating on test set...')
                cumu_loss = cumu_tp = cumu_tn = cumu_fp = cumu_fn = 0
                cumu_outputs = np.array([], dtype=int)
                with torch.no_grad():
                    for batch in tqdm(test_loader):
                        data = batch['data']

                        inputs, seq_lengths = embedding(data)
                        outputs, _ = model(inputs, seq_lengths)

                        outputs = (outputs > 0.5).type(torch.int)
                        cumu_outputs = np.concatenate((cumu_outputs, outputs.cpu().numpy()))

                print('Done.')
                
                # Save outputs to tsv with pandas
                # Columns: index prediction
                pd.DataFrame({'index': range(len(test_loader.dataset)), 'prediction': cumu_outputs}) \
                .to_csv(f'data/sst2/{arch}_{wandb.run.name}_test_predictions.tsv', sep='\t', index=False)

            # Save the best model
            if test == False:
                save_path = os.path.join(model_dir, f'model_{dataset}_{arch}_{wandb.run.name}.pth')
            else:
                save_path = os.path.join(model_dir, f'model_{dataset}_{arch}_{wandb.run.name}_test.pth')
            torch.save([model.kwargs, model.state_dict()], save_path)

            del model
            del embedding
            torch.cuda.empty_cache()
            print('Current memory allocated: ', torch.cuda.memory_allocated())

    return train


def infer(dataset, model_name, model_dir, emb_path, test):
    '''
    Inference function
    '''
    # Finds device to run on
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    # Load model
    print('Loading model...')
    arch = model_name.split('_')[1]
    model_kwargs, model_state_dict = torch.load(os.path.join(model_dir, model_name))
    model_kwargs = DotDict(model_kwargs) # Ugly trick to access kwargs as attributes
    model = build_model(arch, model_kwargs)
    model.load_state_dict(model_state_dict)
    model.to(device)
    if hasattr(model, 'update'):
        model.update()
    print('Done.')

    # Load embedding here
    print('Loading embedding...')
    if dataset == 'sst2' or dataset == 'imdb':
        embedding = NLTKEmbedder(weights_path = emb_path,  vocab_size=model_kwargs.vocab_size)
        assert embedding.emb_dim == model_kwargs.emb_dim, 'Embedding dimension mismatch'
    elif dataset == 'mnist2':
        embedding = FlattenEmbedder()
    print('Done.')

    # Load dataset
    print('Loading dataset...')
    all_datasets = build_dataset(dataset, model_kwargs, test, batch_size=256)
    if test:
        _, _, data_loader = all_datasets
    else:
        _, data_loader, _ = all_datasets
    print('Done.')

    # Define loss function
    criterion = nn.BCELoss()

    # Evaluate on test set
    if test:
        print('Evaluating on test set...')
    else:
        print('Evaluating on dev set...')
    cumu_loss = cumu_tp = cumu_tn = cumu_fp = cumu_fn = 0
    cumu_outputs = np.array([], dtype=int)
    with torch.no_grad():
        for batch in tqdm(data_loader):
            print(batch['idx'])
            data = batch['data']
            labels = batch['label'].type(torch.float).to(device)

            inputs, seq_lengths = embedding(data)
            outputs, _ = model(inputs, seq_lengths)

            if test:
                outputs = (outputs > 0.5).type(torch.int)
                cumu_outputs = np.concatenate((cumu_outputs, outputs.cpu().numpy()))                
            else:
                loss, tp, tn, fp, fn = batch_metrics(criterion, outputs, labels)
                cumu_loss += loss.item()
                cumu_tp += tp
                cumu_tn += tn
                cumu_fp += fp
                cumu_fn += fn


    print('Done.')

    print('Saving results...')
    if test == False:
        # Log loss
        dev_loss, dev_acc, dev_f1 = epoch_metrics(cumu_loss, cumu_tp, cumu_tn, cumu_fp, cumu_fn, len(data_loader.dataset))
        print(f'Dev loss: {dev_loss}, Dev accuracy: {dev_acc}, Dev F1: {dev_f1}')
    else:
        # Save outputs to tsv with pandas
        # Columns: index prediction
        pd.DataFrame({'index': range(len(data_loader.dataset)), 'prediction': cumu_outputs}) \
        .to_csv(f'data/sst2/{arch}_{model_name}_test_predictions.tsv', sep='\t', index=False)
    print('Done.')

    del model
    del embedding
    torch.cuda.empty_cache()
    
    print('Current memory allocated: ', torch.cuda.memory_allocated())


# def infer_simplified(model_name, model_dir, emb_path, data_path, coeff_steps=1000):
#     '''
#     Inference function for simplified Hamiltonians
#     '''
#     # Finds device to run on
#     device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#     print(f'Running on {device}')

#     # Load model
#     model, _ = load_model(model_name, model_dir, device)

#     # Load dataset
#     print('Loading dataset...')
#     dataset = DecompositionDataset(data_path)
#     data_loader = DataLoader(dataset, batch_size=16, collate_fn=decomposition_collate_fn)
#     print('Done.')

#     # Define loss function
#     criterion = nn.BCELoss()

#     outputs_stack = []
#     labels_stack = []
#     print('Evaluating on dataset...')
#     with torch.no_grad():
#         max_len = 0
#         for i, (batch_pauli, batch_coeff, batch_label) in tqdm(enumerate(data_loader)): # Iterates over batches        
#             outputs = torch.zeros(len(batch_pauli), len(batch_pauli[0])//coeff_steps)
#             print(batch_coeff[0])
            
#             # Note: this assumes all hamiltonians have same no of coeffs
#             # Note: this ignores the last few strings in the last batch
#             # Shouldn't be a problem since the coeffs are very small
#             hamiltonians = torch.empty(len(batch_label), 2**len(batch_pauli[0][0]), 2**len(batch_pauli[0][0]), 
#                                        device=device, dtype=torch.complex64)
#             for j in tqdm(range(0, len(batch_coeff[0])//coeff_steps)): # Iterates over no of coefficients 
#                 print('Evaluating on the first {} coefficients'.format((j+1)*coeff_steps))

#                 for k, (paulis, coeffs) in enumerate(zip(batch_pauli, batch_coeff)): # Iterates over samples in batch
#                     string_list = [(p,c) for p, c in zip(paulis[j*coeff_steps:(j+1)*coeff_steps], coeffs[j*coeff_steps:(j+1)*coeff_steps])]
#                     if len(string_list) != 0:
#                         hamiltonians[k] += pauli2matrix(string_list).to(device)

#                 # Evaluate the Hamiltonian
#                 states = model.state(torch.zeros(len(batch_pauli)))
#                 outputs[:, j] = model.expval(hamiltonians, states)

#             outputs_stack.append(outputs)  
#             labels_stack.append(batch_label)
#             if max_len < outputs.shape[1]:
#                 max_len = outputs.shape[1]
            
#         # torch.cat([torch.nn.functional.pad(outputs, (0, 130 - outputs.shape[1])) for outputs in outputs_stack])
#         outputs_stack = [torch.nn.functional.pad(outputs, (0, max_len - outputs.shape[1])) for outputs in outputs_stack]
#         outputs_stack = torch.cat(outputs_stack).to(device)
#         labels = torch.cat(labels_stack)
        
#         assert outputs_stack.shape[0] == labels.shape[0], 'Output shape mismatch'
#     print('Done.')
    
#     print('Saving results...')
#     arch = model_name.split('_')[1]
#     # Create empty df with columns loss, tp, tn, fp, fn, accuracy, f1, string count
#     metrics_df = []
#     outputs_split = torch.split(outputs_stack, 1, dim=1)
#     for i, outputs in enumerate(outputs_split):
#         outputs = outputs.squeeze()
#         epoch_loss, epoch_tp, epoch_tn, epoch_fp, epoch_fn = batch_metrics(criterion, outputs, labels)
#         dev_loss, dev_acc, dev_f1 = epoch_metrics(epoch_loss, epoch_tp, 
#                                                     epoch_tn, epoch_fp, epoch_fn, len(labels))
#         print(f'String count: {(i+1) * coeff_steps} Dev loss: {dev_loss}, Dev accuracy: {dev_acc}, Dev F1: {dev_f1}')
#         metrics_df.append({'loss': dev_loss.cpu().item(), 'tp': epoch_tp, 'tn': epoch_tn, 'fp': epoch_fp, 'fn': epoch_fn,
#                                         'accuracy': dev_acc, 'f1': dev_f1, 'string count': (i+1) * coeff_steps})
#     metrics_df = pd.DataFrame(metrics_df)
#     metrics_df.to_csv(f'data/sst2/{model_name}_decomposed_metrics.tsv', index=False)

#     # Save outputs to tsv with pandas
#     # Columns: index prediction
#     pred_df = pd.DataFrame(outputs_stack.cpu())
#     pred_df.columns = [f'pred_{c*coeff_steps}' for c in pred_df.columns]
#     pred_df.to_csv(f'data/sst2/{arch}_{model_name}_decomposed_predictions.tsv', sep='\t', index=False)
#     print('Done.')

#     del model
#     torch.cuda.empty_cache()
    
#     print('Current memory allocated: ', torch.cuda.memory_allocated())

def infer_simplified(dataset, model_name, model_dir, emb_path, coeff_steps=1000):
    # Finds device to run on
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    model, model_kwargs = load_model(model_name, model_dir, device)
    embedding = load_embedding(emb_path)

    assert embedding.emb_dim == model_kwargs.emb_dim, 'Embedding and model have different dimensions.'

    # Load dataset
    print('Loading dataset...')
    all_datasets = build_dataset(dataset, model_kwargs, test=False, batch_size=128, eval_batch_size=128)
    _, data_loader, _ = all_datasets
    print('Done.')

    criterion = nn.BCELoss()
    rec_outputs_stack = []
    labels_stack = []
    outputs_stack = []
    with torch.no_grad():
        # counter = 0
        for batch in tqdm(data_loader):
            # counter += 1
            # if counter > 2:
            #     break
            data = batch['data']
            labels = batch['label'].type(torch.float).to(device)
            labels_stack.append(labels)

            inputs, seq_lengths = embedding(data)
            hamiltonians = model.hamiltonian(inputs, seq_lengths)
            hamiltonians_list = [h.to('cpu').numpy() for h in hamiltonians]
            states = model.state(inputs)

            outputs = model.expval(hamiltonians, states)
            outputs_stack.append(outputs)
            acc = (outputs > 0.5).type(torch.int) == labels
            # Decompose Hamiltonian
            # decompositions = [pauli_decompose(h) for h in hamiltonians]
            decompositions = decompose_hamiltonians(hamiltonians_list)

            # decompositions_dump = [(d,l) for d,l in zip(decompositions, labels)]
            # # Append decompositions to a file
            # for dec in decompositions_dump:
            #     pickle.dump(dec, f)
            
            # Check similarity of hamiltonian to original
            reconstructions = torch.zeros((len(labels), hamiltonians.shape[1], hamiltonians.shape[1]), 
                                            device=device, dtype=torch.complex64)

            max_len = max([len(d) for d in decompositions])
            rec_outputs = torch.zeros(len(labels), max_len//coeff_steps)

            for j in tqdm(range(0, max_len//coeff_steps)): # Iterates over no of coefficients 
                # if j > 5:
                #     break
                print('Evaluating on the first {} coefficients'.format((j+1)*coeff_steps))
                for i,d in enumerate(decompositions): # Iterates over samples in batch
                    #j*coeff_steps:(j+1)*coeff_steps
                    string_list = d[j*coeff_steps:(j+1)*coeff_steps]
                    if len(string_list) != 0:
                        reconstructions[i] += pauli2matrix(string_list).to(device)

                # Compare Hamiltonians with their decompositions
                fro_diff = torch.norm(hamiltonians - torch.nn.functional.normalize(reconstructions, dim=0), p='fro').item()
                print(f'Frobenius difference: {fro_diff}')

                # Evaluate the model on original vs reconstructed Hamiltonian
                rec_batch_outputs = model.expval(torch.nn.functional.normalize(reconstructions, dim=0), states)
                rec_outputs[:, j] = rec_batch_outputs
                
                # TODO: cache loss
                rec_acc = (rec_batch_outputs > 0.5).type(torch.int) == labels

                print(f'Accuracy on original: {acc.float().mean().item()}')
                print(f'Accuracy on reconstructed: {rec_acc.float().mean().item()}')
            rec_outputs_stack.append(rec_outputs)
            # outputs_stack.append(outputs)

        rec_outputs_stack = [torch.nn.functional.pad(rec_outputs, (0, max_len//coeff_steps - rec_outputs.shape[1])) for rec_outputs in rec_outputs_stack]
        rec_outputs_stack = torch.cat(rec_outputs_stack).to(device)
        labels = torch.cat(labels_stack)
        # outputs = torch.cat(outputs_stack)

        print('Saving results...')
        arch = model_name.split('_')[1]
        # Create empty df with columns loss, tp, tn, fp, fn, accuracy, f1, string count
        metrics_df = []
        outputs_split = torch.split(rec_outputs_stack, 1, dim=1)
        for i, rec_outputs in enumerate(outputs_split):
            rec_outputs = rec_outputs.squeeze()
            epoch_loss, epoch_tp, epoch_tn, epoch_fp, epoch_fn = batch_metrics(criterion, rec_outputs, labels)
            dev_loss, dev_acc, dev_f1 = epoch_metrics(epoch_loss, epoch_tp, 
                                                        epoch_tn, epoch_fp, epoch_fn, len(labels))
            print(f'String count: {(i+1) * coeff_steps} Dev loss: {dev_loss}, Dev accuracy: {dev_acc}, Dev F1: {dev_f1}')
            metrics_df.append({'loss': dev_loss.cpu().item(), 'tp': epoch_tp, 'tn': epoch_tn, 'fp': epoch_fp, 'fn': epoch_fn,
                                            'accuracy': dev_acc, 'f1': dev_f1, 'string_count': (i+1) * coeff_steps})
        metrics_df = pd.DataFrame(metrics_df)
        metrics_df.to_csv(f'data/sst2/{model_name}_decomposed_metrics.tsv', index=False)

        # Save outputs to tsv with pandas
        # Columns: index prediction
        pred_df = pd.DataFrame(rec_outputs_stack.cpu())
        pred_df.columns = [f'pred_{c*coeff_steps}' for c in pred_df.columns]
        pred_df.to_csv(f'data/sst2/{arch}_{model_name}_decomposed_predictions.tsv', sep='\t', index=False)
        print('Done.')

        del model
        del embedding
        torch.cuda.empty_cache()
        
        print('Current memory allocated: ', torch.cuda.memory_allocated())

        print('Done.')