# Define the training loop
import os
import time
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset, concatenate_datasets
from .hamiltonian import HamiltonianClassifier
from .baseline import BagOfWordsClassifier, RecurrentClassifier, QuantumCircuitClassifier
from .embedding import Embedder
from .dataloading import CustomDataset
from .utils import DotDict

# wandb requires programmatically setting some components starting from simple string hyperparameters
# The following functions achieve this

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



def build_dataset(config, test, shuffle=True, eval_batch_size=256):
    # Loads dataset from hf
    datasets = load_dataset("sst2")

    if test == False:
        assert hasattr(config,'batch_size'), 'Batch size must be provided for torch dataset'
        batch_size = config.batch_size
        train_dataset = CustomDataset(datasets["train"], data_key='sentence', label_key='label')
        dev_dataset = CustomDataset(datasets["validation"], data_key='sentence', label_key='label')
        test_dataset = CustomDataset(datasets["test"], data_key='sentence', label_key='label')
    else:
        train_dataset = CustomDataset(datasets["train"], data_key='sentence', label_key='label')
        dev_dataset = CustomDataset(datasets["validation"], data_key='sentence', label_key='label')
        test_dataset = CustomDataset(datasets["test"], data_key='sentence', label_key='label')
        train_dataset = concatenate_datasets([train_dataset, dev_dataset])

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

        return HamiltonianClassifier(emb_dim=config.emb_dim, circ_in=config.circ_in, 
                                     bias=config.bias, gates=config.gates, n_reps=config.n_reps,
                                     pos_enc=config.pos_enc, batch_norm=config.batch_norm)
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


def build_parameters(arch, emb_path, device, test, config):
    '''
    Builds model, datasets and optimizer
    '''
    # Load embedding here
    embedding = Embedder(weights_path = emb_path,  vocab_size=config.vocab_size)
    assert embedding.emb_dim == config.emb_dim, 'Embedding dimension mismatch'

    # Load datasets
    all_datasets = build_dataset(config, test)
    # Build model
    model = build_model(arch, config)
    # Load embeddings
    model.to(device)

    optimizer = build_optimizer(
        model, config)

    return model, all_datasets, optimizer, embedding


def build_train(arch, model_dir, emb_path, test, patience=5):
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
            model, all_datasets, optimizer, embedding = build_parameters(arch, emb_path, device=device, test=test, config=config)
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

            
            # Evaluate on test set
            print('Evaluating on test set...')
            cumu_loss = cumu_tp = cumu_tn = cumu_fp = cumu_fn = 0
            cumu_outputs = np.array([], dtype=int)
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    data = batch['data']
                    labels = batch['label'].type(torch.float).to(device)

                    inputs, seq_lengths = embedding(data)
                    outputs, _ = model(inputs, seq_lengths)

                    if test == False:
                        loss, tp, tn, fp, fn = batch_metrics(criterion, outputs, labels)
                        cumu_loss += loss.item()
                        cumu_tp += tp
                        cumu_tn += tn
                        cumu_fp += fp
                        cumu_fn += fn
                    else:
                        outputs = (outputs > 0.5).type(torch.int)
                        cumu_outputs = np.concatenate((cumu_outputs, outputs.cpu().numpy()))

            print('Done.')
            
            if test == False:
                # Log loss
                test_loss, test_acc, test_f1 = epoch_metrics(cumu_loss, cumu_tp, cumu_tn, cumu_fp, cumu_fn, len(test_loader.dataset))
                print(f'Test loss: {test_loss}, Test accuracy: {test_acc}, Test F1: {test_f1}')
                wandb.log({"test loss": test_loss,
                        "test accuracy": test_acc,
                            "test F1": test_f1})
            else:
                # Save outputs to tsv with pandas
                # Columns: index prediction
                pd.DataFrame({'index': range(len(test_loader.dataset)), 'prediction': cumu_outputs}) \
                .to_csv(f'data/sst2/{arch}_{wandb.run.name}_test_predictions.tsv', sep='\t', index=False)

            # Save the best model
            if test == False:
                save_path = os.path.join(model_dir, f'model_{arch}_{wandb.run.name}.pth')
            else:
                save_path = os.path.join(model_dir, f'model_{arch}_{wandb.run.name}_test.pth')
            torch.save([model.kwargs, model.state_dict()], save_path)

            del model
            del embedding
            torch.cuda.empty_cache()
            print('Current memory allocated: ', torch.cuda.memory_allocated())

    return train


def infer(model_name, model_dir, emb_path, test):
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
    embedding = Embedder(weights_path = emb_path)
    assert embedding.emb_dim == model_kwargs.emb_dim, 'Embedding dimension mismatch'
    print('Done.')

    # Load dataset
    print('Loading dataset...')
    all_datasets = build_dataset(model_kwargs, test)
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