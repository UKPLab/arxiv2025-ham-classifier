# Define the training loop
import os
import time

import numpy as np
import pandas as pd
from torchvision import transforms
from torchvision import datasets as tv_datasets
from torcheval.metrics.functional import multiclass_f1_score

from datasets import concatenate_datasets, load_dataset
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm

import wandb

from .baseline import (BagOfWordsClassifier, MLPClassifier, CNNClassifier, QCNNClassifier,
                       QuantumCircuitClassifier, QLSTMClassifier, RecurrentClassifier)
from .circuit import decompose_hamiltonians, pauli2matrix
from .dataloading import (CustomDataset, DecompositionDataset, ClassFilteredDataset,
                          decomposition_collate_fn)
from .embedding import NLTKEmbedder, FlattenEmbedder, PassEmbedder
from .hamiltonian import HamiltonianClassifier, HamiltonianDecClassifier
from .utils import DotDict, DatasetSetup
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


def metrics(criterion, n_classes, outputs, labels):
    '''
    Computes metrics for a batch of data
    '''
    if n_classes == 2:        
        loss = criterion(outputs, labels.type(torch.float))
        pred = (outputs > 0.5)
        pos = (labels == 1)
        neg = (labels == 0)
        tp = torch.sum((pred == labels) & pos).item()
        tn = torch.sum((pred == labels) & neg).item()
        fp = torch.sum((pred != labels) & neg).item()
        fn = torch.sum((pred != labels) & pos).item()
        assert tp + tn + fp + fn == len(labels)
        accuracy = (tp + tn) / len(labels)
        f1 = 2 * tp / (2 * tp + fp + fn)
    elif n_classes > 2:
        loss = criterion(outputs, labels.type(torch.long))
        pred = torch.argmax(outputs, dim=1)
        accuracy = torch.mean((pred == labels).type(torch.float)).item()
        # Compute micro F1
        f1 = multiclass_f1_score(labels, pred, num_classes=n_classes, average='micro').item()
    else:
        raise ValueError('Invalid criterion')
        
    return loss, accuracy, f1


def build_dataset(dataset, config, test, shuffle=True, eval_batch_size=256, batch_size=None):
    if batch_size is None:
        assert hasattr(config,'batch_size'), 'Batch size must be provided for torch dataset'
        batch_size = config.batch_size
    
    # Loads dataset from hf
    if dataset == 'sst2':
        datasets = load_dataset("sst2")

        n_classes = 2
        criterion = nn.BCELoss()

        if test == False:
            train_dataset = CustomDataset(datasets["train"], data_key='sentence', label_key='label')
            dev_dataset = CustomDataset(datasets["validation"], data_key='sentence', label_key='label')
            test_dataset = CustomDataset(datasets["test"], data_key='sentence', label_key='label')
        else:
            # Set train + dev together and test as dev
            train_dataset = CustomDataset(datasets["train"], data_key='sentence', label_key='label')
            dev_dataset = CustomDataset(datasets["validation"], data_key='sentence', label_key='label')
            test_dataset = CustomDataset(datasets["test"], data_key='sentence', label_key='label')
            train_dataset = ConcatDataset([train_dataset, dev_dataset])
            dev_dataset = test_dataset
    elif dataset == 'imdb':
        datasets = load_dataset("stanfordnlp/imdb").shuffle(seed=42)

        n_classes = 2
        criterion = nn.BCELoss()

        if test == False:
            # Split train in train and dev
            train_size = int(0.8 * len(datasets["train"]))
            dev_size = len(datasets["train"]) - train_size
            train_dataset = CustomDataset(datasets["train"].select(range(train_size)), data_key='text', label_key='label')
            dev_dataset = CustomDataset(datasets["train"].select(range(train_size, train_size + dev_size)), data_key='text', label_key='label')
            test_dataset = CustomDataset(datasets["test"], data_key='text', label_key='label')
        else:
            train_dataset = CustomDataset(datasets["train"], data_key='text', label_key='label')
            dev_dataset = CustomDataset(datasets["test"], data_key='text', label_key='label')
            test_dataset = CustomDataset(datasets["test"], data_key='text', label_key='label')

    elif dataset == 'agnews':
        datasets = load_dataset("ag_news").shuffle(seed=42)

        n_classes = 4
        criterion = nn.CrossEntropyLoss()

        if test == False: # Split train into train and dev
            train_size = int(0.8 * len(datasets["train"]))
            dev_size = len(datasets["train"]) - train_size

            train_dataset = CustomDataset(datasets["train"].select(range(train_size)), data_key='text', label_key='label')
            dev_dataset = CustomDataset(datasets["train"].select(range(train_size, train_size + dev_size)), data_key='text', label_key='label')
            test_dataset = CustomDataset(datasets["test"], data_key='text', label_key='label')
        else:
            train_dataset = CustomDataset(datasets["train"], data_key='text', label_key='label')
            dev_dataset = CustomDataset(datasets["test"], data_key='text', label_key='label')
            test_dataset = CustomDataset(datasets["test"], data_key='text', label_key='label')
    elif dataset == 'mnist2':
        from torchvision import datasets, transforms
        import torch

        n_classes = 2
        criterion = nn.BCELoss()

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
        
        train_dataset = ClassFilteredDataset(full_train_dataset, filtered_train_indices)
        test_dataset = ClassFilteredDataset(full_test_dataset, filtered_test_indices)

        if test == False:
            # Split the training set into train and dev
            train_size = int(0.8 * len(train_dataset))
            dev_size = len(train_dataset) - train_size
            train_dataset, dev_dataset = random_split(train_dataset, [train_size, dev_size])
        else:
            dev_dataset = test_dataset
    elif dataset == 'fashion':
        from torchvision import datasets, transforms
        import torch

        n_classes = 10
        criterion = nn.CrossEntropyLoss()

        # Define a transform to convert the images to tensors and normalize them
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalization for Fashion-MNIST
        ])

        # Load the MNIST dataset
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

        if test == False:
            # Split the training set into train and dev
            train_size = int(0.8 * len(train_dataset))
            dev_size = len(train_dataset) - train_size
            train_dataset, dev_dataset = random_split(train_dataset, [train_size, dev_size])
        else:
            dev_dataset = test_dataset
    elif dataset == 'cifar10':
        from torchvision import datasets, transforms
        import torch

        n_classes = 10
        criterion = nn.CrossEntropyLoss()
        
        # Define transformations for the training set and test set
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        # Download and load the CIFAR-10 training and test datasets
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

        if test == False:
            # Split the training set into train and dev
            train_size = int(0.8 * len(train_dataset))
            dev_size = len(train_dataset) - train_size
            train_dataset, dev_dataset = random_split(train_dataset, [train_size, dev_size])
        else:
            dev_dataset = test_dataset
    elif dataset == 'cifar2':
        from torchvision import datasets, transforms
        import torch

        n_classes = 2
        criterion = nn.BCELoss()
        
        # Define transformations for the training set and test set
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        # Download and load the CIFAR-10 training and test datasets
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
                
        # Relabel as vehicle vs animal
        # Vehicle classes are 0,1,8,9
        # Animal classes are 2,3,4,5,6,7
        train_dataset.targets = torch.tensor([0 if x in [0,1,8,9] else 1 for x in train_dataset.targets])
        test_dataset.targets = torch.tensor([0 if x in [0,1,8,9] else 1 for x in test_dataset.targets])

        if test == False:
            # Split the training set into train and dev
            train_size = int(0.8 * len(train_dataset))
            dev_size = len(train_dataset) - train_size
            train_dataset, dev_dataset = random_split(train_dataset, [train_size, dev_size])
        else:
            dev_dataset = test_dataset


    else:
        raise ValueError('Invalid dataset name')


    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    dev_loader = DataLoader(
        dev_dataset, batch_size=eval_batch_size, shuffle=shuffle, num_workers=8)
    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=8)
    

    return DatasetSetup(n_classes, criterion, train_loader, dev_loader, test_loader)


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
        assert hasattr(config,'n_classes'), 'Number of classes must be provided for hamiltonian model'
        if not hasattr(config,'pauli_strings'):
            config.pauli_strings = None
        if not hasattr(config,'pauli_weight'):
            config.pauli_weight = None

        return HamiltonianClassifier(emb_dim=config.emb_dim, circ_in=config.circ_in, n_classes=config.n_classes,
                                     bias=config.bias, gates=config.gates, n_reps=config.n_reps, pauli_weight=config.pauli_weight,
                                     pos_enc=config.pos_enc, batch_norm=config.batch_norm, pauli_strings=config.pauli_strings,
                                     n_paulis=config.n_paulis, strategy=config.strategy, n_wires=config.n_wires)
    elif arch == 'ham_dec':
        assert hasattr(config,'emb_dim'), 'Embedding dimension must be provided for hamiltonian dec model'
        assert hasattr(config,'circ_in'), 'Type of circ input must be provided for hamiltonian dec model'
        assert hasattr(config,'n_paulis'), 'Number of paulis must be provided for hamiltonian dec model'
        assert hasattr(config,'gates'), 'Gates must be provided for hamiltonian model'
        assert hasattr(config,'n_reps'), 'Number of repetitions must be provided for hamiltonian model'
        if not hasattr(config,'pauli_strings'):
            config.pauli_strings = None
        if not hasattr(config,'pauli_weight'):
            config.pauli_weight = None
        assert hasattr(config,'n_classes'), 'Number of classes must be provided for hamiltonian dec model'

        return HamiltonianDecClassifier(emb_dim=config.emb_dim, circ_in=config.circ_in, n_paulis=config.n_paulis, 
                                 gates=config.gates, n_reps=config.n_reps, pauli_strings=config.pauli_strings, 
                                 pauli_weight=config.pauli_weight, n_classes=config.n_classes) 
    elif arch == 'rnn' or arch == 'lstm':
        assert hasattr(config,'emb_dim'), 'Embedding dimension must be provided for recurrent model'
        assert hasattr(config,'hidden_dim'), 'Hidden dimension must be provided for recurrent model'
        assert hasattr(config,'rnn_layers'), 'Number of rnn layers must be provided for recurrent model'
        assert hasattr(config,'n_classes'), 'Number of classes must be provided for recurrent model'

        return RecurrentClassifier(n_classes=config.n_classes, emb_dim=config.emb_dim, hidden_dim=config.hidden_dim, 
                                   rnn_layers=config.rnn_layers, architecture=arch)
    elif arch == 'circ':
        assert hasattr(config,'emb_dim'), 'Embedding dimension must be provided for quantum circuit model'
        assert hasattr(config,'gates'), 'Gates must be provided for quantum circuit model'
        assert hasattr(config,'pos_enc'), 'Positional encoding must be provided for quantum circuit model'
        assert hasattr(config,'bias'), 'Bias type must be provided for quantum circuit model'
        assert hasattr(config,'n_reps'), 'Number of repetitions must be provided for quantum circuit model'
        assert hasattr(config,'clas_type'), 'Classifier type must be provided for quantum circuit model'
        assert hasattr(config,'n_classes'), 'Number of classes must be provided for quantum circuit model'

        return QuantumCircuitClassifier(emb_dim=config.emb_dim, clas_type=config.clas_type, gates=config.gates,
                                        pos_enc=config.pos_enc, bias=config.bias, n_reps=config.n_reps, n_classes=config.n_classes)
    elif arch == 'qlstm':
        assert hasattr(config,'emb_dim'), 'Embedding dimension must be provided for quantum lstm model'
        assert hasattr(config,'hidden_dim'), 'Hidden dimension must be provided for quantum lstm model'
        assert hasattr(config,'n_wires'), 'Number of wires must be provided for quantum lstm model'
        assert hasattr(config,'n_layers'), 'Number of layers must be provided for quantum lstm model'
        assert hasattr(config,'n_classes'), 'Number of classes must be provided for quantum lstm model'
        assert hasattr(config,'gates'), 'Gates must be provided for quantum lstm model'

        return QLSTMClassifier(emb_dim=config.emb_dim, hidden_dim=config.hidden_dim, n_wires=config.n_wires,
                               n_layers=config.n_layers, gates=config.gates, n_classes=config.n_classes)
    
    elif arch == 'bow':
        assert hasattr(config,'emb_dim'), 'Embedding dimension must be provided for bag of words model'
        assert hasattr(config,'n_classes'), 'Number of classes must be provided for bag of words model'

        return BagOfWordsClassifier(emb_dim=config.emb_dim, n_classes=config.n_classes)
    elif arch == 'mlp':
        assert hasattr(config,'emb_dim'), 'Embedding dimension must be provided for mlp model'
        assert hasattr(config,'hidden_dim'), 'Hidden dimension must be provided for mlp model'
        assert hasattr(config,'n_layers'), 'Number of layers must be provided for mlp model'
        assert hasattr(config,'n_classes'), 'Number of classes must be provided for mlp model'
        return MLPClassifier(emb_dim=config.emb_dim, hidden_dim=config.hidden_dim, n_layers=config.n_layers, n_classes=config.n_classes)
    elif arch == 'cnn':
        assert hasattr(config,'in_channels'), 'Number of input channels must be provided for cnn model'
        assert hasattr(config,'n_layers'), 'Number of layers must be provided for cnn model'
        assert hasattr(config,'emb_dim'), 'Embedding dimension must be provided for cnn model'
        assert hasattr(config,'out_channels'), 'Output channels must be provided for cnn model'
        assert hasattr(config,'n_classes'), 'Number of classes must be provided for cnn model'
        assert hasattr(config,'kernel_size'), 'Kernel size must be provided for cnn model'
        
        return CNNClassifier(in_channels=config.in_channels, n_layers=config.n_layers, emb_dim=config.emb_dim, out_channels=config.out_channels,
                            n_classes=config.n_classes, kernel_size=config.kernel_size)
    elif arch == 'qcnn':
        assert hasattr(config,'n_wires'), 'Number of wires must be provided for qcnn model'
        assert hasattr(config,'n_layers'), 'Number of layers must be provided for qcnn model'
        assert hasattr(config,'n_classes'), 'Number of classes must be provided for qcnn model'
        assert hasattr(config,'ent_layers'), 'Number of entangling layers must be provided for qcnn model'

        return QCNNClassifier(n_wires=config.n_wires, n_layers=config.n_layers, 
                              ent_layers=config.ent_layers, n_classes=config.n_classes)
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
    if dataset in ['sst2', 'imdb', 'agnews']:
        embedding = NLTKEmbedder(weights_path = emb_path,  vocab_size=config.vocab_size)
        assert embedding.emb_dim == config.emb_dim, 'Embedding dimension mismatch'
    elif dataset in ['mnist2', 'cifar10', 'cifar2','fashion']:
        if arch == 'cnn': 
            embedding = PassEmbedder(device=device)
        else:
            embedding = FlattenEmbedder(device=device)


    # Load datasets
    all_datasets = build_dataset(dataset, config, test)
    config.n_classes = all_datasets.n_classes
    # Build model
    model = build_model(arch, config)
    # Load embeddings
    model.to(device)

    optimizer = build_optimizer(
        model, config)

    return model, all_datasets, optimizer, embedding


def build_train(arch, dataset, model_dir, emb_path, test, patience=5, save_test_predictions=False):
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
            torch.manual_seed(config._seed)
            do_dev_eval = True
            compute_test_metrics = True

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

            train_loader = all_datasets.train_loader
            dev_loader = all_datasets.dev_loader
            test_loader = all_datasets.test_loader
            do_dev_eval = not dev_loader.dataset[0][1] < 0
            compute_test_metrics = not test_loader.dataset[0][1] < 0

            # Define loss function and optimizer
            criterion = all_datasets.criterion

            print('Training...')
            counter = 0
            total_time = 0
            train_time = 0
            dev_eval_time = 0
            best_dev_loss = float('inf')
            for epoch in range(config.epochs):
                print(f'Epoch {epoch+1}/{config.epochs}')
                # Save current time
                start_epoch = time.time()

                cumu_labels = []
                cumu_outputs = []
                for batch in tqdm(train_loader):
                    data = batch[0]
                    labels = batch[1].to(device)
                    # Zero the gradients
                    optimizer.zero_grad()

                    inputs, seq_lengths = embedding(data)

                    # Forward pass
                    outputs, _ = model(inputs, seq_lengths)
                    loss, accuracy, f1 = metrics(criterion, model.n_classes, outputs, labels)
                    cumu_labels.append(labels)
                    cumu_outputs.append(outputs)

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
                    wandb.log({"batch loss": loss,
                               "batch accuracy": accuracy,
                               "batch f1": f1})
                print('Done.')

                # Compute metrics for the epoch
                epoch_labels = torch.cat(cumu_labels)
                epoch_outputs = torch.cat(cumu_outputs)
                train_loss, train_acc, train_f1 = metrics(criterion, model.n_classes, epoch_outputs, epoch_labels)
                print(f'Train loss: {train_loss}, Train accuracy: {train_acc}, Train F1: {train_f1}')

                # Log train runtime in minutes
                train_epoch = time.time()
                train_time += (train_epoch - start_epoch) / 60               
                    
                # Evaluate on dev set
                if do_dev_eval:
                    print('Evaluating on dev set...')
                    cumu_outputs = []
                    cumu_labels = []
                    with torch.no_grad():
                        for batch in tqdm(dev_loader):
                            data = batch[0]
                            labels = batch[1].type(torch.float).to(device)
                            inputs, seq_lengths = embedding(data)

                            # Forward pass
                            outputs, _ = model(inputs, seq_lengths)
                            loss, accuracy, f1 = metrics(criterion, model.n_classes, outputs, labels)
                            cumu_labels.append(labels)
                            cumu_outputs.append(outputs)

                    
                    # Compute metrics for the epoch
                    epoch_labels = torch.cat(cumu_labels)
                    epoch_outputs = torch.cat(cumu_outputs)
                    dev_loss, dev_acc, dev_f1 = metrics(criterion, model.n_classes, epoch_outputs, epoch_labels)
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
                    
                    if patience is not None and not test:
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
                
                else:
                    wandb.log({"epoch": epoch,
                            "loss": train_loss,
                            "train accuracy": train_acc,
                            "train F1": train_f1,
                            "train epoch time": train_time})
                    print('Skipping dev evaluation...')
            if test:
                # Evaluate on test set
                print('Evaluating on test set...')
                cumu_outputs = []
                cumu_labels = []
                with torch.no_grad():
                    for batch in tqdm(test_loader):
                        data = batch[0]
                        labels = batch[1].to(device)

                        inputs, seq_lengths = embedding(data)
                        outputs, _ = model(inputs, seq_lengths)
                        if compute_test_metrics:
                            loss, accuracy, f1 = metrics(criterion, model.n_classes, outputs, labels)
                        cumu_labels.append(labels)
                        cumu_outputs.append(outputs)
                # Compute metrics for the epoch
                epoch_labels = torch.cat(cumu_labels)
                epoch_outputs = torch.cat(cumu_outputs)
                if compute_test_metrics:

                    test_loss, test_acc, test_f1 = metrics(criterion, model.n_classes, epoch_outputs, epoch_labels)
                    print(f'Test loss: {test_loss}, Test accuracy: {test_acc}, Test F1: {test_f1}')
                    wandb.log({"test loss": test_loss, "test accuracy": test_acc, "test F1": test_f1})
                print('Done.')
                
                # Save outputs to tsv with pandas
                if save_test_predictions:
                    # Columns: index prediction
                    # If folder doesn't exist, create it
                    if not os.path.exists(f'data/{dataset}/'):
                        os.makedirs(f'data/{dataset}/')
                    predictions = (epoch_outputs > 0.5).type(torch.int).cpu().numpy()
                    pd.DataFrame({'index': range(len(test_loader.dataset)), 'prediction': predictions}) \
                    .to_csv(f'data/{dataset}/{arch}_{wandb.run.name}_test_predictions.tsv', sep='\t', index=False)

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


def infer(arch, dataset, model_name, model_dir, emb_path, test, save_test_predictions=False):
    '''
    Inference function
    '''
    # Finds device to run on
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on {device}')

    # Load model
    print('Loading model...')
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
        embedding = NLTKEmbedder(weights_path = emb_path)
        assert embedding.emb_dim == model_kwargs.emb_dim, 'Embedding dimension mismatch'
    elif dataset == 'mnist2':
        embedding = FlattenEmbedder()
    print('Done.')

    # Load dataset
    print('Loading dataset...')
    all_datasets = build_dataset(dataset, model_kwargs, test, batch_size=256)
    print('Done.')

    # Define loss function
    criterion = all_datasets.criterion
    
    if test:
        data_loader = all_datasets.test_loader
        print('Evaluating on test set...')
    else:
        data_loader = all_datasets.dev_loader
        print('Evaluating on dev set...')

    cumu_outputs = []
    cumu_labels = []
    with torch.no_grad():
        for batch in tqdm(data_loader):
            data = batch[0]
            labels = batch[1].type(torch.float).to(device)

            inputs, seq_lengths = embedding(data)
            outputs, _ = model(inputs, seq_lengths)
            cumu_outputs.append(outputs)
            if not save_test_predictions:
                # Compute metrics for the epoch
                loss, accuracy, f1 = metrics(criterion, model.n_classes, outputs, labels)
                cumu_labels.append(labels)


    print('Done.')

    print('Saving results...')
    epoch_outputs = torch.cat(cumu_outputs)
    # Save outputs to tsv with pandas
    if save_test_predictions:
        # Columns: index prediction
        # If folder doesn't exist, create it
        if not os.path.exists(f'data/{dataset}/'):
            os.makedirs(f'data/{dataset}/')
        predictions = (epoch_outputs > 0.5).type(torch.int).cpu().numpy()
        pd.DataFrame({'index': range(len(data_loader.dataset)), 'prediction': predictions}) \
        .to_csv(f'data/{dataset}/{arch}_{wandb.run.name}_test_predictions.tsv', sep='\t', index=False)
    else:
        # Compute metrics for the epoch
        epoch_labels = torch.cat(cumu_labels)
        loss, acc, f1 = metrics(criterion, model.n_classes, epoch_outputs, epoch_labels)
        print(f'Loss: {loss}, Accuracy: {acc}, F1: {f1}')

    print('Done.')

    del model
    del embedding
    torch.cuda.empty_cache()
    
    print('Current memory allocated: ', torch.cuda.memory_allocated())
