# Define the training loop
import os
import time
from lambeq import Dataset
import numpy as np
import pandas as pd
import wandb
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from quantum_composition.baseline import BaselineSentimentAnalyzer
from quantum_composition.disco import LambeqSentimentAnalyzer
from quantum_composition.quantum import QuantumSentimentAnalyzer
from quantum_composition.utils import unpickle_object

# wandb requires programmatically setting some components starting from simple string hyperparameters
# The following functions achieve this


def build_dataset(type, dataset_dir, shuffle=True, eval_batch_size=256, **kwargs):
    # Access batch_size and n_wires_type from kwargs
    # batch_size=64, , n_wires_type=1

    if type in ['torch_baseline', 'torch_bert', 'torch_quantum']:
        assert 'batch_size' in kwargs.keys(), 'Batch size must be provided for torch dataset'
        batch_size = kwargs['batch_size']

        # Loads dataset from pandas
        train_dataset = SentimentDataset.from_dataframe(
            pd.read_csv(os.path.join(dataset_dir, 'LEXICON_BG_train.csv')))
        dev_dataset = SentimentDataset.from_dataframe(
            pd.read_csv(os.path.join(dataset_dir, 'LEXICON_BG_dev.csv')))
        test_dataset = SentimentDataset.from_dataframe(
            pd.read_csv(os.path.join(dataset_dir, 'LEXICON_BG_test.csv')))

        # Create data loaders
        # , num_workers=8, pin_memory=True)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle)
        # , num_workers=8, pin_memory=True)
        dev_loader = DataLoader(
            dev_dataset, batch_size=eval_batch_size, shuffle=shuffle)
        # , num_workers=8, pin_memory=True)
        test_loader = DataLoader(
            test_dataset, batch_size=eval_batch_size, shuffle=shuffle)
        return train_loader, dev_loader, test_loader
    elif type == 'lambeq':
        assert 'n_wires_type' in kwargs.keys(
        ), 'Number of wires type must be provided for lambeq dataset'
        assert 'batch_size' in kwargs.keys(), 'Batch size must be provided for lambeq dataset'
        n_wires_type = kwargs['n_wires_type']
        batch_size = kwargs['batch_size']

        # Loads circuits from pickle
        train_circuits, train_labels = unpickle_object(os.path.join(
            dataset_dir, f'{n_wires_type}_train_circuits.pickle'))
        dev_circuits, dev_labels = unpickle_object(os.path.join(
            dataset_dir, f'{n_wires_type}_dev_circuits.pickle'))
        test_circuits, test_labels = unpickle_object(os.path.join(
            dataset_dir, f'{n_wires_type}_test_circuits.pickle'))

        # Create lambeq dataset
        # train_dataset = SentimentDataset.from_circuits(circuits=train_circuits, labels=train_labels)
        # dev_dataset = SentimentDataset.from_circuits(circuits=dev_circuits, labels=dev_labels)
        # test_dataset = SentimentDataset.from_circuits(circuits=test_circuits, labels=test_labels)

        # Create data loaders
        # , num_workers=8, pin_memory=True)
        train_loader = Dataset(train_circuits, train_labels,
                               batch_size=batch_size, shuffle=shuffle)
        # , num_workers=8, pin_memory=True)
        dev_loader = Dataset(dev_circuits, dev_labels,
                             batch_size=eval_batch_size, shuffle=shuffle)
        # , num_workers=8, pin_memory=True)
        test_loader = Dataset(test_circuits, test_labels,
                              batch_size=eval_batch_size, shuffle=shuffle)
        return train_loader, dev_loader, test_loader
    else:
        raise ValueError('Invalid dataset type.')


def build_model(type, emb_path, **kwargs):
    if type == 'torch_baseline':
        assert 'embedding_dim' in kwargs.keys(
        ), 'Embedding dimension must be provided for baseline model'
        assert 'classifier_size' in kwargs.keys(
        ), 'Classifier size must be provided for baseline model'
        assert 'limit' in kwargs.keys(), 'Limit must be provided for baseline model'
        embedding_dim = kwargs['embedding_dim']
        classifier_size = kwargs['classifier_size']
        limit = kwargs['limit']

        return BaselineSentimentAnalyzer(embedding_dim=embedding_dim, num_labels=2, dict_path=emb_path,
                                         classifier_size=classifier_size, limit=limit)
    elif type == 'torch_quantum':
        assert 'embedding_dim' in kwargs.keys(
        ), 'Embedding dimension must be provided for quantum model'
        assert 'classifier_size' in kwargs.keys(
        ), 'Classifier size must be provided for quantum model'
        assert 'limit' in kwargs.keys(), 'Limit must be provided for quantum model'
        assert 'resizer_depth' in kwargs.keys(
        ), 'Resizer depth must be provided for quantum model'
        embedding_dim = kwargs['embedding_dim']
        classifier_size = kwargs['classifier_size']
        limit = kwargs['limit']
        resizer_depth = kwargs['resizer_depth']

        return QuantumSentimentAnalyzer(embedding_dim=embedding_dim, num_labels=2, emb_path=emb_path, limit=limit,
                                        classifier_size=classifier_size, resizer_depth=resizer_depth)
    elif type == 'lambeq':
        assert 'circuits' in kwargs.keys(), 'Circuits must be provided for lambeq model'
        assert 'classifier_size' in kwargs.keys(
        ), 'Classifier size must be provided for lambeq model'
        circuits = kwargs['circuits']
        classifier_size = kwargs['classifier_size']

        model = LambeqSentimentAnalyzer(
            circuits=circuits, classifier_size=classifier_size, probabilities=True, normalize=True)
        model.initialise_weights()
        model = model.double()
        return model
    else:
        raise ValueError('Invalid model type.')


def build_optimizer(model, momentum=0.9, **kwargs):
    assert 'optimizer' in kwargs.keys(), 'Optimizer must be provided'
    assert 'learning_rate' in kwargs.keys(), 'Learning rate must be provided'
    optimizer = kwargs['optimizer']
    learning_rate = kwargs['learning_rate']

    if optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=learning_rate, momentum=momentum)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=learning_rate)
    return optimizer

# Combines the above functions to hide complexity from the train function


def build_parameters(type, dataset_dir, emb_path, device, config):
    all_datasets = None
    if type == 'lambeq':
        all_datasets = build_dataset(
            type, dataset_dir, batch_size=config.batch_size, n_wires_type=config.n_wires_type)
        train_dataset, dev_dataset, test_dataset = all_datasets
        all_circuits = train_dataset.data + dev_dataset.data + test_dataset.data
        model = build_model(type, emb_path=emb_path, circuits=all_circuits,
                            classifier_size=config.classifier_size)
        model = model.to(device)
    elif type == 'torch_baseline':
        all_datasets = build_dataset(
            type, dataset_dir, batch_size=config.batch_size)
        model = build_model(type, emb_path=emb_path, embedding_dim=config.embedding_dim, classifier_size=config.classifier_size,
                            limit=config.limit)
        model = model.to(device)
    elif type == 'torch_quantum':
        all_datasets = build_dataset(
            type, dataset_dir, batch_size=config.batch_size)
        model = build_model(type, emb_path=emb_path, embedding_dim=config.embedding_dim, classifier_size=config.classifier_size,
                            limit=config.limit, resizer_depth=config.resizer_depth)
        model = model.to(device)

    optimizer = build_optimizer(
        model, optimizer=config.optimizer, learning_rate=config.learning_rate)

    return model, all_datasets, optimizer


def build_train(type, dataset_dir, model_dir, emb_path, patience=5):
    def train(config=None, verbose=False):
        # Finds device to run on
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if verbose:
            print(f'Running on {device}')

        # Initialize a new wandb run
        with wandb.init(config=config):  # , Tensor.backend('pytorch'):
            config = wandb.config

            # Build model, datasets and optimizer
            if verbose:
                print('Building model, datasets and optimizer...')
            model, all_datasets, optimizer = build_parameters(
                type,
                dataset_dir,
                os.path.join(emb_path, config.emb_name),
                device, config)
            if verbose:
                print('Done.')
                print('Sending model to device...')
            model = model.to(device)
            if verbose:
                print('Done.')

            train_loader, test_loader, dev_loader = all_datasets

            # Define loss function and optimizer
            # Cross-entropy loss, works for both binary and multi-class classification
            criterion = nn.CrossEntropyLoss()

            total_time = 0
            train_time = 0
            dev_eval_time = 0
            best_dev_loss = float('inf')
            for epoch in range(config.epochs):
                print(f'Epoch {epoch+1}/{config.epochs}')
                # Save current time
                start_epoch = time.time()

                cumu_loss = 0
                correct = 0
                total = 0
                for i, data in tqdm(enumerate(train_loader, 0)):
                    if verbose:
                        print('Getting data...')
                    inputs, labels = data
                    labels = labels.to(device)
                    if verbose:
                        print('Done.')

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Forward pass
                    if verbose:
                        print('Forward pass...')
                    outputs = model(inputs)
                    if verbose:
                        print('Done.')
                        print('Computing loss...')
                    loss = criterion(outputs, labels.long())
                    if verbose:
                        print('Done.')
                    cumu_loss += loss.item()

                    # Compute accuracy
                    if verbose:
                        print('Computing accuracy...')
                    predicted = torch.argmax(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    if verbose:
                        print('Done.')

                    # Backward pass and optimization
                    if verbose:
                        print('Backward pass and optimization...')
                    loss.backward()
                    optimizer.step()
                    if verbose:
                        print('Done.')

                    # Log loss
                    wandb.log({"batch loss": loss.item()})

                train_loss = cumu_loss / len(train_loader)
                train_accuracy = correct / total

                # Log train runtime in minutes
                train_epoch = time.time()
                train_time += (train_epoch - start_epoch) / 60

                # Evaluate on dev set
                print('Evaluating on dev set...')
                cumu_loss = 0
                correct = 0
                total = 0
                with torch.inference_mode():  # Analogous to torch.no_grad()
                    for data in dev_loader:
                        inputs, labels = data
                        labels = labels.to(device)
                        # Forward pass
                        outputs = model(inputs)
                        loss = criterion(outputs, labels.long())
                        cumu_loss += loss.item()

                        # Compute accuracy
                        predicted = torch.argmax(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                dev_accuracy = correct / total
                dev_loss = cumu_loss / len(dev_loader)
                print('Done.')

                # Log evaluation runtime in minutes
                eval_epoch = time.time()
                dev_eval_time += (eval_epoch - train_epoch) / 60

                # Log total runtime including evaluation
                total_time += (eval_epoch - start_epoch) / 60
                wandb.log({"epoch": epoch,
                           "loss": train_loss,
                           "dev loss": dev_loss,
                           "train accuracy": train_accuracy,
                           "dev accuracy": dev_accuracy,
                           "train epoch time": train_time,
                           "dev eval epoch time": dev_eval_time,
                           "total epoch time": total_time})

                # Check if the current validation loss is better than the previous best loss
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    counter = 0  # Reset the counter since there's improvement
                else:
                    counter += 1  # Increment the counter as no improvement occurred

                # Check if the counter has exceeded the patience limit
                if counter >= patience:
                    print("Early stopping: No improvement in validation loss for {} epochs.".format(
                        patience))
                    break  # Exit the training loop

            # Evaluate on test set
            cumu_loss = 0
            correct = 0
            total = 0
            with torch.inference_mode():  # Analogous to torch.no_grad()
                for data in test_loader:
                    inputs, labels = data
                    labels = labels.to(device)
                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.long())
                    cumu_loss += loss.item()

                    # Compute accuracy
                    predicted = torch.argmax(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            test_accuracy = correct / total

            # Log loss
            n_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
            wandb.log({"test loss": cumu_loss / len(test_loader),
                      "n_params": n_params, "test accuracy": test_accuracy})

            # Save the best model
            save_path = os.path.join(model_dir, f'model_{type}_{
                                     wandb.run.name}.pth')
            torch.save([model.kwargs, model.state_dict()], save_path)

    return train
