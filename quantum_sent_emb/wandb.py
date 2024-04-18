# Define the training loop
import os
import time
import wandb
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from quantum_sent_emb import HamiltonianClassifier
from quantum_sent_emb import Embedder
from quantum_sent_emb import CustomDataset

# wandb requires programmatically setting some components starting from simple string hyperparameters
# The following functions achieve this

def build_dataset(config, device, shuffle=True, eval_batch_size=256):
    # Access batch_size from kwargs
    assert hasattr(config,'batch_size'), 'Batch size must be provided for torch dataset'
    batch_size = config.batch_size

    # Loads dataset from hf
    datasets = load_dataset("sst2")
    # train_dataset = CustomDataset(datasets["train"].with_format("torch", device=device), data_key='sentence', label_key='label')
    # dev_dataset = CustomDataset(datasets["validation"].with_format("torch", device=device), data_key='sentence', label_key='label')
    # test_dataset = CustomDataset(datasets["test"].with_format("torch", device=device), data_key='sentence', label_key='label')

    # Dev and train are obtained from the train portion of datasets by sampling randomly
    train_dev_dataset = datasets["train"].train_test_split(test_size=0.1)
    train_dataset = CustomDataset(train_dev_dataset['train'], data_key='sentence', label_key='label')
    dev_dataset = CustomDataset(train_dev_dataset['test'], data_key='sentence', label_key='label')
    test_dataset = CustomDataset(datasets["validation"], data_key='sentence', label_key='label')


    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)
    dev_loader = DataLoader(
        dev_dataset, batch_size=eval_batch_size, shuffle=shuffle, num_workers=8)
    test_loader = DataLoader(
        test_dataset, batch_size=eval_batch_size, shuffle=shuffle, num_workers=8)
    return train_loader, dev_loader, test_loader

def build_model(arch, config):
    if arch == 'ham':
        assert hasattr(config,'emb_dim'), 'Embedding dimension must be provided for hamiltonian model'
        # assert hasattr(config,'hamiltonian'), 'Hamiltonian type must be provided for hamiltonian model'
        assert hasattr(config,'circ_in'), 'Type of circ input must be provided for hamiltonian model'
        assert hasattr(config,'bias'), 'Bias type must be provided for hamiltonian model'
        assert hasattr(config,'gates'), 'Gates must be provided for hamiltonian model'
        assert hasattr(config,'n_reps'), 'Number of repetitions must be provided for hamiltonian model'
        assert hasattr(config,'pos_enc'), 'Positional encoding must be provided for hamiltonian model'
        assert hasattr(config,'batch_norm'), 'Batch normalization must be provided for hamiltonian model'

        return HamiltonianClassifier(emb_dim=config.emb_dim, hamiltonian='pure', circ_in=config.circ_in, 
                                     bias=config.bias, gates=config.gates, n_reps=config.n_reps,
                                     pos_enc=config.pos_enc, batch_norm=config.batch_norm)
    elif arch == 'ham_weight':
        raise NotImplementedError('Weighted Hamiltonian model not yet implemented')
    elif arch == 'baseline':
        raise NotImplementedError('Baseline model not yet implemented')
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

def build_parameters(arch, emb_path, device, config):
    '''
    Builds model, datasets and optimizer
    '''
    # Load embedding here
    embedding = Embedder(weights_path = emb_path,  vocab_size=config.vocab_size)
    assert embedding.emb_dim == config.emb_dim, 'Embedding dimension mismatch'

    # Load datasets
    all_datasets = build_dataset(config, device)
    # Build model
    model = build_model(arch, config)
    # Load embeddings
    model.to(device)

    optimizer = build_optimizer(
        model, config)

    return model, all_datasets, optimizer, embedding


def build_train(arch, model_dir, emb_path, patience=5):
    def train(config=None):
        # Finds device to run on
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Running on {device}')

        # Initialize a new wandb run
        with wandb.init(config=config):#, Tensor.backend('pytorch'):
            config = wandb.config

            # Build model, datasets and optimizer
            print('Building model, datasets, optimizer and embedding...')
            model, all_datasets, optimizer, embedding = build_parameters(arch, emb_path, device=device, config=config)
            print('Done.')
            print(f'Now evaluating model: {model.kwargs}')
            
            n_params = model.get_n_params() # Dict containing n_params for every part 
            wandb.log(n_params)


            print('Sending model & embedding to device...')
            
            model = model.to(device)
            embedding = embedding.to(device)
            print('Done.')

            train_loader, test_loader, dev_loader = all_datasets

            # Define loss function and optimizer
            # criterion = nn.MSELoss()
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

                cumu_loss = 0
                cumu_corr = 0
                for batch in tqdm(train_loader):
                    data = batch['data']
                    labels = batch['label'].type(torch.float).to(device)
                    # Zero the gradients
                    optimizer.zero_grad()

                    inputs, seq_lengths = embedding(data)

                    # Forward pass
                    outputs, _ = model(inputs, seq_lengths)
                    loss = criterion(outputs, labels)
                    cumu_loss += loss.item()
                    cumu_corr += torch.sum((outputs > 0.5) == labels).item() 

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

                train_loss = cumu_loss / len(train_loader.dataset)
                train_acc = cumu_corr / len(train_loader.dataset)
                print(f'Train loss: {train_loss}, Train accuracy: {train_acc}')

                # Log train runtime in minutes
                train_epoch = time.time()
                train_time += (train_epoch - start_epoch) / 60               
                    
                # Evaluate on dev set
                print('Evaluating on dev set...')
                cumu_loss = 0
                cumu_corr = 0
                with torch.no_grad():
                    for batch in tqdm(dev_loader):
                        data = batch['data']
                        labels = batch['label'].type(torch.float).to(device)
                        inputs, seq_lengths = embedding(data)

                        # Forward pass
                        outputs, _ = model(inputs, seq_lengths)
                        loss = criterion(outputs, labels)
                        cumu_loss += loss.item()
                        cumu_corr += torch.sum((outputs > 0.5) == labels).item()
                
                dev_loss = cumu_loss / len(dev_loader.dataset)
                dev_acc = cumu_corr / len(dev_loader.dataset)
                print(f'Dev loss: {dev_loss}, Dev accuracy: {dev_acc}')
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
                           "train epoch time": train_time, 
                           "dev eval epoch time": dev_eval_time,
                           "total epoch time": total_time})
                
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
            cumu_loss = 0
            cumu_corr = 0
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    data = batch['data']
                    labels = batch['label'].type(torch.float).to(device)

                    inputs, seq_lengths = embedding(data)

                    # Forward pass
                    outputs, _ = model(inputs, seq_lengths)
                    loss = criterion(outputs, labels)
                    cumu_loss += loss.item()    
                    cumu_corr += torch.sum((outputs > 0.5) == labels).item()        
            print('Done.')
            
            # Log loss
            test_loss = cumu_loss / len(test_loader.dataset)
            test_acc = cumu_corr / len(test_loader.dataset)
            print(f'Test loss: {test_loss}, Test accuracy: {test_acc}')
            wandb.log({"test loss": test_loss,
                       "test accuracy": test_acc})

            # Save the best model
            save_path = os.path.join(model_dir, f'model_{arch}_{wandb.run.name}.pth')
            torch.save([model.kwargs, model.state_dict()], save_path)

            del model
            del embedding
            torch.cuda.empty_cache()
            print('Current memory allocated: ', torch.cuda.memory_allocated())

    return train
