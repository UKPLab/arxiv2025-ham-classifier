# This file contains experimental configurations
# TODO: refactor as JSON file

# Configuration for all sweeps
sweep_global = { 
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

    'vocab_size' : {
        'value': None
        },

    }

sweep_ham = {
    'circ_in': {
        'values': ['sentence','zeros']
        },
    'bias': {
        'values': ['matrix', 'vector', 'none'] #'diag', 'single',
        },
    'batch_norm': {
        'value': True
        },
    'pos_enc': {
        'values': ['learned','none']
        },
    'gates': {
        'values': [#['ry', 'rz', 'cnot_ring', 'ry','rz'], # Proposed in qiskit's EfficientSU2
                    ['rx', 'rz', 'crx_all_to_all', 'rx', 'rz'], # Circuit 6 of Sim et al 2019
                    #['rx', 'rz', 'crz_all_to_all', 'rx', 'rz'], # Circuit 5 of Sim et al 2019
                    #['ry', 'crz_ring', 'ry', 'crz_ring'], # Circuit 13 of Sim et al 2019
                    ['ry', 'crx_ring', 'ry', 'crx_ring'], # Circuit 14 of Sim et al 2019
                    ['rx', 'ry', 'rz'], # Control circuit without entanglement
                    #['i'] # Control empty circuit 
                    ]
        },
    'n_reps': {
        'values': [8, 16, 32]
        },
}

sweep_ham_peffbias = {
    'circ_in': {
        'value': 'zeros'
        },
    'bias': {
        'value': 'vector' #'diag', 'single',
        },
    'batch_norm': {
        'value': True
        },
    'pos_enc': {
        'value': 'none'
        },
    'gates': {
        'values': [#['ry', 'rz', 'cnot_ring', 'ry','rz'], # Proposed in qiskit's EfficientSU2
                    ['rx', 'rz', 'crx_all_to_all', 'rx', 'rz'], # Circuit 6 of Sim et al 2019
                    #['rx', 'rz', 'crz_all_to_all', 'rx', 'rz'], # Circuit 5 of Sim et al 2019
                    #['ry', 'crz_ring', 'ry', 'crz_ring'], # Circuit 13 of Sim et al 2019
                    ['ry', 'crx_ring', 'ry', 'crx_ring'], # Circuit 14 of Sim et al 2019
                    ['rx', 'ry', 'rz'], # Control circuit without entanglement
                    #['i'] # Control empty circuit 
                    ]
        },
    'n_reps': {
        'values': [8, 16, 32]
        },
}

sweep_circ = {
    'bias': {
        'values': ['vector','none']
        },
    'pos_enc': {
        'value': 'none'
        },
    'gates': {
        'values': [#['ry', 'rz', 'cnot_ring', 'ry','rz'], # Proposed in qiskit's EfficientSU2
                    ['rx', 'rz', 'crx_all_to_all', 'rx', 'rz'], # Circuit 6 of Sim et al 2019
                    #['rx', 'rz', 'crz_all_to_all', 'rx', 'rz'], # Circuit 5 of Sim et al 2019
                    #['ry', 'crz_ring', 'ry', 'crz_ring'], # Circuit 13 of Sim et al 2019
                    ['ry', 'crx_ring', 'ry', 'crx_ring'], # Circuit 14 of Sim et al 2019
                    ['rx', 'ry', 'rz'], # Control circuit without entanglement
                    #['i'] # Control empty circuit 
                    ]
        },
    'n_reps': {
        'values': [8, 16, 32]
        },
    'clas_type' : {
        'values': ['tomography', 'hamiltonian']
        },
}

# Shared between rnn and lstm
sweep_rnn = {
    'hidden_dim': {
        'value': 100
        },
    'rnn_layers': {
        'values': [1, 2]
        },
}

sweep_mlp = {
    'n_layers': {
        'values': [3, 4, 5]
        },
    'hidden_dim': {
        'values': [100, 300, 500]
        },
}



# Configuration for best models
run_ham = { 
    'optimizer': {
        'value': 'adam'
        },
    'learning_rate': {
        'value': 1e-3
        },
    'batch_size': {
        'value': 128
        },
    'epochs': {
        'value': 30
        },
    'emb_dim': {
        'value': 300 
        },
    'vocab_size' : {
        'value': None
        },
    'circ_in': {
        'value': 'zeros'
        },
    'bias': {
        'value': 'matrix'
        },
    'batch_norm': {
        'value': True
        },
    'pos_enc': {
        'value': 'none'
        },
    'gates': {
        'value':    ['ry', 'crx_ring', 'ry', 'crx_ring'] # Circuit 14 of Sim et al 2019
        },
    'n_reps': {
        'value': 16
        },
}

run_circ = { 
    'optimizer': {
        'value': 'adam'
        },
    'learning_rate': {
        'value': 1e-4
        },
    'batch_size': {
        'value': 64
        },
    'epochs': {
        'value': 30
        },
    'emb_dim': {
        'value': 300 
        },

    'vocab_size' : {
        'value': None
        },
    'bias': {
        'value': 'vector'
        },
    'pos_enc': {
        'value': 'none'
        },
    'gates': {
        'value':    ['ry', 'crx_ring', 'ry', 'crx_ring'], # Circuit 14 of Sim et al 2019
        },
    'n_reps': {
        'value': 32
        },
    'clas_type' : {
        'value': 'tomography'
        },
}

run_rnn = { 
    'optimizer': {
        'value': 'adam'
        },
    'learning_rate': {
        'value': 1e-4
        },
    'batch_size': {
        'value': 256
        },
    'epochs': {
        'value': 30
        },
    'emb_dim': {
        'value': 300 
        },

    'vocab_size' : {
        'value': None
        },
    'hidden_dim': {
        'value': 100
        },
    'rnn_layers': {
        'value': 1
        },
}

run_lstm = { 
    'optimizer': {
        'value': 'adam'
        },
    'learning_rate': {
        'value': 1e-2
        },
    'batch_size': {
        'value': 64
        },
    'epochs': {
        'value': 30
        },
    'emb_dim': {
        'value': 300 
        },

    'vocab_size' : {
        'value': None
        },
    'hidden_dim': {
        'value': 100
        },
    'rnn_layers': {
        'value': 1
        },
}

run_bow = { 
    'optimizer': {
        'value': 'adam'
        },
    'learning_rate': {
        'value': 1e-2
        },
    'batch_size': {
        'value': 64
        },
    'epochs': {
        'value': 30
        },
    'emb_dim': {
        'value': 300 
        },

    'vocab_size' : {
        'value': None
        },
}

run_mlp = {
    'optimizer': {
        'value': 'adam'
        },
    'learning_rate': {
        'value': 1e-4
        },
    'batch_size': {
        'value': 256
        },
    'epochs': {
        'value': 30
        },
    'emb_dim': {
        'value': 300 
        },

    'vocab_size' : {
        'value': None
        },
    'n_layers': {
        'value': 5
        },
    'hidden_dim': {
        'value': 100
        },
}

run_ablation_peffbias = { 
    'optimizer': {
        'value': 'adam'
        },
    'learning_rate': {
        'value': 1e-3
        },
    'batch_size': {
        'value': 128
        },
    'epochs': {
        'value': 30
        },
    'emb_dim': {
        'value': 300 
        },
    'vocab_size' : {
        'value': None
        },
    'circ_in': {
        'value': 'zeros'
        },
    'bias': {
        'value': 'vector'
        },
    'batch_norm': {
        'value': True
        },
    'pos_enc': {
        'value': 'none'
        },
    'gates': {
        'value':    ['ry', 'crx_ring', 'ry', 'crx_ring'] # Circuit 14 of Sim et al 2019
        },
    'n_reps': {
        'value': 16
        },
}

run_ablation_nobias = { 
    'optimizer': {
        'value': 'adam'
        },
    'learning_rate': {
        'value': 1e-3
        },
    'batch_size': {
        'value': 128
        },
    'epochs': {
        'value': 30
        },
    'emb_dim': {
        'value': 300 
        },
    'vocab_size' : {
        'value': None
        },
    'circ_in': {
        'value': 'zeros'
        },
    'bias': {
        'value': 'none'
        },
    'batch_norm': {
        'value': True
        },
    'pos_enc': {
        'value': 'none'
        },
    'gates': {
        'value':    ['ry', 'crx_ring', 'ry', 'crx_ring'] # Circuit 14 of Sim et al 2019
        },
    'n_reps': {
        'value': 16
        },
}

run_ablation_sentin = { 
    'optimizer': {
        'value': 'adam'
        },
    'learning_rate': {
        'value': 1e-3
        },
    'batch_size': {
        'value': 128
        },
    'epochs': {
        'value': 30
        },
    'emb_dim': {
        'value': 300 
        },
    'vocab_size' : {
        'value': None
        },
    'circ_in': {
        'value': 'sentence'
        },
    'bias': {
        'value': 'matrix'
        },
    'batch_norm': {
        'value': True
        },
    'pos_enc': {
        'value': 'none'
        },
    'gates': {
        'value':    ['ry', 'crx_ring', 'ry', 'crx_ring'] # Circuit 14 of Sim et al 2019
        },
    'n_reps': {
        'value': 16
        },
}

run_ablation_circham = { 
    'optimizer': {
        'value': 'adam'
        },
    'learning_rate': {
        'value': 1e-4
        },
    'batch_size': {
        'value': 64
        },
    'epochs': {
        'value': 30
        },
    'emb_dim': {
        'value': 300 
        },

    'vocab_size' : {
        'value': None
        },
    'bias': {
        'value': 'vector'
        },
    'pos_enc': {
        'value': 'none'
        },
    'gates': {
        'value':    ['ry', 'crx_ring', 'ry', 'crx_ring'], # Circuit 14 of Sim et al 2019
        },
    'n_reps': {
        'values': 32
        },
    'clas_type' : {
        'values': 'hamiltonian'
        },
}

run_ablation_hamhad = { 
    'optimizer': {
        'value': 'adam'
        },
    'learning_rate': {
        'value': 1e-3
        },
    'batch_size': {
        'value': 128
        },
    'epochs': {
        'value': 30
        },
    'emb_dim': {
        'value': 300 
        },
    'vocab_size' : {
        'value': None
        },
    'circ_in': {
        'value': 'hadamard'
        },
    'bias': {
        'value': 'matrix'
        },
    'batch_norm': {
        'value': True
        },
    'pos_enc': {
        'value': 'none'
        },
    'gates': {
        'value':    ['ry', 'crx_ring', 'ry', 'crx_ring'] # Circuit 14 of Sim et al 2019
        },
    'n_reps': {
        'value': 16
        },
}