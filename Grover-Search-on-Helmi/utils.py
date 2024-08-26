from qiskit import transpile, QuantumCircuit
import numpy as np
import os
from datetime import datetime
import grover_circuit_generator_2q as circ_gen

import matplotlib.pyplot as plt
import seaborn as sns

def get_backend_name(backend):
    if backend.name == 'IQMFakeAdonisBackend':
        return 'Helmi simulator'
    elif backend.name == 'aer_simulator':
        return 'Ideal Simulator'
    else:
        return 'Helmi quantum computer'

def transpile_circuit(qc, backend, qubit_order=['QB3', 'QB1', 'QB2', 'QB4'], n_qubits=4, opt_lvl = 3):
    if backend.name == 'aer_simulator':
        transpiled_qc = transpile(qc, backend)
    else: 
        basis_gates = backend.operation_names
        initial_layout = [backend.qubit_name_to_index(name) for name in qubit_order]
        coupling_map = backend.coupling_map
        
        filtered_layout = initial_layout[0:n_qubits]
        qubits_to_remove = [q for q in initial_layout if not q in filtered_layout]
        filtered_map = [list(item) for item in coupling_map if not any(qubit in item for qubit in qubits_to_remove)]

        transpiled_qc = transpile(qc, layout_method='sabre', optimization_level=opt_lvl, basis_gates=basis_gates, initial_layout=filtered_layout, coupling_map = filtered_map)
  
    return transpiled_qc

def run_a_circuit(qc, backend, type, shots):
    """
        backend = [ideal_simulator, backend, backend_fake]
        type = ['unencoded', 'encoded']
    """
    if qc.num_qubits == 4:
        transpiled_circ = transpile_circuit(qc, backend)
    else: 
        transpiled_circ = transpile(qc, backend)
    job = backend.run(transpiled_circ, shots=shots)
    result = job.result()
    counts = result.get_counts()
    reversed_counts = {key[::-1]: value for key, value in counts.items()}
    plot_title =  type + ' circuit run on ' + get_backend_name(backend)
    
    return reversed_counts, plot_title


def run_and_get_accuracy_map(dict, backend, type, shots):
    """
        dict: dictionary of oracle marked states and their corresponding measurement bitstrings
        backend = [ideal_simulator, backend, backend_fake]
        type = ['unencoded', 'encoded']
    """
    accuracy_map = {}
    
    for bitstring in dict.keys():
    
        if type == 'encoded':
            qc = circ_gen.create_grover_enc_circuit(bitstring)
        elif type == 'encoded_2':
            qc = circ_gen.create_grover_enc_circuit(bitstring, enc_ver=2)
        else:
            qc = circ_gen.create_grover_unenc_circuit(bitstring)

        qc.measure_all()
          
        counts, plot_title = run_a_circuit(qc, backend, type, shots)
        accuracy = counts.get(dict[bitstring], 0) / shots
        accuracy_map[bitstring] = accuracy
        
    return accuracy_map, plot_title


def plot_heatmap(accuracy_map_1, title1, accuracy_map_2=None, title2=None, figsize=(12, 6)):
    dir = "./heatmap"
    os.makedirs(dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{current_time}.png"
    
    bitstrings = ['00', '01', '10', '11']
    
    # Initialize matrices with zeros
    accuracy_matrix_1 = np.zeros((len(bitstrings), len(bitstrings)))
    accuracy_matrix_2 = np.zeros((len(bitstrings), len(bitstrings))) if accuracy_map_2 else None

    # Fill the matrices with accuracy values
    for i, bitstring in enumerate(bitstrings):
        accuracy_matrix_1[i, i] = accuracy_map_1.get(bitstring, 0)
        if accuracy_map_2:
            accuracy_matrix_2[i, i] = accuracy_map_2.get(bitstring, 0)

    if accuracy_map_2:
        # Plotting the heatmaps side by side
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        sns.heatmap(accuracy_matrix_1, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=bitstrings, yticklabels=bitstrings, vmin=0, vmax=1, ax=axes[0])
        axes[0].set_title(title1)
        axes[0].set_xlabel('Detected State')
        axes[0].set_ylabel('Oracle Marked State')

        sns.heatmap(accuracy_matrix_2, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=bitstrings, yticklabels=bitstrings, vmin=0, vmax=1, ax=axes[1])
        axes[1].set_title(title2)
        axes[1].set_xlabel('Detected State')
        axes[1].set_ylabel('Oracle Marked State')

        plt.tight_layout()
    else:
        # Plotting a single heatmap
        plt.figure(figsize=(figsize[0] / 2, figsize[1]))
        sns.heatmap(accuracy_matrix_1, annot=True, fmt=".2f", cmap="YlGnBu", xticklabels=bitstrings, yticklabels=bitstrings, vmin=0, vmax=1)
        plt.title(title1)
        plt.xlabel('Detected State')
        plt.ylabel('Oracle Marked State')
        plt.tight_layout()

    plt.savefig(os.path.join(dir, filename))
    plt.show()
    
def plot_bargraph(dict1, label1, dict2, label2, title, xlabel='Oracle Marked State', ylabel='Accuracy', figsize=(8, 6)):
    
    bitstrings = list(dict1.keys())
    list1 = [dict1[bitstring] for bitstring in bitstrings]
    list2 = [dict2[bitstring] for bitstring in bitstrings]

    x = np.arange(len(bitstrings))  # the label locations
    width = 0.35  # the width of the bars

    # Plotting
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width/2, list1, width, label=label1)
    ax.bar(x + width/2, list2, width, label=label2)

    # Labeling
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(bitstrings)
    ax.legend()

    fig.tight_layout()
    
    dir = "./bar_graph"
    os.makedirs(dir, exist_ok=True)
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{current_time}.png"
    
    plt.savefig(os.path.join(dir, filename))
    plt.show()