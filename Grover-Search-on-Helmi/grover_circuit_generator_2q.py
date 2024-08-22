from qiskit import QuantumCircuit, QuantumRegister
import numpy as np


def pauli_basis_measurement(circ, basis, qubit):
    """
    circ is a quantum circuit to which we want to add the basis
    basis is either x,y,z
    qubit is an integer
    """
    if basis == "x":
        circ.h(qubit)
    if basis == "y":
        # if general unitary is possible, then the following works, if not H followed by Sdg has the same effect
        # mat = (1 / np.sqrt(2)) * np.array([[1, -1j], [1, 1j]])
        # circ.unitary(matrix=mat, targets=[qubit])
        circ.si(qubit)
        circ.h(qubit)
    if basis == "z":
        pass
    return circ


# Grover Circuit Unencoded
def create_grover_unenc_circuit(bitstring, qubits=4):

    circ = QuantumCircuit(qubits)

    if qubits == 4: 
        [q0, q1, q2, q3] = [0, 1, 2, 3]
        circ.h([q0, q1, q2, q3])  # h layer to create an unstructured database
        circ.barrier()
    
        # Oracle 
        if bitstring == "11":
            circ.x([q0, q1, q2, q3])
        if bitstring == "01":
            circ.x([q2, q0])
        if bitstring == "10":
            circ.x([q3, q1])
        if bitstring == "00":
            #         "Nothing is added in this case"
            None
        circ.cz(q0, q1)
        circ.cz(q2, q3)

        if bitstring == "00":
            circ.x([q0, q1, q2, q3])
        if bitstring == "01":
            circ.x([q0, q2])
        if bitstring == "10":
            circ.x([q1, q3])
        if bitstring == "00":
            #         "Nothing is added in this case"
            None
        circ.barrier()

        # Amplitude inversion
        circ.h([q0, q1, q2, q3])
        circ.x([q0, q1, q2, q3])
        circ.cz(q0, q1)
        circ.cz(q2, q3)
        circ.x([q0, q1, q2, q3])
        circ.h([q0, q1, q2, q3])

    if qubits == 2:
        [q0, q1] = [0, 1]
        
        circ.h([q0, q1])  # h layer to create an unstructured database
        circ.barrier()
        
        
        # Oracle 
        if bitstring == "11":
            circ.x([q0, q1])
        if bitstring == "01":
            circ.x(q0)
        if bitstring == "10":
            circ.x(q1)
        if bitstring == "00":
            #         "Nothing is added in this case"
            None
        
        circ.cz(0, 1)
        
        if bitstring == "00":
            circ.x([q0, q1])
        if bitstring == "01":
            circ.x(q0)
        if bitstring == "10":
            circ.x(q1)
        if bitstring == "00":
            #         "Nothing is added in this case"
            None
        circ.barrier()

        # Amplitude inversion
        circ.h([q0, q1])
        circ.x([q0, q1])
        circ.cz(0, 1)
        circ.x([q0, q1])
        circ.h([q0, q1])
    return circ

# Grover Circuit Encoded
def encoding(circ):
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(0, 2)
    circ.cx(0, 3)
    circ.barrier()
    return circ


def decoding(circ):
    circ.cx(0, 3)
    circ.cx(0, 2)
    circ.cx(0, 1)
    circ.h(0)

    return circ


def add_syndrome_422(circ):
   
    circ.add_register(QuantumRegister(1, 'ancilla_x'))
    circ.add_register(QuantumRegister(1, 'ancilla_z'))
    # measure ZZZZ
    circ.cx(0, 4)
    circ.cx(1, 4)
    circ.cx(2, 4)
    circ.cx(3, 4)

    circ.barrier()
    # measure XXXX
    circ.h(0)
    circ.cx(0, 5)
    circ.h(0)
    
    circ.h(1)
    circ.cx(1, 5)
    circ.h(1)
    
    circ.h(2)
    circ.cx(2, 5)
    circ.h(2)
    
    circ.h(3)
    circ.cx(3, 5)
    circ.h(3)

    circ.barrier()
    
    return circ


def inversion(circ):
  
    
    circ.h([0, 1, 2, 3])  
    circ.barrier()
    
    circ = oracle(circ, "00")
    
    circ.h([0, 1, 2, 3])
    circ.barrier()
    
    return circ


def oracle(circ, bitstring):
    
    if bitstring == "00":
        circ.x(1)
        circ.x(2)
        circ.barrier()

        #circ.z(1)
        #circ.z(2)
        circ.rz(np.pi / 2, 0)
        circ.rz(np.pi / 2, 1)
        circ.rz(np.pi / 2, 2)
        circ.rz(np.pi / 2, 3)
        
        circ.barrier()
        circ.x(1)
        circ.x(2)
        circ.barrier()
        
    if bitstring == "01":
        circ.x(0)
        circ.x(2)
        circ.barrier()
        #circ.z(1)
        #circ.z(2)
        circ.rz(np.pi / 2, 0)
        circ.rz(np.pi / 2, 1)
        circ.rz(np.pi / 2, 2)
        circ.rz(np.pi / 2, 3)
        circ.barrier()
        circ.x(0)
        circ.x(2)
        circ.barrier()
        
    if bitstring == "10":
        circ.x(0)
        circ.x(1)
        circ.barrier()
        #circ.z(1)
        #circ.z(2)
        circ.rz(np.pi / 2, 0)
        circ.rz(np.pi / 2, 1)
        circ.rz(np.pi / 2, 2)
        circ.rz(np.pi / 2, 3)
        circ.barrier()
        circ.x(0)
        circ.x(1)
        circ.barrier()
        
    if bitstring == "11":
        circ.barrier()
        #circ.z(1)
        #circ.z(2)
        circ.rz(np.pi / 2, 0)
        circ.rz(np.pi / 2, 1)
        circ.rz(np.pi / 2, 2)
        circ.rz(np.pi / 2, 3)
        circ.barrier()
        
    return circ

def encoding_2(circ): 
    circ.h(0)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.cx(2, 3)
    circ.barrier()
    return circ

def decoding_2(circ): 
    circ.cx(2, 3)
    circ.cx(1, 2)
    circ.cx(0, 1)
    circ.h(0)
    return circ

def create_grover_enc_circuit(bitstring, enc_ver=1, add_syndrome=False):
  
    circ = QuantumCircuit(4)
    if enc_ver == 2: 
        circ = encoding_2(circ)
    else: 
        circ = encoding(circ)
    circ.h([0, 1, 2, 3])
    circ.barrier()
    circ = oracle(circ, bitstring)
    circ = inversion(circ)
    
    if add_syndrome:
        circ = add_syndrome_422(circ)
    
    if enc_ver == 2: 
        circ = decoding_2(circ)
    else:
        circ = decoding(circ)
    
    return circ