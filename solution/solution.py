from typing import List, Tuple

import numpy as np
import cirq
import itertools


def PauliOperator(label):
    """Generates a specified Pauli operator

    Parameters:
        label (str): Pauli operator label (e.g. 'ZZZZIIIX')

    Returns:
        numpy matrix of corresponding Pauli operator
    """
    pauli = {
        'I': np.matrix([[1, 0], [0, 1]]),
        'Z': np.matrix([[1, 0], [0, -1]]),
        'X': np.matrix([[0, 1], [1, 0]]),
        'Y': np.matrix([[0, -1j], [1j, 0]])
    }

    operator = pauli[label[0]]
    for letter in label[1:]:
        operator = np.kron(operator, pauli[letter])

    return operator


#PauliOperator('XXX')

def PauliDecompose(hmat):
    """Decompose a Hermitian matrix into a sum of Pauli matrices

    Parameters:
        hmat (matrix): hermitian matrix to decompose

    Returns:
        dictionary of {Pauli matrix(str) : coefficient (float)}
    """
    coeff = {}
    nbits = int(np.log2(hmat.shape[0]))
    labels = itertools.product('IXYZ', repeat=nbits)
    labels = [''.join(i) for i in labels]
    for label in labels:
        tmp = np.matmul(hmat, PauliOperator(label))
        coeff[label] = np.trace(tmp) / hmat.shape[0]

    return coeff


# demo
#PauliDecompose(PauliOperator('Z') + PauliOperator('Z'))

##############################
#   CIRCUIT IMPLEMENTATION  #
#############################

def UGate(umat):
    """Realises the specified unitary digonal matrix in a Qiskit quantum cricuit

    Parameters:
        umat (matrix): unitary diagonal matrix to realise

    Returns:
        QuantumCircuit which implements the unitary
    """
    # Check input
    nbits = np.log2(umat.shape[0])
    if umat.shape[0] != umat.shape[1] or not nbits.is_integer:
        raise Exception('Matrix has incorrect dimensions')
    elif not np.allclose(np.matmul(umat, np.conj(umat)), np.identity(umat.shape[0])):
        raise Exception('Matrix is not unitary')
    elif not np.allclose(umat, np.diag(np.diagonal(umat))):
        raise Exception('matrix is not diagonal')
    nbits = int(nbits)  # n classical bits 1 < n < 8

    # Pauli Decompose
    hmat = np.angle(umat)  # Tells us the complex argument of the matrix, thereby parameterizing the circuit
    components = PauliDecompose(hmat)  # Decomposing the circuit according to the decomposing method

    # order to implement Pauli component (reduces CNOTs used)
    # iteratively add each pauli component

    # Define the qubits in Cirq
    circuit = cirq.Circuit()

    circuit.append(cirq.I(q) for q in cirq.LineQubit.range(nbits))

    for operator, coeff in components.items():
        # find qubits CX-RZ-CX
        cxlist = []
        for i in range(len(operator)):
            cxlist.append(i) if operator[i] == 'Z' else None
        cxlist = [nbits - 1 - i for i in cxlist]
        if coeff == 0 or len(cxlist) == 0:
            continue
        elif len(cxlist) == 1:
            circuit.append(cirq.ops.rz(-2 * coeff).on(cirq.LineQubit(cxlist[0])))
        else:
            for ctrl in cxlist[:-1]:
                circuit.append(cirq.ops.CNOT(control=ctrl, target=cirq.LineQubit(cxlist[:-1])))
            circuit.append(cirq.ops.rz(-2 * coeff).on(cirq.LineQubit(cxlist[-1])))
            for ctrl in reversed(cxlist[:-1]):
                circuit.append(cirq.ops.CNOT(control=ctrl, target=cirq.LineQubit(cxlist[:-1])))
    return circuit


# Testing!
#ckt = UGate(np.diag([-1, 1, -1, 1]))
#print(ckt)




def matrix_to_sycamore_operations(
    target_qubits: List[cirq.GridQubit], matrix: np.ndarray
) -> Tuple[cirq.OP_TREE, List[cirq.GridQubit]]:
    """A method to convert a unitary matrix to a list of Sycamore operations.

    This method will return a list of `cirq.Operation`s using the qubits and (optionally) ancilla
    qubits to implement the unitary matrix `matrix` on the target qubits `qubits`.
    The operations are also supported by `cirq.google.gate_sets.SYC_GATESET`.

    Args:
        target_qubits: list of qubits the returned operations will act on. The qubit order defined by the list
            is assumed to be used by the operations to implement `matrix`.
        matrix: a matrix that is guaranteed to be unitary and of size (2**len(qs), 2**len(qs)).
    Returns:
        A tuple of operations and ancilla qubits allocated.
            Operations: In case the matrix is supported, a list of operations `ops` is returned.
                `ops` acts on `qs` qubits and for which `cirq.unitary(ops)` is equal to `matrix` up
                 to certain tolerance. In case the matrix is not supported, it might return NotImplemented to
                 reduce the noise in the judge output.
            Ancilla qubits: In case ancilla qubits are allocated a list of ancilla qubits. Otherwise
                an empty list.
        .
    """
    return NotImplemented, []
