import tequila as tq
import qutip as qt
import numpy as np
import itertools

from difflib import SequenceMatcher

def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def calculate_similarity(list_of_circuit_data, new_data):
    """
    This function calculates the similarity between

    param: list_of_circuit_data (list <str>) ->
    param: new_data (str) ->

    e.g.:
    input:
    list_of_circuit_data ->
    new_data ->

    output:
    similarity ->
    """
    similarity = []
    max_similarity = 0.0
    new_data_str = "".join(map(str, new_data))
    for circ_d1 in list_of_circuit_data:
        #print(circ_d1)
        similarity.append(similar("".join(map(str,circ_d1)), new_data_str))
    return similarity

def get_density_matrix(dimension, dims, wavefunction):
    """
    This function returns the density matrix of a pure state wavefunction

    param: dimension (int) -> the dimension of the Hilbert space
    param: dims(list) -> the dimension of the qutip Qobj
    param: wavefunction (tq.QubitWaveFunction) -> the tequila wavefunction object

    returns:
    density (qutip.Qobj) -> the density matrix corresponding to the wavefunction

    e.g.:
    input:
    dimension -> 8
    dims -> [[2]*3]*2
    wavefunction -> +0.7070|000> +0.7070|111>

    output:
    Quantum object: dims = [[2, 2, 2], [2, 2, 2]], shape = (8, 8), type = oper, isherm = True
                    Qobj data =
                    [[0.5 0.  0.  0.  0.  0.  0.  0.5]
                     [0.  0.  0.  0.  0.  0.  0.  0. ]
                     [0.  0.  0.  0.  0.  0.  0.  0. ]
                     [0.  0.  0.  0.  0.  0.  0.  0. ]
                     [0.  0.  0.  0.  0.  0.  0.  0. ]
                     [0.  0.  0.  0.  0.  0.  0.  0. ]
                     [0.  0.  0.  0.  0.  0.  0.  0. ]
                     [0.5 0.  0.  0.  0.  0.  0.  0.5]]
    """
    state = None
    wavefunction = convert_wavefunction_format(wavefunction)
    for ind, key in enumerate(wavefunction.keys()):
        if ind == 0:
            state = qt.states.basis(dimension, int(key)) * complex(wavefunction[key])
        else:
            state += qt.states.basis(dimension, int(key)) * complex(wavefunction[key])
    state = state.unit()
    density = qt.Qobj(state * state.dag(), dims=dims)
    return density

def convert_wavefunction_format(wavefunction):
    """
    This function converts the tequila wavefunction object into a dictionary

    param: wavefunction (tq.QubitWaveFunction) -> the tequila wavefunction object

    returns:
    wfn (dict) -> the corresponding dictionary with decimal form of the basis sets as
                the key and amplitudes as the value

    e.g.:
    input:
    wavefunction -> +0.7070|000> +0.7070|111>

    output:
    wfn -> {'0': 0.707, '7': 0.707}
    """
    wfn = {}
    #print(wavefunction.to_array())
    for ind, value in wavefunction.items():
        wfn.update({str(ind):value})
    #print(wavefunction, wfn)
    return wfn

def get_wavefunction_partial_trace(dimension, dims, wavefunction, qubit_set):
    """
    This function returns the reduced density matrix of the wavefunction after tracing out
    the qubits other than the one in "qubit_set"

    param: dimension (int) -> the dimension of the Hilbert space
    param: dims (list) -> the dimension of the qutip Qobj
    param: wavefunction (tq.QubitWaveFunction) -> the tequila wavefunction object
    param: qubit_set (list) -> the list of qubits remaining after tracing

    returns:
    density (qutip.Qobj) -> the reduced ensity matrix corresponding to the wavefunction

    e.g.:
    input:
    dimension -> 8
    dims ->  [[2]*3, [1]*3]
    wavefunction ->  +0.7070|000> +0.7070|111>
    qubit_set -> [1,2]

    output:
    density -> Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
               Qobj data =
                [[0.5 0.  0.  0. ]
                 [0.  0.  0.  0. ]
                 [0.  0.  0.  0. ]
                 [0.  0.  0.  0.5]]
    """
    state = None
    wavefunction = convert_wavefunction_format(wavefunction)
    for ind, key in enumerate(wavefunction.keys()):
        if ind == 0:
            state = qt.states.basis(dimension, int(key)) * complex(wavefunction[key])
        else:
            state += qt.states.basis(dimension, int(key)) * complex(wavefunction[key])
    state = qt.Qobj(state.unit(),dims=dims)
    return state.ptrace(qubit_set)

def get_reduced_density_matrix(density, qubit_set):
    """
    This function returns the reduced density matrix of the density matrix after tracing out
    the qubits other than the one in "qubit_set"

    param: density (qutip.Qobj) -> the density matrix
    param: qubit_set (list) -> the list of qubits remaining after tracing

    returns:
    density (qutip.Qobj) -> the reduced ensity matrix corresponding to the wavefunction

    e.g.:
    input:
    density -> Quantum object: dims = [[2, 2, 2], [2, 2, 2]], shape = (8, 8), type = oper, isherm = True
               Qobj data =
                [[0.5 0.  0.  0.  0.  0.  0.  0.5]
                 [0.  0.  0.  0.  0.  0.  0.  0. ]
                 [0.  0.  0.  0.  0.  0.  0.  0. ]
                 [0.  0.  0.  0.  0.  0.  0.  0. ]
                 [0.  0.  0.  0.  0.  0.  0.  0. ]
                 [0.  0.  0.  0.  0.  0.  0.  0. ]
                 [0.  0.  0.  0.  0.  0.  0.  0. ]
                 [0.5 0.  0.  0.  0.  0.  0.  0.5]]
    qubit_set -> [1,2]

    output:
    density -> Quantum object: dims = [[2, 2], [2, 2]], shape = (4, 4), type = oper, isherm = True
               Qobj data =
                [[0.5 0.  0.  0. ]
                 [0.  0.  0.  0. ]
                 [0.  0.  0.  0. ]
                 [0.  0.  0.  0.5]]
    """
    return density.ptrace(qubit_set)

def get_1_rdm_distance(rdm1, rdm2, qubits, quadratic=True):
    """
    This function calculates the distance between the all possible 1-rdms of the wavefunctions
    and return the sum or the sum of the squared distance

    param: rdm1
    param: rdm2
    param: qubits (list) -> the list of qubits
    param: quadratic (bool) -> a boolean values suggesting if to return the sum of the squared distance or not

    returns:
    distance (float) -> the sum of the "distance" or "squared distance" b/w all the 1-rdms

    e.g:
    input:
    rdm1 ->
    rdm2 ->
    qubits -> [0,1,2]
    quadratic -> False

    output:
    distance -> 1.0000000000000002
    """

    rdm_distance = 0.0
    for qubit in qubits:
        t_rdm1 = get_reduced_density_matrix(rdm1, [qubit])
        t_rdm2 = get_reduced_density_matrix(rdm2, [qubit])
        #print(rdm1, rdm2)
        diff = qt.Qobj(np.abs((t_rdm1 - t_rdm2).data))
        #print(diff)
        if quadratic:
            rdm_distance += (diff.tr())**2
        else:
            rdm_distance += diff.tr()
    return rdm_distance

def get_2_rdm_distance(rdm1, rdm2, qubits, quadratic=True):
    """
    This function calculates the distance between the all possible 2-rdms of the wavefunctions
    and return the sum or the sum of the squared distance

    param: rdm1
    param: rdm2
    param: qubits (list) -> the list of qubits
    param: quadratic (bool) -> a boolean values suggesting if to return the sum of the squared distance or not

    returns:
    distance (float) -> the sum of the "distance" or "squared distance" b/w all the 2-rdms

    e.g:
    input:
    rdm1 ->
    rdm2 ->
    qubits -> [0,1,2]
    quadratic -> False

    output:
    distance -> 2.0
    """

    rdm_distance = 0.0
    if len(qubits) > 2:
        all_comb = list(itertools.combinations(qubits, 2))
        for qubit in all_comb:
            t_rdm1 = get_reduced_density_matrix(rdm1, list(qubit))
            t_rdm2 = get_reduced_density_matrix(rdm2, list(qubit))
            #print(rdm1, rdm2)
            diff = qt.Qobj(np.abs((t_rdm1 - t_rdm2).data))
            #print(diff)
            if quadratic:
                rdm_distance += (diff.tr())**2
            else:
                rdm_distance += diff.tr()
    else:
        diff = qt.Qobj(np.abs((rdm1 - rdm2).data))
        #print(diff)
        if quadratic:
            rdm_distance += (diff.tr())**2
        else:
            rdm_distance += diff.tr()
    return rdm_distance

def get_infidelity(rdm1, rdm2):
    """
    This function calculates the infidelity between the two wavefunctions

    param: rdm1
    param: rdm2

    returns:
    infidelity (float) -> the infidelity value

    e.g.:
    input:
    rdm1 ->
    rdm2 ->

    output:
    infidelity ->
    """

    return 1 - qt.metrics.fidelity(rdm1, rdm2)


g_seed = int(np.abs(np.random.randn(1)[0]*100) + 1) #random seed

def random_choice(list_val, size=1, replace=False, prob = None):
    """
    This function calls the numpy.random.choice function with
    a new seed every time, and returns the values
    """
    if len(list_val) == 1:
        return list_val[0]
    else:
        global g_seed
        rand_state = np.random.RandomState(seed = g_seed)
        g_seed += int(np.abs(np.random.randn(1)[0]*10) + 1)
        if prob == None:
            choice = rand_state.choice(a=list_val, size=size, replace=replace)
        else:
            choice = rand_state.choice(a=list_val, p=prob, size=size, replace=replace)
        if size == 1:
            return choice[0]
        else:
            return choice

def random_uniform(low, high):
    """
    This function calls the numpy.random.uniform function with
    a new seed every time, and returns the values
    """
    global g_seed
    rand_state = np.random.RandomState(seed = g_seed)
    g_seed += int(np.abs(np.random.randn(1)[0]*10) + 1)
    choice = rand_state.uniform(low=low, high=high)
    return choice

def random_permutation(list_val):
    """
    This function calls the numpy.random.permutation function with
    a new seed every time, and returns the values
    """
    global g_seed
    rand_state = np.random.RandomState(seed = g_seed)
    g_seed += int(np.abs(np.random.randn(1)[0]*10) + 1)
    return rand_state.permutation(list_val)
