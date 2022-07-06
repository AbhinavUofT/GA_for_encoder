import numpy as np
import copy

def apply_generation_filter(sorted_circuit_str_l, generation_size, compression= False):
    """
    This function calculates the probabilities that the circuit should be kept or not
    using the Fermi-Dirac distribution and then returns the inst of indices of the circuits
    to keep and replace

    param: sorted_circuit_data_l (list <str>) -> a list containing the string representation of the circuits
                                        sorted according to their fitness value
    param: generation_size (int) -> number of circuits in a particular generation

    e.g.:
    input:
    sorted_circuit_data_l ->
    generation_size -> 10

    output: (these  might change based on the probabilities)
    to_keep ->
    to_replace ->
    """
    # Get the probabilities that a circuits with a given fitness will be replaced
    # a fermi function is used to smoothen the transition
    positions     = np.array(range(0, len(sorted_circuit_str_l))) - 0.5*float(len(sorted_circuit_str_l))
    probabilities = None
    if compression:
        to_replace = list(range(1,len(sorted_circuit_str_l)))# all circuits that are replaced
        to_keep    = [0]
        return to_replace, to_keep
    else:
        probabilities = 1.0 / (1.0 + np.exp(-0.1 * generation_size * positions / float(len(sorted_circuit_str_l))))
    """import matplotlib.pyplot as plt
    plt.plot(positions, probabilities)
    plt.show()
    raise Exception("test")"""
    to_replace = [] # all circuits that are replaced
    to_keep    = [] # all circuits that are kept
    for idx in range(0,len(sorted_circuit_str_l)):
        if np.random.rand(1) < probabilities[idx]:
            to_replace.append(idx)
        else:
            to_keep.append(idx)

    return to_replace, to_keep

def compress_circuit_str(circuit_data):
    """

    """
    new_data = copy.deepcopy(circuit_data)
    circuit_data = sort_by_qubits(circuit_data)

    #removing duplicates
    prev_data = circuit_data[0]
    prev_data = "".join(map(str, prev_data))
    for data in circuit_data[1:]:
        curr_data_str = "".join(map(str, data))
        if curr_data_str == prev_data:
            new_data.remove(data)
            #return True
        else:
            prev_data = curr_data_str

    return new_data, False

def sort_by_qubits(circuit_data):
    """

    """
    circuit_data = sorted(circuit_data, key=lambda item: item[0])
    return circuit_data

def pick_mutation(list_of_choices):
    """
    This function samples randomly from the list and return the corresponding string
    """
    #add more later
    mutation_list = ['add']
    full_mutation_l = ['add', 'replace', 'remove', 'repeat']
    compress_mutation_l = ['remove']
    choice = random_choice(list_of_choices)
    if choice == 0:
        return random_choice(mutation_list)
    elif choice == 1:
        return random_choice(full_mutation_l)
    else:
        return random_choice(compress_mutation_l)
