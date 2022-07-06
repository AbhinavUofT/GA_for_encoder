import numpy as np
import tequila as tq
import copy, random
from general_utils import *
from GA_utils import *
import multiprocessing

def generate_next_generation_circuits(encoder_obj, circuit_data_dict, keep_list, replace_list, list_of_choices, all_circuit_str):
    """
    This function generates the next generation of circuits by replacing the circuits to be
    removed with mutations of circuits from the keep list

    param: circuit_data_dict (dict) ->
    param: keep_list (list) -> a list of indices of circuits to keep in the next generation
    param: replace_list (list) -> a list of indices of circuits to replace in the next generation
    param: list_of_choices (list) ->

    e.g.:
    input:
    circuit_data_dict ->
    keep_list ->
    replace_list ->
    list_of_choices ->

    output:

    """
    next_gen_circuits = {}
    keys_l = list(circuit_data_dict.keys())
    for index in keep_list:
        next_gen_circuits[keys_l[index]] = copy.deepcopy(circuit_data_dict[keys_l[index]])

    for index in replace_list:
        sim = [1.0]
        mutated_circuit_data = None
        done = True
        while done == True:
            random_index = random_choice(keep_list)
            mutated_circuit_data = apply_random_mutation(encoder_obj, circuit_data_dict[keys_l[random_index]], list_of_choices)
            if len(next_gen_circuits) == 0:
                sim = [0.0]
            else:
                sim = calculate_similarity(next_gen_circuits.values(), mutated_circuit_data)
            mutated_circuit_data, repetition = compress_circuit_str(mutated_circuit_data)
            if mutated_circuit_data not in all_circuit_str:
                if np.mean(sim) <= 0.75 and np.median(sim) <= 0.75 and max(sim) < 0.9 and repetition == False:
                    done = False

        next_gen_circuits[keys_l[index]] = mutated_circuit_data

    return next_gen_circuits


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


def apply_random_mutation(encoder_obj, circuit_data, list_of_choices):
    """

    """
    mutation = pick_mutation(list_of_choices)
    success = False
    while  not success:
        if mutation == "add":
            circuit_data.append(encoder_obj.sample_connection())
            success = True
        elif mutation == "replace":
            choice = random_choice(list(range(len(circuit_data))))
            num_controls = 1
            connections = None
            try:
                num_controls = len(circuit_data[choice]) - 1
                if num_controls == 2:
                    controls = list(random.sample(encoder_obj.qubits_choice, k = num_controls))
                    reduced = [q for q in encoder_obj.qubits_choice if q not in controls]
                    target = np.random.choice(reduced, size=1, replace=True)
                    connections = [list(target)[0]]+[x for x in controls]
            except:
                num_controls = encoder_obj.n_qubits - len(encoder_obj._trash_qubits) - 1

            if num_controls == 0:
                target = np.random.choice(encoder_obj.qubits_choice, size=1, replace=True)
                connections = [list(target)[0]]
            elif num_controls == 1:
                controls = list(random.sample(encoder_obj.qubits_choice, k = num_controls))
                reduced = [q for q in encoder_obj.qubits_choice if q not in controls]
                target = np.random.choice(reduced, size=1, replace=True)
                connections = [list(target)[0]]+[x for x in controls]

            circuit_data[choice] = tuple(connections)
            success = True
        elif mutation == "remove":
            if len(circuit_data) <= 1:
                mutation = random_choice(["add", "replace"])
            else:
                choice = random_choice(list(range(len(circuit_data))))
                circuit_data.pop(choice)
                success = True
        elif mutation == "permute":
            if len(circuit_data) <= 1:
                mutation = random_choice(["add", "replace"])
            else:
                circuit_data = random_permutation(circuit_data)
                for ind, data in enumerate(circuit_data):
                    circuit_data[ind] = tuple(data)
                circuit_data = list(circuit_data)
                success = True
        elif mutation == "repeat":
            if len(circuit_data) <= 1:
                mutation = random_choice(["add", "replace"])
            else:
                choice = random_choice(list(range(len(circuit_data))))
                circuit_data.append(circuit_data[choice])
                success = True
        else:
            raise Exception("The mutation type selected, {0}, is not implemented yet".format(mutation))
    #print(mutation, circuit_data)
    return circuit_data
