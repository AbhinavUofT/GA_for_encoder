import tequila as tq
import numpy as np
import os, copy, random
import multiprocessing

from utils import *

from ccx_encoder import *

manager = multiprocessing.Manager()
lock = multiprocessing.Lock()

class evolved_ccx(CCXEncoder):

    def __str__(self, circuit_data):
        """

        """
        circuit = self.make_circuit(circuit_data)
        #tq.draw(circuit, backend='qiskit')
        return circuit, circuit.__str__()

    def __call__(self, circuit_data:list, *args, **kwargs):
        """

        """
        U = self.make_circuit(circuit_data=circuit_data)

        depth = U.depth
        num_2_q_gate =  0
        num_1_q_gate = 0
        for gate in U.gates:
            if gate.is_controlled():
                num_2_q_gate += 1
            else:
                num_1_q_gate += 1

        infidelity =  0.0
        _1rdm = 0.0
        _2rdm = 0.0

        input_samples=self.get_input_samples()
        for U0 in input_samples:
            U0.n_qubits = self.num_qubits
            U.n_qubits = self.num_qubits
            wfn1= tq.simulate(U0+U, backend='qulacs', *args, **kwargs)

            dimension = 2**(len(self.qubits))
            dims = [[2]*len(self.qubits), [1]*len(self.qubits)]
            rdm1 = get_wavefunction_partial_trace(dimension,dims,wfn1,self._trash_qubits)

            rdm2 = self._target_dm

            infidelity += get_infidelity(rdm1, rdm2)
            _1rdm += get_1_rdm_distance(rdm1, rdm2, self._trash_qubits)
            if len(self._trash_qubits) >= 2:
                _2rdm += get_2_rdm_distance(rdm1, rdm2, self._trash_qubits)

        return infidelity, _1rdm, _2rdm, depth, num_2_q_gate, num_1_q_gate


def get_chunks(arr, num_processors, ratio):
    """
    Get chunks based on a list

    param: arr (list) -> a list to be divided into chunks for multiprocessing
    param: num_processors (int) -> number of processors to be used for calculation
    param: ratio (float) -> the maximum number of jobs on each processor

    e.g.:
    input:
    arr ->
    num_processors -> 2
    ratio -> 5.0

    output:
    chunks ->
    """
    chunks = []  # Collect arrays that will be sent to different processorr
    counter = int(ratio)
    for i in range(num_processors):
        if i == 0:
            chunks.append(arr[0:counter])
        if i != 0 and i<num_processors-1:
            chunks.append(arr[counter-int(ratio): counter])
        if i == num_processors-1 and i != 0:
            chunks.append(arr[counter-int(ratio): ])
        counter += int(ratio)
    return chunks

def evaluate_circuit(dict_metric, encoder_obj,  circuit_data_dict, circuit_chunks, metric_weight):
    """

    """
    for circuit_name in circuit_chunks:
        infidelity, _1rdm, _2rdm, depth, num_2_q_gate, num_1_q_gate = encoder_obj(circuit_data_dict[circuit_name])
        dict_metric[circuit_name] = metric_weight['infidelity'] * infidelity + metric_weight['depth'] * depth + metric_weight['num_2_q_gate'] * num_2_q_gate + \
                                    metric_weight['num_1_q_gate'] * num_1_q_gate + metric_weight['2_rdm'] * _2rdm + metric_weight['1_rdm'] * _1rdm

def create_parr_process(encoder_obj, circuit_data_dict, circuit_chunks, metric_weight):
    """

    """
    process_collector    = []
    collect_dictionaries = []
    for circuit_l in circuit_chunks:
        dict_metric = manager.dict(lock=True)
        collect_dictionaries.append(dict_metric)
        process_collector.append(multiprocessing.Process(target=evaluate_circuit, args=(dict_metric,  encoder_obj,  circuit_data_dict, circuit_l, metric_weight,  )))

    for item in process_collector:
        item.start()

    for item in process_collector: # wait for all parallel processes to finish
        item.join()

    combined_dict = {} # collect results from multiple processess
    for i,item in enumerate(collect_dictionaries):
        combined_dict.update(item)

    return combined_dict

def order_based_on_fitness(dict_metrics):
    """
    This function orders the dictionary as per the metric value

    param: dict_metrics (dict) -> dictionary with all the metrics

    e.g.:
    input:
    dict_metrics ->

    output:
    sorted_circuit_dict ->
    sorted_fitness ->
    """
    dict_metrics = dict(sorted(dict_metrics.items(), key=lambda item: item[1]))
    dict_metrics.pop('lock')
    sorted_fitness = list(dict_metrics.values())
    return dict_metrics, (sorted_fitness)

def fitness(encoder_obj, circuit_data_dict, num_processors, metric_weight):
    """

    """
    ratio = len(circuit_data_dict)/num_processors
    circuit_chunks = get_chunks(list(circuit_data_dict.keys()), num_processors, ratio)

    fitness_dict = create_parr_process(encoder_obj, circuit_data_dict, circuit_chunks, metric_weight)
    sorted_circuit_data_dict, sorted_fitness = order_based_on_fitness(fitness_dict)

    return sorted_circuit_data_dict, sorted_fitness

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

def sort_by_qubits(circuit_data):
    """

    """
    #print(circuit_data)
    circuit_data = sorted(circuit_data, key=lambda item: item[0])
    #print(circuit_data)
    #raise Exception("testing")
    return circuit_data


def compress_circuit_str(circuit_data):
    """

    """
    new_data = copy.deepcopy(circuit_data)
    circuit_data = sort_by_qubits(circuit_data)
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

def generate_next_generation_circuits(encoder_obj, circuit_data_dict, keep_list, replace_list, list_of_choices):
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
        #print(index)
        next_gen_circuits[keys_l[index]] = copy.deepcopy(circuit_data_dict[keys_l[index]])
    for index in replace_list:
        #print(index)
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
            #print(mutated_circuit_data)
            #print(repetition, np.mean(sim), np.median(sim), max(sim))
            if np.mean(sim) <= 0.75 and np.median(sim) <= 0.75 and max(sim) < 0.9 and repetition == False:
                done = False
        #print(mutated_circuit_data)
        next_gen_circuits[keys_l[index]] = mutated_circuit_data

    return next_gen_circuits
