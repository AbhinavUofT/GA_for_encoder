import tequila as tq
import numpy as np
from evolve import *
from fitness import *
from general_utils import *
from ccx_encoder import *

def initiate_GA(num_qubits = 4, qubits=[0,1,2,3], num_generations = 1000, generation_size = 15, circuits_data_dict = {}, max_circuit_len = 100,
                metric_weight = {}, num_processors = 1, input_space = None, trash_qubits = None, max_controls= 1, init_trash_qub_len=1):
    """
    This fucntion is the main GA function that runs the genetic algorithm to generate
    different circuits

    param: num_generations (int) -> number of generations of the GA to run
    param: generation_size (int) -> number of circuits in a particular generation
    param: circuit_data_dict (dict) -> the initial generation of circuits
    param: max_circuit_len (int) -> the maximum number of gates in a circuit
    param: num_processors (int) -> number of processors to be used for calculation
    param: metric_weight
    param: input_space
    param: trash_qubits
    param: max_controls

    e.g.:
    input:
    num_generations ->
    generation_size ->
    starting_circuits ->
    max_circuit_len ->
    metric_weight ->
    num_processors ->
    input_space ->
    trash_qubits ->
    max_controls ->

    output:

    """

    all_circuit_str = []
    best_circuit = None
    best_circuit_str = None
    #generating the initial set of circuits

    circuits_data_dict  = circuits_data_dict
    temp_trash = trash_qubits
    if max_controls - len(trash_qubits) <= 0:
        max_controls = num_qubits - len(trash_qubits) - 1
    encoder = evolved_ccx(num_qubits=num_qubits, qubits=qubits, input_space=input_space, trash_qubits=temp_trash, max_controls=max_controls)

    while len(circuits_data_dict ) < generation_size:
        circuit_data = encoder.sample_connection()
        #print(circuit_data)
        if len(circuits_data_dict) == 0:
            circuits_data_dict["cir{0}".format(len(circuits_data_dict ))] = [circuit_data]
        else:
            sim = calculate_similarity(circuits_data_dict.values() , circuit_data)
            if np.mean(sim) <= 0.7 and np.median(sim) <= 0.7 and max(sim)<=0.9:
                if "cir{0}".format(len(circuits_data_dict)) in circuits_data_dict.keys():
                    circuits_data_dict["cir{0}".format(len(circuits_data_dict))].append(circuit_data)
                else:
                    circuits_data_dict["cir{0}".format(len(circuits_data_dict ))] = [circuit_data]

    sorted_circuit_data_dict, sorted_fitness = calculate_fitness(encoder, circuits_data_dict, num_processors, metric_weight)

    prev_circuit_data_dict = {key : circuits_data_dict[key] for key in list(sorted_circuit_data_dict.keys())}

    all_circuit_str.extend(list(prev_circuit_data_dict.values()))

    print("****Starting Evolution: \n circuits {0}, \n fitness {1} \n".format(prev_circuit_data_dict, sorted_fitness))

    print("**** best transforamtion at the begining: ")
    encoder.analyze(encoder.make_circuit(circuits_data_dict[list(sorted_circuit_data_dict.keys())[0]]))

    list_of_choices = [2]
    # Startting the evolution loop
    for index in range(1, num_generations+1):
        print("   ###   On generation %i of %i"%(index, num_generations))
        replace, keep =  apply_generation_filter(list(prev_circuit_data_dict.keys()), generation_size, True)

        curr_gen_data_dict = generate_next_generation_circuits(encoder, prev_circuit_data_dict, keep, replace, list_of_choices, all_circuit_str)

        #print("*** new population details: \n circuits {0}, \n ".format(curr_gen_cirs_str))
        sorted_circuit_data_dict, sorted_fitness = calculate_fitness(encoder, curr_gen_data_dict, num_processors, metric_weight)
        prev_circuit_data_dict = {key : curr_gen_data_dict[key] for key in list(sorted_circuit_data_dict.keys())}
        print("*** population details: 5 best circuits \n circuits {0}, \n fitness {1} \n".format(dict(list(prev_circuit_data_dict.items())[0: 5]) , sorted_fitness[0:5]))

        print("**** best transforamtion of this generation: ")
        infidelity = encoder.analyze(encoder.make_circuit(prev_circuit_data_dict[list(sorted_circuit_data_dict.keys())[0]]))
        print("Infidelity : ", infidelity)

        all_circuit_str.extend(list(prev_circuit_data_dict.values()))

        #think of a better stopping criteria
        if (np.mean(sorted_fitness[0:5]) - sorted_fitness[0])  < 1e-3 or index==num_generations:
            intermed_cir, intermed_circ_str =  encoder.__str__(prev_circuit_data_dict[list(sorted_circuit_data_dict.keys())[0]])
            best_circuit = intermed_cir

    return best_circuit
