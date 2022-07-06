import tequila as tq
import numpy as np
import os, copy, random
import multiprocessing

def evaluate_circuit(process_id, encoder_obj, circuit_data_dict, metric_weight, tasks, results):
    """
    This function evaluates a circuit and returns the fitness value

    param:
    process_id: A unique identifier for each process
    encoder_obj (evolved_ccx object):
    circuit_data_dict (dict):
    metric_weight (dict):
    tasks: multiprocessing queue to pass circuit_name
        circuit_name: A unique identifier for each circuit
    results: multiprocessing queue to pass results back
    """
    print('[%s] evaluation routine starts' % process_id)
    while True:
        try:
            circuit_name = tasks.get()
            infidelity, _1rdm, _2rdm, depth, num_2_q_gate, num_1_q_gate = encoder_obj(circuit_data_dict[circuit_name])
            fitness = metric_weight['infidelity'] * infidelity + metric_weight['depth'] * depth + metric_weight['num_2_q_gate'] * num_2_q_gate + \
                    metric_weight['num_1_q_gate'] * num_1_q_gate + metric_weight['2_rdm'] * _2rdm + metric_weight['1_rdm'] * _1rdm
            results.put((circuit_name, fitness))
        except:
            print('[%s] evaluation routine quits' % process_id)

            # Indicate finished
            results.put(-1)
            break
    return

def parallelize_evaluation(encoder_obj, circuit_data_dict, metric_weight, num_processors):
    """
    This fucntion creates parallel processes to run the circuit to evaluate the
    objective value and returns the results

    param:
    encoder_obj (evolved_ccx object):
    circuit_data_dict (dict):
    num_processors (int): number of processors to be used for calculation
    metric_weight (dict):
    """

    # Define IPC manager
    manager = multiprocessing.Manager()

    # Define a list (queue) for tasks and computation results
    tasks = manager.Queue()
    results = manager.Queue()

    processes = []
    pool = multiprocessing.Pool(processes=num_processors)

    for i in range(num_processors):
        process_id = 'P%i' % i

        # Create the process, and connect it to the worker function
        new_process = multiprocessing.Process(target=evaluate_circuit,
                                              args=(process_id, encoder_obj,
                                                    circuit_data_dict,  metric_weight,
                                                    tasks, results))

        # Add new process to the list of processes
        processes.append(new_process)

        # Start the process
        new_process.start()

    for single_task in list(circuit_data_dict.keys()):
        tasks.put(single_task)

    multiprocessing.Barrier(num_processors)

    # Quit the worker processes by sending them -1
    for i in range(num_processors):
        tasks.put(-1)

    combined_dict = {}
    # Read calculation results
    num_finished_processes = 0
    while True:
        # Read result
        try:
            circuit_name, new_fitness = results.get()
            combined_dict[circuit_name] = new_fitness
        except:
            # Process has finished
            num_finished_processes += 1

            if num_finished_processes == num_processors:
                break

    return combined_dict

def order_based_on_fitness(dict_fitness):
    """
    This function orders the dictionary as per the metric value

    param: dict_fitness (dict) -> dictionary with keys as circuit identifiers
                                and values as fitness

    e.g.:
    input:
    dict_metrics ->

    output:
    sorted_circuit_dict ->
    sorted_fitness ->
    """
    dict_fitness = dict(sorted(dict_fitness.items(), key=lambda item: item[1]))
    sorted_fitness = list(dict_fitness.values())
    return dict_fitness, sorted_fitness

def calculate_fitness(encoder_obj, circuit_data_dict, num_processors, metric_weight):
    """
    This fucntion evaluates the fitness of all the circuits in the population by
    using multiple processes in parallel

    param:
    encoder_obj (evolved_ccx object):
    circuit_data_dict (dict):
    num_processors (int): number of processors to be used for calculation
    metric_weight (dict):
    """
    fitness_dict = parallelize_evaluation(encoder_obj, circuit_data_dict,
                                            metric_weight, num_processors)
    sorted_circuit_data_dict, sorted_fitness = order_based_on_fitness(fitness_dict)

    return sorted_circuit_data_dict, sorted_fitness
