import tequila as tq

from evolve import *

def initiate_GA(num_generations = 1000, generation_size = 15, starting_circuits = [], max_circuit_len = 100,
                metric_weight = {}, num_processors = 1, input_space = None, trash_qubits = None, max_controls= 1):
    """
    This fucntion is the main GA function that runs the genetic algorithm to generate
    different circuits

    param: num_generations (int) -> number of generations of the GA to run
    param: generation_size (int) -> number of circuits in a particular generation
    param: starting_circuits (dict) -> the initial generation of circuits
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
    #generating the initial set of circuits

    circuits_data_dict  = {}
    encoder = evolved_ccx(input_space=input_space, trash_qubits=trash_qubits, max_controls=max_controls)
    while len(circuits_data_dict ) < generation_size:
        circuit_data = encoder.sample_connection()
        #print(circuit_data)
        if len(circuits_data_dict) == 0:
            circuits_data_dict["cir{0}".format(len(circuits_data_dict ))] = [circuit_data]
        else:
            sim = calculate_similarity(circuits_data_dict.values() , circuit_data)
            if np.mean(sim) <= 0.7 and np.median(sim) <= 0.7 and max(sim)<1.0:
                if "cir{0}".format(len(circuits_data_dict)) in circuits_data_dict.keys():
                    circuits_data_dict["cir{0}".format(len(circuits_data_dict))].append(circuit_data)
                else:
                    circuits_data_dict["cir{0}".format(len(circuits_data_dict ))] = [circuit_data]

    sorted_circuit_data_dict, sorted_fitness = fitness(encoder, circuits_data_dict, num_processors, metric_weight)

    prev_circuit_data_dict = {key : circuits_data_dict[key] for key in list(sorted_circuit_data_dict.keys())}

    print("****Starting Evolution: \n circuits {0}, \n fitness {1} \n".format(prev_circuit_data_dict, sorted_fitness))

    print("**** best transforamtion at the begining: ")
    encoder.analyze(encoder.make_circuit(circuits_data_dict[list(sorted_circuit_data_dict.keys())[0]]))

    # Startting the evolution loop
    for index in range(1, num_generations+1):
        print("   ###   On generation %i of %i"%(index, num_generations))
        replace, keep =  apply_generation_filter(list(prev_circuit_data_dict.keys()), generation_size)

        curr_gen_data_dict = generate_next_generation_circuits(encoder, prev_circuit_data_dict, keep, replace)

        #print("*** new population details: \n circuits {0}, \n ".format(curr_gen_cirs_str))
        sorted_circuit_data_dict, sorted_fitness = fitness(encoder, curr_gen_data_dict, num_processors, metric_weight)

        prev_circuit_data_dict = {key : prev_circuit_data_dict[key] for key in list(sorted_circuit_data_dict.keys())}
        print("*** population details: \n circuits {0}, \n fitness {1} \n".format(prev_circuit_data_dict, sorted_fitness))

        print("**** best transforamtion at the begining: ")
        infidelity = encoder.analyze(encoder.make_circuit(prev_circuit_data_dict[list(sorted_circuit_data_dict.keys())[0]]))
        print("Infidelity : ", infidelity)
        if infidelity == 0.0:
            encoder.__str__(prev_circuit_data_dict[list(sorted_circuit_data_dict.keys())[0]])
            break

if __name__ == "__main__":
    input_space=["1100", "1001", "0110", "0011"]
    trash_qubits=[0,1]
    max_controls=1
    metric_weight = {'infidelity':1.0, '2_rdm':1.0, '1_rdm':1.0, 'depth':0.1, 'num_2_q_gate':0.1, 'num_1_q_gate':0.1}
    initiate_GA(input_space=input_space, trash_qubits=trash_qubits, max_controls=max_controls, metric_weight=metric_weight)
