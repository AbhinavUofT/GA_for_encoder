import tequila as tq
import math

from evolve import *

def initiate_GA(num_qubits = 4, num_generations = 1000, generation_size = 15, starting_circuits = [], max_circuit_len = 100,
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
    encoder = evolved_ccx(num_qubits=num_qubits, input_space=input_space, trash_qubits=trash_qubits, max_controls=max_controls)
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

def generate_target(num_qubits, keyword):
    """

    """
    vaccum_state = ""
    for _ in range(num_qubits):
        vaccum_state += "0"
    target_state = []
    trash_qubits = []
    if keyword == 'unary':
        for ind in range(num_qubits):
            temp_state = copy.deepcopy(vaccum_state)
            temp = list(temp_state )
            temp[ind] = "1"
            temp_state  = "".join(temp)
            target_state.append(temp_state)
        data_q = math.ceil(math.log(num_qubits, 2))
        for i in range(num_qubits - data_q):
            trash_qubits.append(i)
        return target_state, trash_qubits
    elif keyword == "random_unary":
        for _ in range(num_qubits):
            temp_state = copy.deepcopy(vaccum_state)
            temp = list(temp_state )
            num_ = random_choice(list(range(1, num_qubits+1)))
            choice = random_choice(list(range(num_qubits)), num_)
            if isinstance(choice, np.int64):
                choice = [choice]
            for ind in choice:
                temp[ind] = "1"
            temp_state  = "".join(temp)
            target_state.append(temp_state)
        data_q = math.ceil(math.log(num_qubits, 2))
        for i in range(num_qubits - data_q):
            trash_qubits.append(i)
        return target_state, trash_qubits
    elif keyword == "random":
        num_s = random_choice(list(range(2, 2**(num_qubits-1)-1)))
        #print(num_s)
        for _ in range(num_s):
            temp_state = copy.deepcopy(vaccum_state)
            temp = list(temp_state )
            num_ = random_choice(list(range(1, num_qubits+1)))
            choice = random_choice(list(range(num_qubits)), num_)
            if isinstance(choice, np.int64):
                choice = [choice]
            for ind in choice:
                temp[ind] = "1"
            temp_state  = "".join(temp)
            target_state.append(temp_state)
        data_q = math.ceil(math.log(num_s, 2))
        for i in range(num_qubits - data_q):
            trash_qubits.append(i)
        return target_state, trash_qubits

if __name__ == "__main__":
    num_qubits = xxxx
    input_space, trash_qubits = generate_target(num_qubits, "yyyy")
    max_controls = 2
    metric_weight = {'infidelity':1.0, '2_rdm':0.5, '1_rdm':0.5, 'depth':0.1, 'num_2_q_gate':0.01, 'num_1_q_gate':0.01}
    initiate_GA(num_qubits=num_qubits, input_space=input_space, num_processors=8, generation_size=num_qubits*5, trash_qubits=trash_qubits, max_controls=max_controls, metric_weight=metric_weight)
