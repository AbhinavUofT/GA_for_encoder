import tequila as tq
import math

from evolve import *

def initiate_GA_adapt(num_qubits = 4, qubits=[0,1,2,3], num_generations = 10000, generation_size = 15, circuits_data_dict = {},
                max_circuit_len = 100, metric_weight = {}, num_processors = 1, input_space = None, trash_qubits = None,
                max_controls= 1, init_trash_qub_len=1):
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
    best_circuit = tq.QCircuit()
    best_circuit_str = []

    temp_trash = [trash_qubits[0]]

    encoder = evolved_ccx(num_qubits=num_qubits, qubits=qubits, input_space=input_space, trash_qubits=temp_trash, max_controls=max_controls)

    circuits_data_dict  = circuits_data_dict

    for index, qubit in enumerate(trash_qubits):
        if index > 0:
            encoder._trash_qubits.append(qubit)
        encoder._target_dm = encoder.create_tar_dm()

        #generating the initial set of circuits
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

        sorted_circuit_data_dict, sorted_fitness = fitness(encoder, circuits_data_dict, num_processors, metric_weight)

        prev_circuit_data_dict = {key : circuits_data_dict[key] for key in list(sorted_circuit_data_dict.keys())}

        print("****Starting Evolution: \n circuits {0}, \n fitness {1} \n".format(prev_circuit_data_dict, sorted_fitness))

        print("**** best transforamtion at the begining: ")
        encoder.analyze(encoder.make_circuit(circuits_data_dict[list(sorted_circuit_data_dict.keys())[0]]))

        list_of_choices = [0,0,1]
        # Startting the evolution loop
        for index in range(1, num_generations+1):
            print("   ###   On generation %i of %i"%(index, num_generations))
            replace, keep =  apply_generation_filter(list(prev_circuit_data_dict.keys()), generation_size)

            curr_gen_data_dict = generate_next_generation_circuits(encoder, prev_circuit_data_dict, keep, replace, list_of_choices)

            #print("*** new population details: \n circuits {0}, \n ".format(curr_gen_cirs_str))
            sorted_circuit_data_dict, sorted_fitness = fitness(encoder, curr_gen_data_dict, num_processors, metric_weight)
            prev_circuit_data_dict = {key : curr_gen_data_dict[key] for key in list(sorted_circuit_data_dict.keys())}
            print("*** population details: 5 best circuits \n circuits {0}, \n fitness {1} \n".format(dict(list(prev_circuit_data_dict.items())[0: 5]) , sorted_fitness[0:5]))

            print("**** best transforamtion of this generation: ")
            infidelity = encoder.analyze(encoder.make_circuit(prev_circuit_data_dict[list(sorted_circuit_data_dict.keys())[0]]))
            print("Infidelity : ", infidelity)
            if infidelity == 0.0:
                intermed_cir, circ_str =  encoder.__str__(prev_circuit_data_dict[list(sorted_circuit_data_dict.keys())[0]])
                best_circuit += intermed_cir
                best_circuit_str += prev_circuit_data_dict[list(sorted_circuit_data_dict.keys())[0]]
                #qubits.remove(qubit)
                encoder.qubits_choice.remove(qubit)
                for ind, _ in enumerate(encoder._input_space):
                    encoder._input_space[ind] += intermed_cir
                break
    return best_circuit, best_circuit_str


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

    sorted_circuit_data_dict, sorted_fitness = fitness(encoder, circuits_data_dict, num_processors, metric_weight)

    prev_circuit_data_dict = {key : circuits_data_dict[key] for key in list(sorted_circuit_data_dict.keys())}

    print("****Starting Evolution: \n circuits {0}, \n fitness {1} \n".format(prev_circuit_data_dict, sorted_fitness))

    print("**** best transforamtion at the begining: ")
    encoder.analyze(encoder.make_circuit(circuits_data_dict[list(sorted_circuit_data_dict.keys())[0]]))

    list_of_choices = [2]
    # Startting the evolution loop
    for index in range(1, num_generations+1):
        print("   ###   On generation %i of %i"%(index, num_generations))
        replace, keep =  apply_generation_filter(list(prev_circuit_data_dict.keys()), generation_size, True)

        curr_gen_data_dict = generate_next_generation_circuits(encoder, prev_circuit_data_dict, keep, replace, list_of_choices)

        #print("*** new population details: \n circuits {0}, \n ".format(curr_gen_cirs_str))
        sorted_circuit_data_dict, sorted_fitness = fitness(encoder, curr_gen_data_dict, num_processors, metric_weight)
        prev_circuit_data_dict = {key : curr_gen_data_dict[key] for key in list(sorted_circuit_data_dict.keys())}
        print("*** population details: 5 best circuits \n circuits {0}, \n fitness {1} \n".format(dict(list(prev_circuit_data_dict.items())[0: 5]) , sorted_fitness[0:5]))

        print("**** best transforamtion of this generation: ")
        infidelity = encoder.analyze(encoder.make_circuit(prev_circuit_data_dict[list(sorted_circuit_data_dict.keys())[0]]))
        print("Infidelity : ", infidelity)

        #think of a better stopping criteria
        if (np.mean(sorted_fitness[0:5]) - sorted_fitness[0])  < 1e-3 or index==num_generations:
            intermed_cir, intermed_circ_str =  encoder.__str__(prev_circuit_data_dict[list(sorted_circuit_data_dict.keys())[0]])
            best_circuit = intermed_cir

    return best_circuit

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
    qubits = list(range(num_qubits))
    input_space, trash_qubits = generate_target(num_qubits, "yyyy")

    max_controls = 2
    if num_qubits <= 4:
        metric_weight = {'infidelity':1.0, '2_rdm':1.0, '1_rdm':1.0, 'depth':0.0125, 'num_2_q_gate':0.01, 'num_1_q_gate':0.0099}
    else:
        metric_weight = {'infidelity':1.0, '2_rdm':1.0, '1_rdm':1.0, 'depth':0.00, 'num_2_q_gate':0.0, 'num_1_q_gate':0.0}
    best_circuit, circuit_str = initiate_GA_adapt(init_trash_qub_len=0, num_qubits=num_qubits, qubits=qubits, input_space=input_space, num_processors=2, generation_size=num_qubits*10, trash_qubits=trash_qubits, max_controls=max_controls, metric_weight=metric_weight)
    print(best_circuit)
    metric_weight = {'infidelity':1.0, '2_rdm':1.0, '1_rdm':1.0, 'depth':0.125, 'num_2_q_gate':0.1, 'num_1_q_gate':0.099}
    initial_circuit_dict={"cir0":circuit_str}
    best_circuit = initiate_GA(num_generations = 100,circuits_data_dict=initial_circuit_dict, init_trash_qub_len=len(trash_qubits), num_qubits=num_qubits, qubits=qubits, input_space=input_space, num_processors=2, generation_size=num_qubits*10, trash_qubits=trash_qubits, max_controls=max_controls, metric_weight=metric_weight)
    print(best_circuit)
