import tequila as tq
import math, copy, itertools
import numpy as np
from GA_utils import *

def generate_target(num_qubits, num_particles=1, keyword='unary', num_states=0):
    """
    This function generates the target state that has to be compressed by
    the CCX encoder

    param:
    num_qubits (int): Number of qubits in each circuit of the population
    num_particles (int): Number of particles in each basis states of the target state
    keyword (str): A string indicating which target states to prepare
    num_states (int): The number of basis states in the target state
    """
    vaccum_state = ""
    for _ in range(num_qubits):
        vaccum_state += "0"

    target_state = []
    trash_qubits = []

    #preparing UNARY states with num_particles "1" and returns only "num_states" basis states
    #if has more states than "num_states"
    if keyword == 'unary':
        for indices in itertools.combinations(list(range(num_qubits)), num_particles):
            temp_state = copy.deepcopy(vaccum_state)
            temp = list(temp_state)
            for ind in indices:
                temp[ind] = "1"
            temp_state  = "".join(temp)
            target_state.append(temp_state)

        if num_states != 0 and num_states < len(target_state):
            choices = random_choice(list(range(len(target_state))), len(target_state) - num_states)
            choices.sort()
            choices = list(choices)
            choices.reverse()
            for choice in choices:
                target_state.pop(choice)

        data_q = math.ceil(math.log(len(target_state), 2))
        for i in range(num_qubits - data_q):
            trash_qubits.append(i)
        return target_state, trash_qubits

    #preparing random combinations of states with same number of states as UNARY states
    elif keyword == "random_unary":
        for _ in range(num_qubits):
            repeat = True
            while repeat == True:
                temp_state = copy.deepcopy(vaccum_state)
                temp = list(temp_state )
                num_ = random_choice(list(range(1, num_qubits+1)))
                choice = random_choice(list(range(num_qubits)), num_)
                if isinstance(choice, np.int64):
                    choice = [choice]
                for ind in choice:
                    temp[ind] = "1"
                temp_state  = "".join(temp)
                if temp_state not in target_state:
                    repeat = False
                    target_state.append(temp_state)
        data_q = math.ceil(math.log(num_qubits, 2))
        for i in range(num_qubits - data_q):
            trash_qubits.append(i)
        return target_state, trash_qubits

    #preparing random combinations of states with same number of states as "num_states"
    elif keyword == "random":
        if num_states == 0:
            num_states = random_choice(list(range(1, num_qubits-2)))
            num_states = 2**(num_states)
        for _ in range(num_states):
            repeat = True
            while repeat == True:
                temp_state = copy.deepcopy(vaccum_state)
                temp = list(temp_state )
                num_ = random_choice(list(range(1, num_qubits+1)))
                choice = random_choice(list(range(num_qubits)), num_)
                if isinstance(choice, np.int64):
                    choice = [choice]
                for ind in choice:
                    temp[ind] = "1"
                temp_state  = "".join(temp)
                if temp_state not in target_state:
                    repeat = False
                    target_state.append(temp_state)
        data_q = math.ceil(math.log(num_states, 2))
        for i in range(num_qubits - data_q):
            trash_qubits.append(i)
        return target_state, trash_qubits

    #preparing GHZ states as target states
    elif keyword == "GHZ":
        for value in range(2):
            temp_state = copy.deepcopy(vaccum_state)
            for ind in range(num_qubits):
                temp = list(temp_state )
                temp[ind] = "{0}".format(value)
                temp_state  = "".join(temp)
            target_state.append(temp_state)
        data_q = 1
        for i in range(num_qubits - data_q):
            trash_qubits.append(i)
        return target_state, trash_qubits

    #preparing Prime states  as target states
    elif keyword == "Prime":
        temp_target_state = np.zeros(2 ** num_qubits)
        temp_target_state = np.array(temp_target_state, dtype=np.complex64)
        prime_l = [2]
        for num in range(3,2 ** num_qubits,2):
            if all(num%i!=0 for i in range(3,int(math.sqrt(num))+1, 2)):
                prime_l.append(num)
        coeff = np.complex64(1/((len(prime_l))**0.5))
        for i in prime_l:
            temp_state = np.zeros(2 ** num_qubits)
            temp_state[i] = 1
            temp_target_state += coeff*temp_state
        temp_target_state = np.array(temp_target_state, dtype=np.complex64)
        temp_target_state = str(tq.QubitWaveFunction.from_array(temp_target_state))

        for value in temp_target_state.split(">")[:-1]:
            temp_state = copy.deepcopy(value.split("|")[-1])
            target_state.append(temp_state)
        data_q = math.ceil(math.log(len(target_state), 2))
        for i in range(num_qubits - data_q):
            trash_qubits.append(i)
        return target_state, trash_qubits
