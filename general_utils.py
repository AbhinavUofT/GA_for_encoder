import numpy as np
from difflib import SequenceMatcher

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
