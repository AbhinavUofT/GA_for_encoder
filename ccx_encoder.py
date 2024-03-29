import tequila as tq
import numpy
import random
import numbers

from general_utils import *
from encoder_utils import *

# basically the same as the old binary encoder, just uses tq

class CCXEncoder:

    def assign_states(self, states):

        circuits = []
        for state in states:
            if hasattr(state, "gates"):
                circuits.append(state)
            elif isinstance(state, str):
                state=state.strip(">")
                if "|" in state:
                    state = state.split("|")[1]
                state = tq.BitString.from_binary(state)
                state = sum([tq.gates.X(q) for q,x in enumerate(state.array) if x == 1], tq.QCircuit())
                circuits.append(state)
            else:
                raise Exception("unknown input format state={}".format(state))

        return circuits

    @property
    def qubits(self):
        return self._qubits

    @property
    def n_qubits(self):
        return len(self.qubits)

    @property
    def trash_qubits(self):
        return self._trash_qubits

    @property
    def input_space(self):
        return self._input_space

    def get_qubits(self):
        qubits = list(set(sum([x.qubits for x in self.input_space],[])))
        qubits += self.trash_qubits
        return list(set(qubits))

    def create_tar_dm(self):
        wfn_str = "1.0*|"
        for _ in self._trash_qubits:
            wfn_str += "0"
        wfn_str += ">"

        dimension = 2**(len(self._trash_qubits))
        dims = [[2]*len(self._trash_qubits), [1]*len(self._trash_qubits)]
        wfn = tq.QubitWaveFunction.from_string(wfn_str).normalize()
        dims = [[2]*len(self._trash_qubits), [2]*len(self._trash_qubits)]
        return get_density_matrix(dimension, dims, wfn)

    def __init__(self, num_qubits:int, qubits:list,  input_space:list, trash_qubits:list, max_controls=2, *args, **kwargs):
        """
        :param input_states: list of computational basis states (circuits or strings like 00100)
        """
        self.num_qubits = num_qubits
        self._input_space = self.assign_states(input_space)
        self._trash_qubits = trash_qubits
        self._qubits = self.get_qubits()
        self.qubits_choice = qubits

        self._target_dm = self.create_tar_dm()

        self.n_input_samples=None
        self.p_input_samples=None

        self.max_controls=max_controls

        #assert len(self.input_space) <= 2**len(self.trash_qubits)

    def __call__(self, circuit_data:list, *args, **kwargs):
        """
        Will create the circuit and evaluate
        :param args:
        :param kwargs:
        :param circuit: List of tuples, each with 3 integers defining a CCX gate
        :return: evaluated cost function
        """
        O = self.make_objective(circuit_data=circuit_data)
        result = tq.simulate(O, backend='qulacs', *args, **kwargs)
        return result

    def make_circuit(self, circuit_data:list):
        U = tq.QCircuit()
        for x in circuit_data:
            control = [q for q in x[1:] if q is not None]
            U += tq.gates.X(target=x[0], control=control)
        return U

    def sample_connection(self, p=None):
        num_controls = numpy.random.choice(list(range(self.max_controls+1)))
        if num_controls == 0:
            target = numpy.random.choice(self.qubits_choice, size=1, replace=True, p=p)
            connections = [list(target)[0]]
            return tuple(connections)
        elif num_controls == 1:
            controls = list(random.sample(self.qubits_choice, k = num_controls))
            reduced = [q for q in self.qubits_choice if q not in controls]
            target = numpy.random.choice(reduced, size=1, replace=True, p=p)
            connections = [list(target)[0]]+[x for x in controls]
            #check if the trash qubits are in the connections randomly and add it
            #randomly if not there with probability 0.25
            t_q = random_choice(self._trash_qubits)
            if t_q not in connections:
                choice = random_choice(list(range(len(connections))))
                choice_1 = random_choice([0, 1, 1, 1])
                if choice_1 == 0:
                    connections[choice] = t_q
            return tuple(connections)
        elif num_controls == 2:
            controls = list(random.sample(self.qubits_choice, k = num_controls))
            reduced = [q for q in self.qubits_choice if q not in controls]
            target = numpy.random.choice(reduced, size=1, replace=True, p=p)
            connections = [list(target)[0]]+[x for x in controls]
            #check if the trash qubits are in the connections randomly and add it
            #randomly if not there with probability 0.25
            t_q = random_choice(self._trash_qubits)
            if t_q not in connections:
                choice = random_choice(list(range(len(connections))))
                choice_1 = random_choice([0, 1, 1, 1])
                if choice_1 == 0:
                    connections[choice] = t_q
            return tuple(connections)

    def make_objective(self, circuit_data):
        U = self.make_circuit(circuit_data=circuit_data)
        H = tq.paulis.Qp(self.trash_qubits)
        input_samples=self.get_input_samples()
        objective = sum([tq.ExpectationValue(H=H, U=U0+U) for U0 in input_samples], 0.0)
        return -1.0/len(input_samples)*objective

    def get_input_samples(self):
        # for now the whole thing
        if self.n_input_samples is None:
            return self.input_space
        else:
            return numpy.random.choice(self.input_space, n=self.n_input_samples, replace=False, p=self.p_input_samples)

    def analyze(self, circuit, *args, **kwargs):
        infidelity = 0.0
        for U0 in self.input_space:
            U = U0 + circuit
            U0.n_qubits = self.num_qubits
            U.n_qubits = self.num_qubits
            input = tq.simulate(U0, backend='qulacs', *args, **kwargs)
            target = tq.simulate(U, backend='qulacs', *args, **kwargs)
            print("{:25} --> {:25}".format(str(input), str(target)))

            dimension = 2**(len(self.qubits))
            dims = [[2]*len(self.qubits), [1]*len(self.qubits)]
            rdm1 = get_wavefunction_partial_trace(dimension,dims,target,self._trash_qubits)
            rdm2 = self._target_dm
            infidelity += get_infidelity(rdm1, rdm2)
        return infidelity

class evolved_ccx(CCXEncoder):

    def __str__(self, circuit_data):
        """

        """
        circuit = self.make_circuit(circuit_data)
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


if __name__ == "__main__":
    encoder = CCXEncoder(input_space=["1100", "1001", "0110", "0011"], trash_qubits=[0,1], max_controls=1)
    n_gates = 4
    trials = 100
    best = [100.0, None]
    for x in range(trials):
        circuit_data = [encoder.sample_connection() for x in range(n_gates)]
        cost = encoder(circuit_data=circuit_data)
        if cost < best[0]:
            best = [cost, encoder.make_circuit(circuit_data)]

    print(best)
    encoder.analyze(best[1])
    print("best possible")
    encoder.analyze(tq.gates.CNOT(2,0)+tq.gates.CNOT(3,1)+tq.gates.X([0,1]))
