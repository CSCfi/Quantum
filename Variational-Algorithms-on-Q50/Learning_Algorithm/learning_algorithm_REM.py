import numpy as np
from qiskit import QuantumCircuit
from qiskit.result import Counts
from qiskit.compiler import transpile
from qiskit.quantum_info import Operator
from typing import List
from iqm.qiskit_iqm.fake_backends import fake_aphrodite
from scipy.optimize import minimize, dual_annealing, basinhopping
from qiskit_algorithms.optimizers import SPSA
from qiskit.result import CorrelatedReadoutMitigator
import scipy.linalg as la

class Learning:
	def __init__(self, ansatz, backend, hamiltonian, estimator, optimizer: List[float], layout) -> None:
		self.optimizer = optimizer
		self.backend = backend
		self.seed = None
		self.hamiltonian = hamiltonian
		self.estimator = estimator
		self.ansatz = ansatz
		self.energy = None
		self.num_params = ansatz.num_parameters
		self.n = ansatz.num_qubits
		self.min_kwargs = None
		self.x0 = None
		self.bounds = None
		self.params = None
		self.cost = None
		self.p = None
		self.iter = None
		self.nfev = None
		self.result = None
		self.shots = None
		self.cost_function = None
		self.layout = layout
        
		self.costs_dictionary = {
            "params": None,
            "iterations": 0,
            "costs": [],
        }

	def run(self,
	        min_kwargs: dict | None = None,
	        x0: List[float] | None = None,
            params: List[float] | None = None,
	        shots: int | None = None,
	        seed: int | None = None
	        ) -> dict:
		optimizer = self.optimizer
		self.num_params = self.ansatz.num_parameters
		rng = np.random.default_rng()
		self.x0 = rng.normal(0, 0.01, self.ansatz.num_parameters)
		# self.x0 = np.random.normal(0, 0.01, self.ansatz.num_parameters)  # legacy generator method
		self.bounds = [(-np.pi, np.pi)] * self.num_params
		self.min_kwargs = min_kwargs if min_kwargs else {}
		self.cost_function = self.energy_cost_function

		if optimizer == 'dual_annealing':
			self.optimizer = dual_annealing
			self.result = dual_annealing(func=self.cost_function, bounds=self.bounds, callback=self.callback)
		elif optimizer == 'basinhopping':
			self.optimizer = basinhopping
			self.result = basinhopping(func=self.cost_function, x0=self.x0)
		elif optimizer == 'cobyla':
			self.result = minimize(self.cost_function, self.x0, method='COBYLA', tol=1e-5, options={"maxiter":5000})#, "rhobeg": 0.5})
		elif optimizer == 'BFGS':
			self.result = minimize(self.cost_function, self.x0, method='BFGS', tol=1e-5, options={"maxiter": 5000})
		elif optimizer == 'SPSA':
			# SPSA with possible decaying perturbation and learning rate
			def sc_generator(values: list, switch_points:list, total_iters:int=100):
				def generator():
					for k in range(total_iters):
						if k < switch_points[0]:
							yield values[0]
						elif k < switch_points[1]:
							yield values[1]
						else:
							yield values[2]
				return generator
			
			def two_phase_learning_rate(a0:float, a1:float, a2:float, switch_iter1:int, switch_iter2:int):
				return sc_generator([a0, a1, a2], [switch_iter1, switch_iter2])
			def two_phase_pertubation(c0:float, c1:float, c2:float, switch_iter1:int, switch_iter2:int):
				return sc_generator([c0, c1, c2], [switch_iter1, switch_iter2])

			spsa = SPSA(
				maxiter=75,
				learning_rate=two_phase_learning_rate(a0 = 0.00000002, a1=0.02, a2=0.02, switch_iter1 = 3, switch_iter2=70),
				perturbation=two_phase_pertubation(c0 = 0.00000001, c1=0.3, c2=0.3, switch_iter1 = 3, switch_iter2=70))
			self.result = spsa.minimize(self.cost_function, self.x0)
		elif optimizer == 'SPSA2':
			spsa = SPSA(maxiter=100)
			self.result = spsa.minimize(self.cost_function, self.x0)

			self.params = self.result.x
			return {
                    "n": self.n,
                    "iter": self.iter,
                    "nfev": self.nfev,
                    "cost": self.cost,
                    "params": self.params,
                    "ansatz": self.ansatz,
                    "optimizer": self.optimizer.__class__.__name__,
                    "min_kwargs": self.min_kwargs,
                    "shots": self.shots,
                }

	@staticmethod
	def filter_counts_by_particle_number(counts: Counts, target_particles: int) -> Counts:
		filtered_counts = {
			bitstr: count for bitstr, count in counts.items()
			if bitstr.count('1') == target_particles
		}
		total = sum(filtered_counts.values())
		if total == 0:
			raise ValueError("No bitstrings matched the target particle number.")
		probabilities = {bitstr: count/total for bitstr, count in filtered_counts.items()}
		
		scaled_counts = {}
		remainders = []
		
		for bitstr, prob in probabilities.items():
			exact = prob * 1000
			integer = int(exact)
			scaled_counts[bitstr] = integer
			remainders.append((bitstr, exact - integer))
		
		remaining = 1000 - sum(scaled_counts.values())
		for bitstr, remainder in sorted(remainders, key=lambda x: -x[1])[:remaining]:
			scaled_counts[bitstr] += 1
		
		return scaled_counts
	

	@staticmethod
	def calibration_circuits(num_qubits):
		circuits = []
		for i in range(2**num_qubits):
			state = format(i, f'0{num_qubits}b')
			qc = QuantumCircuit(num_qubits, num_qubits)
                
			for j, bit in enumerate(reversed(state)):
				if bit == '1':
					qc.x(j)
                
			qc.measure(range(num_qubits), range(num_qubits))
			circuits.append(qc)
            
		return circuits
	

	@staticmethod
	def mitigation_matrix(calibration_results, num_qubits):
		num_states = 2**num_qubits
		matrix = np.zeros((num_states, num_states))
            
		for state_index in range(num_states):
			counts = calibration_results.get_counts(state_index)
			total_shots = sum(counts.values())
                
			for measured_state, count in counts.items():
				measured_state_index = int(measured_state, 2)
				matrix[measured_state_index, state_index] = count / total_shots
            
		return matrix
	
	@staticmethod
	def mitigate_counts(noisy_counts, minv, num_qubits, shots):
		counts_vector = np.zeros(2**num_qubits)
		for state, count in noisy_counts.items():
			index = int(state, 2)
			counts_vector[index] = count
            
		mitigated_vector = np.dot(minv, counts_vector)
            
		mitigated_vector = np.maximum(mitigated_vector, 0)
		mitigated_vector = mitigated_vector * shots / np.sum(mitigated_vector)
            
		mitigated_counts = {}
		for i in range(len(mitigated_vector)):
			state = format(i, f'0{num_qubits}b')
			mitigated_counts[state] = round(mitigated_vector[i])
            
		return mitigated_counts

	@property
	def calibration_matrix(self):
		"""
		Returns the calibration matrix (and computes it if needed).
		"""
		if not hasattr(self, "_calibration_matrix"):
			self._compute_calibration_matrix()
		return self._calibration_matrix


	def _compute_calibration_matrix(self):
		"""
		Builds and stores the calibration matrix
		"""
		backend = fake_aphrodite.IQMFakeAphrodite()
		num_qubits = 6
		shots = 10000

		# Generating and running calibration circuits
		calibration_circuits_1 = self.calibration_circuits(num_qubits)
		calibration_circuits = transpile(
			calibration_circuits_1,
			backend=backend,
			initial_layout=self.layout,
			optimization_level=3
		) 
		calibration_job = backend.run(calibration_circuits, shots=shots)
		calibration_results = calibration_job.result()

		# Build calibration matrix
		self._calibration_matrix = self.mitigation_matrix(calibration_results, num_qubits)
		print("\nCalibration matrix:\n", self._calibration_matrix)

		# Calculate and print inverse matrix
		inverse_matrix = la.inv(self._calibration_matrix)
		print("\nInverse matrix:\n", inverse_matrix)



	def energy_cost_function(self, x: List[float]) -> float:	
		self.params = x
		calibration_matrix = self.calibration_matrix
		
		# Compiling with Qiskit to backend and binding parameters
		values = self.params
		bound_circuit = self.ansatz.assign_parameters(values)
		bound_circuit.measure_all()
		transpiled_job = transpile(bound_circuit, backend=self.backend, initial_layout=self.layout, optimization_level=3)
    
		mitigator = CorrelatedReadoutMitigator(calibration_matrix, qubits=self.layout)
        
		def is_diagonal(pauli_str):
			return all(p in {'I', 'Z'} for p in pauli_str)

		diagonal_terms = []
		diagonal_coeffs = []
		non_diagonal_terms = []
		non_diagonal_coeffs = []

		for p, c in zip(self.hamiltonian.paulis.to_labels(), self.hamiltonian.coeffs):
			if is_diagonal(p):
				diagonal_terms.append(p)
				diagonal_coeffs.append(c)
			else:
				non_diagonal_terms.append(p)
				non_diagonal_coeffs.append(c)

		total_mitigated_energy = 0.0
		mitigated_energy = 0.0
		energies_diag = []
    
		circuits_batch = [transpiled_job] * 2 
		batch_job = self.backend.run(circuits_batch, shots=10000)
		batch_results = batch_job.result()

		energies_diag = []
		for i in range(2):
			counts = batch_results.get_counts(i)
			energy_this_run = 0.0
			for pauli_str, coeff in zip(diagonal_terms, diagonal_coeffs):
				diag_vector = np.real(Operator.from_label(pauli_str).to_matrix().diagonal())
				expval, _ = mitigator.expectation_value(counts, diag_vector)
				energy_this_run += coeff * expval
			energies_diag.append(energy_this_run)

		mitigated_energy = np.mean(energies_diag)
		total_mitigated_energy += mitigated_energy

		self.total_energy = np.real(total_mitigated_energy)

		self.cost = self.total_energy
        
		self.costs_dictionary["iterations"] += 1
		self.costs_dictionary["parameters"] = self.params
		self.costs_dictionary["costs"].append(self.total_energy)
		print(f"Iterations: {self.costs_dictionary['iterations']} | Current cost: {self.total_energy}")
		return self.total_energy