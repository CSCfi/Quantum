from typing import List
import numpy as np
from scipy.optimize import minimize, dual_annealing, basinhopping, cobyla, BFGS
from skopt import gp_minimize
from qiskit_algorithms.optimizers import SPSA

class Learning:
	def __init__(self, ansatz, hamiltonian, backend, estimator, optimizer: List[float]) -> None:
		self.optimizer = optimizer
		self.seed = None
		self.hamiltonian = hamiltonian
		self.backend = backend
		self.estimator = estimator
		self.ansatz = ansatz
		self.energy = None
		self.num_params = ansatz.num_parameters
		self.n = ansatz.num_parameters
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
		self.x0 = np.random.normal(0, 0.01, self.ansatz.num_parameters) 
		self.bounds = [(-np.pi/2, np.pi/2)] * self.num_params
		self.min_kwargs = min_kwargs if min_kwargs else dict()
		self.cost_function = self.energy_cost_function
		if optimizer == 'gp_minimize':
			self.optimizer = gp_minimize
			self.result = gp_minimize(func=self.cost_function, dimensions=self.bounds, callback=self.callback)
		elif optimizer == 'dual_annealing':
			self.optimizer = dual_annealing
			self.result = dual_annealing(func=self.cost_function, bounds=self.bounds)
		elif optimizer == 'basinhopping':
			self.optimizer = basinhopping
			self.result = basinhopping(func=self.cost_function, x0=self.x0)
		elif optimizer == 'cobyla':
			self.result = minimize(self.cost_function, self.x0, method='COBYLA', tol=1e-5)
		elif optimizer == 'BFGS':
			self.result = minimize(self.cost_function, self.x0, method='BFGS', tol=1e-5)
		elif optimizer == 'SPSA':
			spsa = SPSA(maxiter=100, learning_rate=0.02, perturbation=0.1)
			self.result = spsa.minimize(self.cost_function, self.x0)
		elif optimizer == 'SPSA2':
			spsa = SPSA(maxiter=100)
			self.result = spsa.minimize(self.cost_function, self.x0)
		
			self.params = self.result.x
			return dict(
				n=self.n,
				iter=self.iter,
				nfev=self.nfev,
				cost=self.cost,
				params=self.params,
				ansatz=self.ansatz,
				optimizer=self.optimizer.__class__.__name__,
				min_kwargs=self.min_kwargs,
				shots=self.shots)


	def energy_cost_function(self, x: List[float]) -> float:	
		self.params = x
		pub = (self.ansatz, [self.hamiltonian], [self.params])
		job = self.estimator.run(pubs=[pub])
		result = job.result() 
		self.energy = result[0].data.evs[0]

		print()
		self.cost = self.energy

		self.costs_dictionary["iterations"] += 1
		self.costs_dictionary["parameters"] = self.params
		self.costs_dictionary["costs"].append(self.energy)
		print(f"Iterations: {self.costs_dictionary['iterations']} | Current cost: {self.energy}")
		return self.energy
