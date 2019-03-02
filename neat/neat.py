import math
import random
import pickle
import copy
import itertools

def sigmoid(x):
	"""Return the S-Curve activation of x."""
	return 1/(1+math.exp(-x))

def genomic_distance(a, b):
	"""Calculate the genomic distance between two genomes."""
	a_in = set([i[0] for i in a.genes])
	b_in = set([i[0] for i in b.genes])

	# Does not distinguish between disjoint and excess
	matching = a_in & b_in
	disjoint = (a_in - b_in) | (b_in - a_in)
	N = len(max(a_in, b_in, key=len))

	weight_diff = 0
	for i in range(len(matching)):
		if a.genes[i][0] in matching:
			weight_diff += abs(a.genes[i][1]-b.genes[i][1])

	return len(disjoint)/N + 0.4*weight_diff/len(matching)

def genomic_crossover(a, b):
	"""Breed two genomes and return the child. Matching genes 
	are inherited randomly, while disjoint genes are inherited 
	from the fitter parent.
	"""
	a_in = set([i[0] for i in a.genes])
	b_in = set([i[0] for i in b.genes])

	# Template genome for child
	child = Genome(a.inputs, a.outputs, a.innovations)

	# Inherit homologous gene from a random parent
	for i in a_in & b_in:
		parent = random.choice([a, b])
		for g in parent.genes:
			if g[0] == i:
				child.genes.append(g)

	# Inherit disjoint/excess genes from fitter parent
	disjoint_a = [i for i in a.genes if i[0] in a_in - b_in]
	disjoint_b = [i for i in b.genes if i[0] in b_in - a_in]

	if a.fitness > b.fitness:
		child.genes.extend(disjoint_a)
	elif b.fitness > a.fitness:
		child.genes.extend(disjoint_b)
	else:
		for d in (disjoint_a + disjoint_b):
			if random.uniform(0, 1) > 0.5:
				child.genes.append(d)

	child.max_node = max(itertools.chain.from_iterable(child.get_edges()))
	return child


class Genome(object):
	"""Base class for a standard genome used by the NEAT algorithm."""
	def __init__(self, inputs, outputs, innovations=[]):
		# Nodes
		self.inputs = inputs
		self.outputs = outputs

		self.unhidden = inputs+outputs
		self.max_node = inputs+outputs
		self.activations = []

		# Structure
		self.genes = [] # (innovation number, weight, enabled)
		self.innovations = innovations

		# Performance
		self.fitness = 0
		self.adjusted_fitness = 0
	
	def generate(self):
		"""Generate the neural network of this genome with minimal 
		initial topology, i.e. (no hidden nodes). Call on genome 
		creation.
		"""
		# Minimum initial topology
		# Input to output only, no hidden layer
		for i in range(self.inputs):
			for j in range(self.inputs, self.unhidden):
				self.add_edge(i+1, j+1)

		self.reset()

	def forward(self, inputs):
		"""Evaluate inputs and calculate the outputs of the 
		neural network via the forward propagation algorithm.
		"""
		if len(inputs) < self.inputs:
			raise ValueError("Incorrect number of inputs.")

		# Set input activations
		for i in range(self.inputs):
			self.activations[i] = inputs[i]

		# Iterate through edges and perform forward propagation algorithm
		# Node Sort: INPUT -> HIDDEN -> OUTPUT (exclude INPUT though)
		nodes = itertools.chain(range(self.unhidden, self.max_node), 
								range(self.inputs, self.unhidden))
		for n in nodes:
			total = 0
			# OUTPUT = sigmoid(sum(weight*INPUT))
			for gene in self.genes:
				(__in, __out) = self.innovations[gene[0]]
				weight = gene[1]
				enabled = gene[2]
				if n+1 == __out and enabled:
					total += weight * self.activations[__in-1]

			self.activations[n] = sigmoid(total)

		return [self.activations[n-1] for n in range(1, self.max_node+1) if self.is_output(n)]

	def mutate(self):
		"""Randomly mutate the genome to initiate variation."""
		if self.is_disabled():
			self.add_enabled()

		# Has a chance of nothing happening
		if random.uniform(0, 1) < 0.5:
			self.add_edge(*self.random_pair())
		else:
			self.add_node()
		self.shift_weight()

		self.reset()

	def random_pair(self):
		"""Generate random nodes (i, j) such that:
		1. i < j
		2. i is not an output
		3. j is not an input

		Ensures a directed acyclic graph (DAG).
		"""
		i = random.choice([n for n in range(1, self.max_node+1) if not self.is_output(n)])
		j_list = [n for n in range(1, self.max_node+1) if not self.is_input(n) and n > i]
		
		if len(j_list) == 0:
			self.add_node()
			j = self.max_node
		else:
			j = random.choice(j_list)

		return (i, j)

	def add_node(self):	
		"""Add a new node between a randomly selected edge, 
		disabling the parent edge.
		"""
		enabled = [g for g in self.genes if g[2]]
		gene = random.choice(enabled)
		gene[2] = False # Disable the gene

		self.max_node += 1
		(i, j) = self.innovations[gene[0]]

		self.add_edge(i, self.max_node)
		self.add_edge(self.max_node, j)

	def add_edge(self, i, j):
		"""Add a new connection between existing nodes."""
		# Ignore an already present gene
		if (i, j) in self.get_edges():
			return

		# Add it to the edge database
		if (i, j) not in self.innovations:
			self.innovations.append((i, j))

		# Update this genome's genes
		inv = self.innovations.index((i, j))
		self.genes.append([inv, random.uniform(0, 1)*random.choice([-1, 1]), True])

	def add_enabled(self):
		"""Re-enable a random disabled gene."""
		disabled = [g for g in self.genes if not g[2]]
		
		if len(disabled) > 0:
			random.choice(disabled)[2] = True

	def shift_weight(self):
		"""Randomly shift, perturb, or set one of the edge weights."""
		# Shift a randomly selected weight
		gene = random.choice(self.genes)
		if random.uniform(0, 1) <= 0.2:
			# Perturb
			gene[1] += 0.1*random.choice([-1, 1])
		elif 0.2 < random.uniform(0, 1) <= 0.5:
			# New random value
			gene[1] = random.uniform(0, 1)*random.choice([-1, 1])
		else:
			# Reflect
			gene[1] *= -1

		# Keep within [-1.0, 1.0]
		if gene[1] < 0:
			gene[1] = max(-1.0, gene[1])
		else:
			gene[1] = min(1.0, gene[1])

	def is_input(self, node):
		"""Determine if the node is an input."""
		return node <= self.inputs

	def is_output(self, node):
		"""Determine if the node is an output."""
		return self.inputs < node <= self.unhidden

	def is_disabled(self):
		"""Determine if all of its genes are disabled."""
		return all(g[2] == False for g in self.genes)

	def get_fitness(self):
		"""Return the fitness of the genome."""
		return self.fitness

	def get_edges(self):
		"""Generate the network's edges given its innovation numbers."""
		return [self.innovations[g[0]] for g in self.genes]

	def set_fitness(self, score):
		"""Set the fitness score of this genome."""
		self.fitness = score

	def reset(self):
		"""Reset the genome's activation and fitness values."""
		self.activations = [0 for i in range(self.max_node)]
		self.fitness = 0

	def clone(self):
		"""Return a clone of the genome, maintaining internal 
		reference to global innovation database.
		"""
		# DON'T FORGET TO UPDATE INTERNAL REFERENCES TO OTHER OBJECTS WHEN CLONING
		clone = copy.deepcopy(self)
		clone.innovations = self.innovations
		return clone


class Brain(object):
	"""Base class for a 'brain' that learns through the evolution
	of a population of genomes.
	"""
	def __init__(self, inputs, outputs,	population=100,  max_fitness=-1, max_generations=-1, delta_threshold=0.5):
		self.inputs = inputs
		self.outputs = outputs
		self.edges = [] # Edge database (INPUT, OUTPUT)
		
		self.species = []
		self.population = population

		self.delta_threshold = delta_threshold
		self.cull_percent = 0.75

		self.generation = 0
		self.max_generations = max_generations

		self.current_species = 0
		self.current_genome = 0

		self.max_fitness = max_fitness
		self.global_max_fitness = 0
		self.fitness_sums = []

	def generate(self):
		"""Generate the initial population of genomes."""
		for i in range(self.population):
			g = Genome(self.inputs, self.outputs, self.edges)
			g.generate()
			self.classify_genome(g)

	def classify_genome(self, genome):
		"""Classify genomes into species via the genomic 
		distance algorithm.
		"""
		if len(self.species) == 0:
			# Empty population
			self.species.append([genome])
		else:
			# Compare genome against representative s[0] in each specie
			for s in self.species:
				rep =  s[0]
				if genomic_distance(genome, rep) < self.delta_threshold:
					s.append(genome)
					return

			# Doesn't fit with any other specie, create a new one
			self.species.append([genome])

	def update_fitness(self):
		"""Update the adjusted fitness values of each genome."""
		for s in self.species:
			if self.get_population() == len(s):
				sh = 1
			else:
				sh = self.get_population()-len(s)

			for g in s:
				g.adjusted_fitness = g.fitness/float(sh)

			self.fitness_sums.append(sum([g.adjusted_fitness for g in s]))

	def breed(self, specie):
		"""Return a child as a result of either a mutated clone
		or crossover between two parent genomes.
		"""
		# Either mutate a clone or breed two random genomes
		if random.uniform(0, 1) < 0.4 or len(specie) == 1:
			child = random.choice(specie).clone()
		else:
			mom = random.choice(specie)
			dad = random.choice([i for i in specie if i != mom])
			child = genomic_crossover(mom, dad)

		child.mutate()
		return child

	def evolve(self):
		"""Evolve the population by eliminating the poorest performing
		genomes and repopulating with mutated children, prioritizing
		the most promising species.
		"""
		self.update_fitness()
		global_fitness_sum = float(sum(self.fitness_sums))

		if global_fitness_sum == 0:
			# No progress, mutate everybody
			for s in self.species:
				for g in s:
					g.mutate()
		else:
			# 1. Eliminate lowest performing genomes per specie
			# 2. Repopulate
			self.cull_genomes(False)
			
			children = []
			for i, s in enumerate(self.species):
				ratio = self.fitness_sums[i]/global_fitness_sum
				offspring = math.floor(ratio * (self.population-self.get_population()))
				for j in range(int(offspring)):
					children.append(self.breed(s))

			for g in children:
				self.classify_genome(g)

		self.generation += 1

	def cull_genomes(self, fittest_only):
		"""Exterminate the weakest genomes per specie."""
		for i, s in enumerate(self.species):
			s.sort(key=lambda g: g.fitness)

			if fittest_only:
				# Only keep the winning genome
				remaining = len(s)-1
			else:
				# Keep top 25%
				remaining = int(math.ceil(self.cull_percent*len(s)))
			self.species[i] = s[remaining-1:]

	def get_population(self):
		"""Return the true (calculated) population size."""
		return sum([len(s) for s in self.species])

	def get_fittest(self):
		"""Get the highest fitness score of the whole population."""
		best = [max(s, key=lambda g: g.fitness).fitness for s in self.species]
		
		if max(best) > self.global_max_fitness:
			self.global_max_fitness = max(best)

		return self.global_max_fitness

	def should_evolve(self):
		"""Determine if the system should continue to evolve
		based on the maximum fitness and generation count.
		"""
		fit = self.get_fittest() != self.max_fitness 
		end = self.generation != self.max_generations

		return fit and end

	def next_iteration(self):
		"""Call after every evaluation of individual genomes to
		progress training.
		"""
		if self.current_genome < len(self.species[self.current_species])-1:
			self.current_genome += 1
		else:
			if self.current_species < len(self.species)-1:
				self.current_species += 1
				self.current_genome = 0
			else:
				# Evolve to the next generation
				self.evolve()
				self.current_species = 0
				self.current_genome = 0

	def get_current(self):
		"""Get the current genome for evaluation."""
		return self.species[self.current_species][self.current_genome]

	def save(self, filename):
		"""Save an instance of the population to disk."""
		with open(filename+'.neat', 'wb') as _out:
			pickle.dump(self, _out, pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def load(filename):
		"""Return an instance of a population from disk."""
		with open(filename+'.neat', 'rb') as _in:
			return pickle.load(_in)
