import math
import random
import copy

try:
    import cPickle as pickle
except ImportError:
    import pickle

def sigmoid(x):
	"""
		Common activation function with a distinct S-Curve. It is 
	often called by a neuron, taking the sum of the products of 
	weights and activation values of its parent neurons.
	"""
	y = 1/(1+math.exp(-x))
	return y

def genomic_distance(g1, g2):
	"""
		Calculates the distance between two genomes. It is used
	to group similar neural networks together under the same 
	specie.
	"""
	e1 = g1.edges
	e2 = g2.edges

	# g2 must always be the bigger network
	if len(e1) > len(e2):
		e1, e2 = e2, e1

	N = float(len(e2)) # Size of larger genome (g2)
	D = abs(g1.max_edge-g2.max_edge) # Number of disjoint genes

	# Average weight difference
	w_diffs = 0
	w_count = 0
	for g in e1:
		g2_m = [i for i in e2 if e2[i][2] == e1[g][2]]
		if len(g2_m) > 0:
			w_diffs += abs(e1[g][1] - e2[g2_m[0]][1])
			w_count += 1

	if w_count > 0:
		W = w_diffs / float(w_count)
	else:
		W = 0

	if N > 0:
		return abs(D/N + W * 0.4)
	else:
		# They are equal because both genomes are empty
		return 0

def genomic_crossover(g1, g2):
	"""
		Breeds two genomes. Matching genes are inherited 
	from the fitter parent, while disjoint genes from 
	either parent are inherited by the child.
	"""
	new_edges = {}
	new_nodes = {}

	p1, p2 = g1, g2

	if len(p1.edges) > len(p2.edges):
		p1, p2 = g2, g1

	f = max([p1, p2], key=lambda p : p.fitness)
	m = max([p1, p2], key=lambda p : len(p.nodes))

	# Matching genes
	for g in p1.edges:
		# Find gene from p2 with matching innovation number
		p2_m = [i for i in p2.edges if p2.edges[i][2] == p1.edges[g][2]]

		if len(p2_m) > 0:
			if p1.fitness == p2.fitness:
				# If both have equal fitness, pick random parent
				random_parent = random.choice([p1, p2])
				if random_parent == p1:
					new_edges[g] = random_parent.edges[g]
				else:
					new_edges[p2_m[0]] = random_parent.edges[p2_m[0]]
			else:
				# Inherit from the more fit parent
				if f == p2:
					new_edges[p2_m[0]] = f.edges[p2_m[0]]
				else:
					new_edges[g] = f.edges[g]

	# Disjoint genes
	disjoint = {}
	for g in p1.edges:
		if not any(p2.edges[i][2] == p1.edges[g][2] for i in p2.edges):
			disjoint[g] = p1.edges[g]

	for g in p2.edges:
		if not any(p1.edges[i][2] == p2.edges[g][2] for i in p1.edges):
			disjoint[g] = p2.edges[g]

	new_edges.update(disjoint)

	# Update nodes
	for n in xrange(len(m.nodes)):
		new_nodes[n] = [[], [], 0, 0]

	for e in new_edges:
		new_nodes[e[1]][0].append(e[0])
		new_nodes[e[0]][1].append(e[1])

	for n in new_nodes:
		if n in p1.nodes:
			new_nodes[n][3] = p1.nodes[n][3]
		elif n in p2.nodes:
			new_nodes[n][3] = p2.nodes[n][3]
		else:
			new_nodes[e[1]][3] = f.nodes[e[1]][3]

	child = NeuralNetwork(p1.inputs, p2.outputs)
	child.edges = new_edges
	child.nodes = new_nodes

	return child


class NeuralNetwork(object):
	"""
		Base class for a standard neural network used by
	the NEAT algorithm.
	"""
	def __init__(self, inputs, outputs, fitness=0, cost=0):
		self.nodes = dict() # id : [[*IN], [*OUT], activation, bias]
		self.edges = dict() # edge : [weight, enabled, innovation]

		self.inputs = inputs
		self.outputs = outputs

		self.max_node = inputs + outputs
		self.max_edge = 0 # Innovation number

		# Determines the accuracy/feasibility of network
		self.fitness = fitness
		self.adjusted_fitness = 0

	def generate(self):
		"""
			Generates a new neural network. Call on
		object creation.
		"""
		for i in xrange(self.max_node):
			self.nodes[i] = [[], [], 0, random.random()*random.choice([-5, 5])] # Initial activations

	def is_input(self, node):
		"""
			Determines whether or not a node is an input.
		"""
		return node < self.inputs

	def is_output(self, node):
		"""
			Determines whether or not a node is an output.
		"""
		return self.inputs <= node < self.inputs + self.outputs

	def add_edge(self):
		"""
			Edge mutation adds a new connection between
		existing neurons.
		"""
		# Select random nodes [i, j] such that:
		# 	i != j
		#	i is not an output
		#	j is not an input
		i = random.choice([n for n in self.nodes if not self.is_output(n)])
		j = random.choice([n for n in self.nodes if not self.is_input(n) and n != i])

		if not (i, j) in self.edges:
			# Add a new edge
			self.max_edge += 1
			self.edges[(i, j)] = [random.random()*random.choice([-1, 1]), True, self.max_edge]

			# Update IN and OUT nodes
			self.nodes[j][0].append(i)
			self.nodes[i][1].append(j)

	def add_node(self):
		"""
			Neural mutation adds a new node between two
		connected nodes.
		"""
		if len(self.edges) == 0:
			self.add_edge()
			return

		# Pick a random edge (which is enabled)
		valid_edges = [i for i in self.edges.keys() if self.edges[i][1]]
		if len(valid_edges) == 0:
			return
		e = random.choice(valid_edges)

		# Add a new node
		self.nodes[self.max_node] = [[], [], 0, random.random()*random.choice([-5, 5])]
		self.edges[e][1] = False # Disable this edge because we are adding a node in-between

		# Update edge connections
		self.max_edge += 1
		self.edges[(e[0], self.max_node)] = [1.0, True, self.max_edge]

		self.max_edge += 1
		self.edges[(self.max_node, e[1])] = [self.edges[e][0], True, self.max_edge]

		# Update nodes [IN] and [OUT]
		self.nodes[self.max_node][0].append(e[0])
		self.nodes[self.max_node][1].append(e[1])

		self.nodes[e[0]][1].append(self.max_node)
		self.nodes[e[1]][0].append(self.max_node)

		self.max_node += 1

	def shift_weight(self):
		"""
			Randomly mutate one of the neural network's
		connection weights.
		"""
		if len(self.edges) == 0:
			self.add_edge()
			return

		# Pick a random edge-weight
		e = random.choice(self.edges.keys())

		# Shift one of the weights
		ch = random.choice(xrange(3))
		if ch == 0:
			self.edges[e][0] *= -1
		elif ch == 1:
			self.edges[e][0] = random.random()*random.choice([-1, 1])
		else:
			self.edges[e][0] *= random.random()+0.5

	def mutate(self):
		"""
			Randomly mutate the topology and weights of the genome.
		"""
		ch = random.random()
		if ch <= 0.3:
			self.add_node()
		elif 0.3 < ch <= 0.8:
			self.add_edge()
		else:
			self.shift_weight()

	def clone(self):
		"""
			Return a clone of a neural network.
		"""
		return copy.deepcopy(self)

	def forward(self, inputs):
		"""
			Forward propagation allows the neural network
		to perform tasks by calculating a set of outputs
		from a set of inputs.
		"""
		if len(inputs) < self.inputs:
			raise ValueError("Incorrect number of inputs.")
		
		# Reset activation values
		for n in self.nodes:
			self.nodes[n][2] = 0

		# Assign input values
		for i in xrange(self.inputs):
			self.nodes[i][2] = inputs[i]

		# Iterate through nodes		
		for n in self.nodes.keys()[self.inputs:]:
			total = 0
			for _in in self.nodes[n][0]:
				edge = self.edges[(_in, n)]
				enabled = edge[1]

				if enabled:
					total += self.edges[(_in, n)][0] * self.nodes[_in][2]

			self.nodes[n][2] = sigmoid(total + self.nodes[n][3])

		return [self.nodes[i][2] for i in xrange(self.inputs, self.inputs+self.outputs)]


class Brain(object):
	"""
		Base class for a 'brain' that begins with a designated 
	population. Gradually evaluates, culls, breeds, and mutates 
	neural networks.
	"""
	def __init__(self, inputs, outputs, 
								max_fitness=-1, 
								population=100, 
								delta_threshold=1.0, 
								initial_mutations=10, 
								max_generations=-1):
		self.species = {}
		self.population = population

		self.generation = 0
		self.max_generations = max_generations

		self.current_species = 0
		self.current_genome = 0

		self.inputs = inputs
		self.outputs = outputs

		self.initial_mutations = initial_mutations
		self.delta_threshold = delta_threshold

		self.max_fitness = max_fitness
		self.global_max_fitness = 0

	def generate(self):
		"""
			Generate a new population of neural networks. 
		Call on object instantiation.
		"""
		for i in xrange(self.population):
			g = NeuralNetwork(self.inputs, self.outputs)
			g.generate()
			for j in xrange(self.initial_mutations):
				g.mutate()
			self.classify_genome(g)

	def get_population(self):
		"""
			Returns the total population of neural networks.
		"""
		return sum([len(self.species[i]) for i in self.species])

	def classify_genome(self, genome):
		"""
			Classify genomes according to their genomic distance.
		Groups neural networks into species.
		"""
		if len(self.species) == 0:
			self.species[0] = [genome]
		else:
			classified = any(genome in self.species[i] for i in self.species)
			if not classified:
				for s in self.species:
					r = self.species[s][0]
					if genomic_distance(r, genome) < self.delta_threshold:
						self.species[s].append(genome)
						return

				self.species[len(self.species)] = [genome]
				classified = True

	def adjusted_fitness(self):
		"""
			Update the adjusted fitness values for each genome
		as well as the total adjusted fitness for each specie.

		Returns fitness sums.
		"""
		f = {}
		for n in self.species:
			s = self.species[n]
			for g in s:
				g.adjusted_fitness = g.fitness / len(s)

			f[n] = float(sum([g.adjusted_fitness for g in s]))

		return f

	def cull_species(self, only_best):
		"""
			Eliminate the poorest genomes in a species.
		"""
		for n in self.species:
			s = self.species[n]
			s.sort(reverse=True, key=lambda genome : genome.fitness)

			if only_best:
				remaining = 1
			else:
				remaining = int(math.ceil(len(s)/2.0))
			self.species[n] = s[:remaining]

	def breed(self, specie):
		"""
			Returns a child as the mutation of either an existing
		genome or the crossover between two parent genomes.
		"""
		if random.random() < 0.4 or len(specie) == 1:
			child = random.choice(specie).clone()
		else:
			mom = random.choice(specie)
			dad = random.choice([i for i in specie if i != mom])
			child = genomic_crossover(mom, dad)
		
		child.mutate()
		return child

	def evolve(self):
		"""
			Takes the top genomes of the population based on their
		individual fitness scores and discards the rest. System
		is repopulated via breeding and random mutations.
		"""
		# Exterminate weak species and genomes
		self.cull_species(False)

		fitness_sums = self.adjusted_fitness()
		global_fitness_sum = sum(fitness_sums.values())

		# Restart if population has no progress
		if global_fitness_sum == 0:
			self.species = {}
			self.generate()
			self.generation += 1
			return

		# Eliminate poor performing species
		survivors = {}
		for n in self.species:
			s = self.species[n]

			breed = math.floor(fitness_sums[n]/float(global_fitness_sum) * self.population)
			if breed > 0:
				survivors[n] = self.species[n]
		self.species = dict(zip(range(len(survivors)), survivors.values()))

		# Update new fitness sums
		fitness_sums = self.adjusted_fitness()
		global_fitness_sum = sum(fitness_sums.values())

		# Breed top genes
		children = []
		for n in self.species:
			# Breed the genomes in each species, adding them to the next generation
			s = self.species[n]

			breed = math.floor(fitness_sums[n]/float(global_fitness_sum) * self.population) - 1
			for i in range(int(breed)):
				children.append(self.breed(s))

		# Keep only the best
		self.cull_species(True)

		# Fill in the gap
		while len(children) + self.get_population() < self.population:
			s = random.choice(self.species.values())
			children.append(self.breed(s))

		# Repopulate
		for g in children:
			self.classify_genome(g)

		# Increment generation
		self.generation += 1

	def next_iteration(self):
		"""
			Used during neural network training. Call after
		every evaluation of individual neural networks.
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

	def get_fittest(self):
		"""
			Updates and returns the global maximum fitness.
		"""
		max_fitness_species = {}
		for i in self.species:
			self.species[i].sort(reverse=True, key=lambda genome: genome.fitness)
			max_fitness_species[i] = self.species[i][0].fitness

		mfs = max(max_fitness_species.values())
		if mfs > self.global_max_fitness:
			self.global_max_fitness = mfs

		return self.global_max_fitness

	def should_evolve(self):
		"""
			Determines whether or not the system should continue
		learning or evolving.
		"""
		return self.get_fittest() != self.max_fitness and self.generation != self.max_generations

	def get_current(self):
		"""
			Gets the current genome for evaluation.
		"""
		return self.species[self.current_species][self.current_genome]

	def save(self, filename):
		"""
			Save an instance of the brain to disk.
		"""
		with open(filename+'.neat', 'wb') as _out:
			pickle.dump(self, _out, pickle.HIGHEST_PROTOCOL)

	@staticmethod
	def load(filename):
		"""
			Returns an instance of a brain from disk.
		"""
		with open(filename+'.neat', 'rb') as _in:
			return pickle.load(_in)
