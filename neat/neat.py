import random
import pickle
import copy
import itertools
import math
import multiprocessing as mp

def sigmoid(x):
    """Return the S-Curve activation of x."""
    return 1/(1+math.exp(-x))

def tanh(x):
    """Wrapper function for hyperbolic tangent activation."""
    return math.tanh(x)

def LReLU(x):
    """Leaky ReLU function for x"""
    if x >= 0:
        return x
    else:
        return 0.01 * x

def genomic_distance(a, b, c1, c2):
    """Calculate the genomic distance between two genomes."""
    a_in = set(a._genes)
    b_in = set(b._genes)

    # Does not distinguish between disjoint and excess
    matching = a_in & b_in
    disjoint = (a_in - b_in) | (b_in - a_in)
    N = len(max(a_in, b_in, key=len))

    weight_diff = 0
    for i in matching:
        weight_diff += abs(a._genes[i].weight - b._genes[i].weight)

    return c1 * len(disjoint)/N + c2 * weight_diff/len(matching)

def genomic_crossover(a, b):
    """Breed two genomes and return the child. Matching genes
    are inherited randomly, while disjoint genes are inherited
    from the fitter parent.
    """
    a_in = set(a._genes)
    b_in = set(b._genes)

    # Template genome for child
    child = Genome(a._inputs, a._outputs, a._innovations)

    # Inherit homologous gene from a random parent
    for i in a_in & b_in:
        parent = random.choice([a, b])
        child._genes[i] = copy.deepcopy(parent._genes[i])

    # Inherit disjoint/excess genes from fitter parent
    # If fitnesses are the same, inherit genes from a random parent
    disjoint_a = {}
    disjoint_b = {}
    for i in a_in - b_in:
        disjoint_a[i] = copy.deepcopy(a._genes[i])
    for i in b_in - a_in:
        disjoint_b[i] = copy.deepcopy(b._genes[i])

    if a._fitness > b._fitness:
        child._genes.update(disjoint_a)
    elif b._fitness > a._fitness:
        child._genes.update(disjoint_b)
    else:
        if random.uniform(0, 1) < 0.5:
            child._genes.update(disjoint_a)
        else:
            child._genes.update(disjoint_b)

    child._max_node = max(itertools.chain.from_iterable(child.get_edges()))
    child.reset()
    return child


class Hyperparameters(object):
    """Hyperparameter settings for the Brain object."""
    delta_threshold = 3
    c1 = 1.0
    c2 = 1.0

    max_fitness = -1
    max_generations = -1
    max_fitness_history = 15


class Gene(object):
    """A gene object representing an edge in the neural network."""
    def __init__(self, weight):
        self.weight = weight
        self.enabled = True


class Genome(object):
    """Base class for a standard genome used by the NEAT algorithm."""
    def __init__(self, inputs, outputs, innovations=[]):
        # Nodes
        self._inputs = inputs
        self._outputs = outputs

        self._unhidden = inputs+outputs
        self._max_node = inputs+outputs
        self._activations = []

        # Structure
        self._genes = {} # (innovation number : Gene)
        self._innovations = innovations

        # Performance
        self._fitness = 0
        self._adjusted_fitness = 0

    def generate(self):
        """Generate the neural network of this genome with minimal
        initial topology, i.e. (no hidden nodes). Call on genome
        creation.
        """
        # Minimum initial topology
        # Input to output only, no hidden layer
        for i in range(self._inputs):
            for j in range(self._inputs, self._unhidden):
                self.add_edge(i+1, j+1)

        self.reset()

    def forward(self, inputs, activation=sigmoid):
        """Evaluate inputs and calculate the outputs of the
        neural network via the forward propagation algorithm.
        """
        if len(inputs) != self._inputs:
            raise ValueError("Incorrect number of inputs.")

        # Set input activations
        for i in range(self._inputs):
            self._activations[i] = inputs[i]

        # Iterate through edges and perform forward propagation algorithm
        # Node Sort: INPUT -> HIDDEN -> OUTPUT (exclude INPUT though)
        nodes = itertools.chain(range(self._unhidden, self._max_node),
                                range(self._inputs, self._unhidden))
        for n in nodes:
            total = 0
            # OUTPUT = activation(sum(weight*INPUT))
            for gene in self._genes:
                (__in, __out) = self._innovations[gene]
                if n+1 == __out and self._genes[gene].enabled:
                    # Add all incoming connections to this node * their weights
                    total += self._genes[gene].weight * self._activations[__in-1]

            self._activations[n] = activation(total)

        return [self._activations[n-1] for n in range(1, self._max_node+1) if self.is_output(n)]

    def mutate(self):
        """Randomly mutate the genome to initiate variation."""
        if self.is_disabled():
            self.add_enabled()

        rand = random.uniform(0, 1)
        if rand < 0.03:
            self.add_node()
        elif 0.03 <= rand < 0.08:
            pair = self.random_pair()
            if pair not in self.get_edges():
                self.add_edge(*pair)
            else:
                self.shift_weight()
        else:
            self.shift_weight()

        self.reset()

    def add_node(self):
        """Add a new node between a randomly selected edge,
        disabling the parent edge.
        """
        self._max_node += 1
        enabled = [i for i in self._genes if self._genes[i].enabled]
        i = random.choice(enabled)
        self._genes[i].enabled = False
        (__in, __out) = self._innovations[i]

        self.add_edge(__in, self._max_node)
        self.add_edge(self._max_node, __out)

    def add_edge(self, i, j):
        """Add a new connection between existing nodes."""
        # Ignore an already present gene
        if (i, j) in self.get_edges():
            return

        # Add it to the edge database
        if (i, j) not in self._innovations:
            self._innovations.append((i, j))

        # Update this genome's genes
        inv = self._innovations.index((i, j))
        self._genes[inv] = Gene(random.uniform(0, 1)*random.choice([-1, 1]))

    def add_enabled(self):
        """Re-enable a random disabled gene."""
        disabled = [i for i in self._genes if not self._genes[i].enabled]

        if len(disabled) > 0:
            self._genes[random.choice(disabled)].enabled = True

    def shift_weight(self):
        """Randomly shift, perturb, or set one of the edge weights."""
        # Shift a randomly selected weight
        i = random.choice(list(self._genes.keys()))
        rand = random.uniform(0, 1)
        if rand <= 0.9:
            # Perturb
            self._genes[i].weight += random.uniform(0, 1)*random.choice([-1, 1])
        else:
            # New random value
            self._genes[i].weight = random.uniform(0, 1)*random.choice([-1, 1])

    def random_pair(self):
        """Generate random nodes (i, j) such that:
        1. i < j
        2. i is not an output
        3. j is not an input
        Ensures a directed acyclic graph (DAG).
        """
        i = random.choice([n for n in range(1, self._max_node+1) if not self.is_output(n)])
        j_list = [n for n in range(1, self._max_node+1) if not self.is_input(n) and n > i]

        if len(j_list) == 0:
            self.add_node()
            j = self._max_node
        else:
            j = random.choice(j_list)

        return (i, j)

    def is_input(self, node):
        """Determine if the node is an input."""
        return node <= self._inputs

    def is_output(self, node):
        """Determine if the node is an output."""
        return self._inputs < node <= self._unhidden

    def is_disabled(self):
        """Determine if all of its genes are disabled."""
        return all(self._genes[i].enabled == False for i in self._genes)

    def get_fitness(self):
        """Return the fitness of the genome."""
        return self._fitness

    def get_genes(self):
        """Return this genome's genes (innovation number : Gene)."""
        return self._genes

    def get_innovations(self):
        """Get this genome's innovation database."""
        return self._innovations

    def get_nodes(self):
        """Get the number of nodes in the network."""
        return self._max_node

    def get_edges(self):
        """Generate the network's edges given its innovation numbers."""
        return [self._innovations[i] for i in self._genes]

    def set_fitness(self, score):
        """Set the fitness score of this genome."""
        self._fitness = score

    def reset(self):
        """Reset the genome's activation and fitness values."""
        self._activations = [0 for i in range(self._max_node)]
        self._fitness = 0

    def clone(self):
        """Return a clone of the genome, maintaining internal
        reference to global innovation database.
        """
        # DON'T FORGET TO UPDATE INTERNAL REFERENCES TO OTHER OBJECTS WHEN CLONING
        clone = copy.deepcopy(self)
        clone._innovations = self._innovations
        clone._activations = [0 for i in range(clone._max_node)]
        return clone


class Specie(object):
    """A specie represents a set of genomes whose genomic distances 
    between them fall under the Brain's delta threshold.
    """
    def __init__(self, *members):
        self._members = list(members)
        self._fitness_history = []
        self._fitness_sum = 0

    def breed(self):
        """Return a child as a result of either a mutated clone
        or crossover between two parent genomes.
        """
        # Either mutate a clone or breed two random genomes
        if random.uniform(0, 1) < 0.25 or len(self._members) == 1:
            child = random.choice(self._members).clone()
            child.mutate()
        else:
            mom = random.choice(self._members)
            dad = random.choice([i for i in self._members if i != mom])
            child = genomic_crossover(mom, dad)

        return child

    def cull_genomes(self, fittest_only):
        """Exterminate the weakest genomes per specie."""
        self._members.sort(key=lambda g: g._fitness)
        if fittest_only:
            # Only keep the winning genome
            remaining = len(self._members)-1
        else:
            # Keep top 50%
            remaining = int(math.ceil(0.5*len(self._members)))

        self._members = self._members[remaining-1:]

    def get_best(self):
        """Get the member with the highest fitness score."""
        return max(self._members, key=lambda g: g._fitness)


class Brain(object):
    """Base class for a 'brain' that learns through the evolution
    of a population of genomes.
    """
    def __init__(self, inputs, outputs, population=100, hyperparams=Hyperparameters()):
        self._inputs = inputs
        self._outputs = outputs
        self._edges = [] # Edge database (INPUT, OUTPUT)

        self._species = []
        self._population = population

        # Hyper-parameters
        self._delta_threshold = hyperparams.delta_threshold
        self._max_fitness_history = hyperparams.max_fitness_history
        self._c1 = hyperparams.c1
        self._c2 = hyperparams.c2

        self._generation = 0
        self._max_generations = hyperparams.max_generations

        self._current_species = 0
        self._current_genome = 0

        self._max_fitness = hyperparams.max_fitness

        self._global_best = None

    def generate(self):
        """Generate the initial population of genomes."""
        for i in range(self._population):
            g = Genome(self._inputs, self._outputs, self._edges)
            g.generate()
            self.classify_genome(g)
        
        # Set the initial best genome
        self._global_best = self._species[0]._members[0]

    def classify_genome(self, genome):
        """Classify genomes into species via the genomic
        distance algorithm.
        """
        if len(self._species) == 0:
            # Empty population
            self._species.append(Specie(genome))
        else:
            # Compare genome against representative s[0] in each specie
            for s in self._species:
                rep =  s._members[0]
                if genomic_distance(genome, rep, self._c1, self._c2) < self._delta_threshold:
                    s._members.append(genome)
                    return

            # Doesn't fit with any other specie, create a new one
            self._species.append(Specie(genome))

    def update_fitness(self):
        """Update the adjusted fitness values of each genome."""
        for s in self._species:
            for g in s._members:
                g._adjusted_fitness = g._fitness/len(s._members)

            s._fitness_sum = sum([g._adjusted_fitness for g in s._members])
            s._fitness_history.append(s._fitness_sum)
            if len(s._fitness_history) > self._max_fitness_history:
                s._fitness_history.pop(0)

    def update_fittest(self):
        """Update the highest fitness score of the whole population."""
        top_performers = [s.get_best() for s in self._species]
        current_top = max(top_performers, key=lambda g: g._fitness)

        if current_top._fitness > self._global_best._fitness:
            self._global_best = current_top.clone()

    def evolve(self):
        """Evolve the population by eliminating the poorest performing
        genomes and repopulating with mutated children, prioritizing
        the most promising species.
        """
        self.update_fitness()
        global_fitness_sum = 0
        for s in self._species:
            global_fitness_sum += s._fitness_sum

        if global_fitness_sum == 0:
            # No progress, mutate everybody
            for s in self._species:
                if len(s._members) < 5:
                    for g in s._members:
                        g.mutate()
                else:
                    for g in s._members[1:]:
                        g.mutate()
        else:
            # Only keep the species with potential to improve
            surviving_species = []
            for s in self._species:
                if len(s._fitness_history) < self._max_fitness_history:
                    surviving_species.append(s)
                elif s._fitness_history[-1] - s._fitness_history[0] > 0.01:
                    surviving_species.append(s)
            self._species = surviving_species

            # Eliminate lowest performing genomes per specie
            for s in self._species:
                s.cull_genomes(False)

            # Repopulate
            children = []
            for i, s in enumerate(self._species):
                ratio = s._fitness_sum/global_fitness_sum
                offspring = math.floor(ratio * (self._population-self.get_population()))
                for j in range(int(offspring)):
                    children.append(s.breed())

            for g in children:
                self.classify_genome(g)

            # No species survived, repopulate based on fittest genome
            if len(self._species) == 0:
                for i in range(self._population):
                    g = self._global_best.clone()
                    g.mutate()
                    self.classify_genome(g)

        self._generation += 1

    def should_evolve(self):
        """Determine if the system should continue to evolve
        based on the maximum fitness and generation count.
        """
        self.update_fittest()
        fit = self._global_best._fitness <= self._max_fitness
        end = self._generation != self._max_generations

        return (fit or self._max_fitness == -1) and end

    def next_iteration(self):
        """Call after every evaluation of individual genomes to
        progress training.
        """
        s = self._species[self._current_species]
        if self._current_genome < len(s._members)-1:
            self._current_genome += 1
        else:
            if self._current_species < len(self._species)-1:
                self._current_species += 1
                self._current_genome = 0
            else:
                # Evolve to the next generation
                self.evolve()
                self._current_species = 0
                self._current_genome = 0

    def evaluate_parallel(self, evaluator, *args, **kwargs):
        """Evaluate the entire population on separate processes
        to progress training. The evaluator function must take a Genome
        as its first parameter and return a numerical fitness score.

        Any global state passed to the evaluator is copied and will not
        be modified at the parent process.
        """
        max_proc = mp.cpu_count()
        pool = mp.Pool(processes=max_proc)
        
        results = {}
        for i in range(len(self._species)):
            for j in range(len(self._species[i]._members)):
                results[(i, j)] = pool.apply_async(
                    evaluator, 
                    args=[self._species[i]._members[j]]+list(args), 
                    kwds=kwargs
                )

        for key in results:
            genome = self._species[key[0]]._members[key[1]]
            genome.set_fitness(results[key].get())

        self.evolve()

    def get_fittest(self):
        """Return the genome with the highest global fitness score."""
        return self._global_best

    def get_population(self):
        """Return the true (calculated) population size."""
        return sum([len(s._members) for s in self._species])

    def get_current(self):
        """Get the current genome for evaluation."""
        s = self._species[self._current_species]
        return s._members[self._current_genome]

    def get_current_species(self):
        """Get index of current species being evaluated."""
        return self._current_species

    def get_current_genome(self):
        """Get index of current genome being evaluated."""
        return self._current_genome

    def get_generation(self):
        """Get the current generation number of this population."""
        return self._generation

    def get_species(self):
        """Get the list of species and their respective member genomes."""
        return self._species

    def get_innovations(self):
        """Get this population's innovation database."""
        return self._edges

    def save(self, filename):
        """Save an instance of the population to disk."""
        with open(filename+'.neat', 'wb') as _out:
            pickle.dump(self, _out, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filename):
        """Return an instance of a population from disk."""
        with open(filename+'.neat', 'rb') as _in:
            return pickle.load(_in)
