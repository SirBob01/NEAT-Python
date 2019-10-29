# NEAT-Python

NEAT (NeuroEvolution of Augmenting Topologies) is a _genetic algorithm_ 
developed by Ken Stanley that applies genetic algorithms to machine learning.

1. Generates a population of genomes (neural networks)
2. Groups genomes into species based on their _genomic distances_
3. Evaluates the _fitness score_ of each genome
4. Breeds and mutates the best genomes over the course of generations

This implementation is a modified version of the algorithm written in Python.

[Here](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf) is the original paper.

# Dependencies

None. This is written in pure Python. :)

# Installation

To install via pip, simply enter `pip install git+https://github.com/SirBob01/NEAT-Python.git` on the command prompt.

# Basic Usage

Import the NEAT module.
```py
from neat import neat
```

Generate the genomic population of a new brain, denoting the number of inputs and outputs respectively, as well as its population count, for its base parameters.
```py
brain = neat.Brain(3, 2, population=100) # Takes 3 inputs, produces 2 outputs
brain.generate()
```

In evaluating the genomes of a brain, you may use the following format:
```py
while brain.should_evolve():
    genome = brain.get_current()
    output = genome.forward([0.3, 0.1, 0.25])
    
    genome.set_fitness(some_function(output))
    
    brain.next_iteration() # Next genome to be evaluated
```

The brain's `.should_evolve()` function determines whether or not to continue evaluating genomes based on the maximum number of generations or fitness score to be achieved.

A genome's `.forward()` function takes a list of input values and produces a list of output values. These outputs may be evaluated by a fitness function and the fitness score of this current genome may be updated via the genome's `.set_fitness()` method.

A brain and all its neural networks can be saved to disk and loaded. Files are automatically read and saved as `.neat` files.
```py
brain.save('filename')
loaded_brain = neat.Brain.load('filename') # Static method
```

Read NEAT's doc-strings for more information on the module's classes and methods.

# TODO
- Modify the mutation probabilities (perhaps allow custom probabilities)
- Fine tune population parameters, such as the delta threshold