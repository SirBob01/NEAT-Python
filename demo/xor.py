import os
from neat import neat

def fitness(expected, output):
    """Fitness is calculated as 1 - sum of (expected - output)**2"""
    s = 0
    for i in range(4):
        s += (expected[i] - output[i])**2
    return 1-s


def main():
    brain = neat.Brain(2, 1, population=150, max_generations=300)
    brain.generate()
    
    print("Training...")
    last_gen = None
    while brain.should_evolve():
        genome = brain.get_current()

        f1 = genome.forward([0.0, 0.0])[0]
        f2 = genome.forward([1.0, 0.0])[0]
        f3 = genome.forward([0.0, 1.0])[0]
        f4 = genome.forward([1.0, 1.0])[0]
        genome.set_fitness(fitness([0.0, 1.0, 1.0, 0.0], [f1, f2, f3, f4]))

        # Print training progress
        current_gen = brain.get_generation()
        if last_gen != current_gen:
            current_best = brain.get_fittest()
            print("Current Accuracy: {:.2f}% | Generation {}/{}".format(
                current_best.get_fitness() * 100, 
                current_gen + 1, 
                brain._max_generations
            ))
            last_gen = current_gen
        brain.next_iteration()

    best = brain.get_fittest()
    f1 = best.forward([0.0, 0.0])[0]
    f2 = best.forward([1.0, 0.0])[0]
    f3 = best.forward([0.0, 1.0])[0]
    f4 = best.forward([1.0, 1.0])[0]
    fit = fitness([0.0, 1.0, 1.0, 0.0], [f1, f2, f3, f4])

    print("Accuracy: {:.2f}%".format(fit * 100))
    print("0 ^ 0 = {:.3f}".format(f1))
    print("1 ^ 0 = {:.3f}".format(f2))
    print("0 ^ 1 = {:.3f}".format(f3))
    print("1 ^ 1 = {:.3f}".format(f4))

if __name__ == "__main__":
    main()