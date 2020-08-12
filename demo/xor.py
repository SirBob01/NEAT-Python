from neat import neat

def fitness(expected, output):
    """Fitness is calculated as 1 - sum of (expected - output)**2"""
    s = 0
    for i in range(4):
        s += (expected[i] - output[i])**2
    return 1-s


def main():
    brain = neat.Brain(2, 1, population=100, max_generations=100000)
    brain.generate()

    print("Training...")
    while brain.should_evolve():
        genome = brain.get_current()
        f1 = genome.forward([0.0, 0.0])[0]
        f2 = genome.forward([1.0, 0.0])[0]
        f3 = genome.forward([0.0, 1.0])[0]
        f4 = genome.forward([1.0, 1.0])[0]
        fit = fitness([0.0, 1.0, 1.0, 0.0], [f1, f2, f3, f4])

        genome.set_fitness(fit)
        brain.next_iteration()

        if fit > brain.get_fittest():
            print(fit)


    print(genome.forward([0.0, 0.0])) # 0
    print(genome.forward([1.0, 0.0])) # 1
    print(genome.forward([0.0, 1.0])) # 1
    print(genome.forward([1.0, 1.0])) # 0

if __name__ == "__main__":
    main()