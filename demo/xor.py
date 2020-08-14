import os
from neat import neat

def fitness(expected, output):
    """Fitness is calculated as 1 - sum of (expected - output)**2"""
    s = 0
    for i in range(4):
        s += (expected[i] - output[i])**2
    return 1-s


def main():
    if os.path.isfile('xor.neat'):
        brain = neat.Brain.load('xor')
    else:
        brain = neat.Brain(2, 1, population=100, max_generations=1000)
        brain.generate()
    
    if brain.should_evolve():
        print("Training...")
    else:
        print("Final model...")

    while brain.should_evolve():
        genome = brain.get_current()
        f1 = genome.forward([0.0, 0.0])[0]
        f2 = genome.forward([1.0, 0.0])[0]
        f3 = genome.forward([0.0, 1.0])[0]
        f4 = genome.forward([1.0, 1.0])[0]
        fit = fitness([0.0, 1.0, 1.0, 0.0], [f1, f2, f3, f4])

        genome.set_fitness(fit)
        brain.next_iteration()

        if fit > brain.get_fittest().get_fitness():
            # Values should progressively approach [0, 1, 1, 0]
            print("[{:.3f} {:.3f} {:.3f} {:.3f}] | Acc.: {:.2f}% | Gen.: {:d}".format(
                f1, f2, f3, f4, (fit * 100.0), brain.get_generation())
            )

        brain.save('xor')

    best = brain.get_fittest()
    print("Accuracy: {:.2f}%".format(best.get_fitness() * 100))
    print("0 ^ 0 = {:.3f}".format(best.forward([0.0, 0.0])[0]))
    print("1 ^ 0 = {:.3f}".format(best.forward([1.0, 0.0])[0]))
    print("0 ^ 1 = {:.3f}".format(best.forward([0.0, 1.0])[0]))
    print("1 ^ 1 = {:.3f}".format(best.forward([1.0, 1.0])[0]))

if __name__ == "__main__":
    main()