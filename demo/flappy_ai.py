# A Flappy Bird clone that learns to play Flappy Bird
# This is a simple demonstration of the capabilities of the NEAT algorithm
# Requires Pygame to run
import sys
import os
import random
import pygame
from pygame.locals import *
import NEAT as neat


# Constants
WIDTH, HEIGHT = 640, 480
NETWORK_WIDTH = 480

# Flags
AI = True
DRAW_NETWORK = True

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)


class Bird:
    def __init__(self, x, y):
        self.dim = [20, 20]
        self.pos = [x, y]
        self.vel = [0, 0]

        self.bounce = 7
        self.gravity = 0.5
        self.terminal_vel = 10
        self.onGround = False
        self.alive = True
        self.jump = False

    def draw(self, surf):
        pygame.draw.rect(surf, WHITE, ((self.pos[0]-self.dim[0]/2, self.pos[1]-self.dim[1]/2), 
                                       (self.dim[0], self.dim[1])))

    def update(self):
        if self.vel[1] < self.terminal_vel:
            self.vel[1] += self.gravity

        self.pos[1] += self.vel[1]

        if self.jump:
            self.vel[1] = -self.bounce

        self.jump = False
    
    def colliding(self, pipe):
        x = abs(self.pos[0]-pipe.pos[0]) < (self.dim[0]+pipe.dim[0])/2.0
        y = abs(self.pos[1]-pipe.pos[1]) < (self.dim[1]+pipe.dim[1])/2.0
        if x and y:
            return True
        return False

    def kill(self):
        self.alive = False


class Pipe:
    def __init__(self, x, y):
        self.dim = [50, 480]
        self.pos = [x, y]
        self.vel = [-2, 0]

        self.alive = True

    def draw(self, surf):
        pygame.draw.rect(surf, YELLOW, ((self.pos[0]-self.dim[0]/2, self.pos[1]-self.dim[1]/2), 
                                        (self.dim[0], self.dim[1])))

    def update(self):
        self.pos[0] += self.vel[0]
        if self.pos[0] < -self.dim[0]:
            self.kill()

    def kill(self):
        self.alive = False


def generate_pipes(pipes):
    # Generate a new random pair of pipes to the world
    y = random.randint(150, HEIGHT-150)
    th = 150

    p1 = Pipe(WIDTH, y-(th/2+HEIGHT/2))
    p2 = Pipe(WIDTH, y+(th/2+HEIGHT/2))
    pipes.extend([p1, p2])

def generate_net(current, nodes):
    # Generate the neural network to be displayed
    for i in range(1, current.get_nodes()+1):
        if current.is_input(i):
            color = (0, 0, 255)
            x = 50
            y = 140 + i*60
        elif current.is_output(i):
            color = (255, 0, 0)
            x = NETWORK_WIDTH-50
            y = HEIGHT/2
        else:
            color = (0, 0, 0)
            x = random.randint(NETWORK_WIDTH/3, int(NETWORK_WIDTH * (2.0/3)))
            y = random.randint(20, HEIGHT-20)
        nodes.append([(int(x), int(y)), color])

def render_net(current, display, nodes):
    # Render the current neural network
    for j in current.get_genes():
        if j[2]: # Enabled or disabled edge
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        points = current.get_innovations()[j[0]]
        pygame.draw.line(display, color, nodes[points[0]-1][0], nodes[points[1]-1][0], 3)

    for n in nodes:
        pygame.draw.circle(display, n[1], n[0], 7)

def main():
    # Main game function
    pygame.init()

    display = pygame.display.set_mode((WIDTH+NETWORK_WIDTH, HEIGHT), 0, 32)
    network_display = pygame.Surface((NETWORK_WIDTH, HEIGHT))

    clock = pygame.time.Clock()
    timer = 0

    pipes = []
    ahead = []

    # Load the bird's brain
    if os.path.isfile('flappy_bird.neat'):
        brain = neat.Brain.load('flappy_bird')
    else:
        brain = neat.Brain(4, 1)
        brain.generate()

    current = brain.get_current()
    inputs = [0, 0, 0, 0]
    output = 0
    nodes = []

    player = Bird(WIDTH/4, HEIGHT/2)
    generate_net(current, nodes)

    while True:
        # Main loop
        display.fill(BLACK)
        network_display.fill(WHITE)

        # Game logic
        if player.alive:
            player.draw(display)
            player.update()

            timer += 1
            if timer%150 == 0:
                generate_pipes(pipes)

            for p in pipes:
                p.draw(display)
                if p.alive:
                    p.update()
                    if player.colliding(p):
                        player.kill()
                else:
                    pipes.remove(p)

            if player.pos[1] >= HEIGHT-player.dim[1]/2 or player.pos[1] <= 0:
                player.kill()
        else:
            timer = 0
            pipes = []
            nodes = []

            if AI:
                # Save the bird's brain
                brain.save('flappy_bird')
                brain.next_iteration()
                current = brain.get_current()

                # Restart the game
                generate_net(current, nodes)
                player = Bird(WIDTH/4, HEIGHT/2)

            else:
                player = Bird(WIDTH/4, HEIGHT/2)

        # Training brain
        if AI:
            ahead = [p for p in pipes if p.pos[0]-player.pos[0] >= 0][:2]
            if len(ahead) > 0:
                # inputs = [x-dist, player height, top-pipe, bottom-pipe]
                inputs = [(ahead[0].pos[0]-player.pos[0])/WIDTH, 
                          (player.pos[1])/HEIGHT,
                          (ahead[0].pos[1]+ahead[0].dim[1]/2)/HEIGHT,
                          (ahead[1].pos[1]-ahead[1].dim[1]/2)/HEIGHT]
            else:
                inputs = [0, player.pos[1]/HEIGHT, 0, 0]

            if brain.should_evolve():
                output = current.forward(inputs)[0]
                player.jump = (output <= 0.5)
                current.set_fitness(timer)

                if DRAW_NETWORK:
                    render_net(current, network_display, nodes)

        # Handle events
        for e in pygame.event.get():
            if e.type == QUIT:
                sys.exit()
            if e.type == KEYDOWN:
                if e.key == K_SPACE and not AI:
                    player.jump = True

        # Update display and its caption
        display.blit(network_display, (WIDTH, 0))
        pygame.display.set_caption("GNRTN : {0}; SPC : {1}; CRRNT : {2}; FIT : {3}; MXFIT : {4}; OUT : {5} IN : {6}".format(
                                            brain.get_generation()+1, 
                                            brain.get_current_species()+1, 
                                            brain.get_current_genome()+1,
                                            timer,
                                            brain.get_fittest(),
                                            round(output, 3),
                                            [round(i, 3) for i in inputs]
        ))
        pygame.display.update()

        # Uncomment this to cap the framerate
        # clock.tick(100)

if __name__ == "__main__":
    main()