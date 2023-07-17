import os
import sys
import random
import numpy as np
import pygame

class SelfDriveCarGame:
    display_width = 800
    display_height = 600

    def __init__(self, silent_mode=True):
        if not silent_mode:
            pygame.init()
            pygame.display.set_caption("Self Drive Car Game")
            self.screen = pygame.display.set_mode((self.display_width, self.display_height))
            self.font = pygame.font.Font(None, 36)
        else:
            self.screen = None
            self.font = None
        self.reset()

    def reset(self):
        StartX = 100
        StartY = 400

    def step(self, action):
        a = action

    def render(self):
        self.screen.fill((0, 0, 0))
        # Draw border
        pygame.draw.rect(self.screen, (255, 255, 255),
                        (100 - 2, 100 - 2, 10 + 4, 10 + 4), 2)
        pygame.display.flip()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    game = SelfDriveCarGame(silent_mode=False)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        pygame.time.wait(1)
        
