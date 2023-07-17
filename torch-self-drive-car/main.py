import os
import sys
import random
import numpy as np
import pygame

from game import CanvasWidth, CanvasHeight, RoadMap
from pygameGraphic import PygameGraphic


class SelfDriveCarGame:

    def __init__(self, silent_mode=True):
        if not silent_mode:
            self.screen = PygameGraphic()
            self.screen.create(screen_size=(CanvasWidth,CanvasHeight))
        else:
            self.screen = None
        self.road_map = RoadMap()
        self.reset()

    def reset(self):
        startX = 100
        startY = 400

    def step(self, action):
        a = action

    def render(self):
        self.screen.render_start()
        self.road_map.render(self.screen)
        self.screen.render_end()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()


if __name__ == '__main__':
    game = SelfDriveCarGame(silent_mode=False)
    while True:
        game.render()
        pygame.time.wait(1)
        
