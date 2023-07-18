import os
import sys
import random
import numpy as np
import pygame

from car import Car
from game import CanvasWidth, CanvasHeight, RoadMap, StartX, StartY, CarPos
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
        self.car = Car()
        self.car.x = StartX
        self.car.y = StartY
        self.car.pos = CarPos

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        self.screen.render_start()
        self.road_map.render(self.screen)
        self.car.render(self.screen)
        self.screen.render_end()


if __name__ == '__main__':
    game = SelfDriveCarGame(silent_mode=False)
    while True:
        game.render()
        pygame.time.wait(1)
        
