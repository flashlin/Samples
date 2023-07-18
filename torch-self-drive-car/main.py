import os
import sys
import random
import numpy as np
import pygame

from car import Car, Action
from game import CanvasWidth, CanvasHeight, RoadMap, StartX, StartY, CarPos
from math_utils import Position
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

    def step(self, action: Action):
        self.car.control(action)

    def render(self):
        screen = self.screen
        road_map = self.road_map
        car = self.car

        screen.render_start()
        road_map.pos = Position(car.x, car.y)
        road_map.render(screen)
        car.move(screen, road_map)
        car.render(screen)
        screen.render_end()


if __name__ == '__main__':
    game = SelfDriveCarGame(silent_mode=False)
    while True:
        game.render()
        pygame.time.wait(1)
        
