import os
import sys
import random
import numpy as np
import pygame

from car import Car, Action, CarState
from constants import CanvasWidth, CanvasHeight, StartX, StartY, CarPos, CenterX, CenterY, FrameWidth, FrameHeight
from roads import RoadMap
from math_utils import Position, convert_obj_to_space
from pygameGraphic import PygameGraphic


class SelfDriveCarGame:

    def __init__(self, silent_mode=True):
        if not silent_mode:
            self.screen = PygameGraphic()
            self.screen.create(screen_size=(CanvasWidth, CanvasHeight))
        else:
            self.screen = None
        self.reset()
        self.road_map = RoadMap()
        self.car = Car()
        self.car.pos = Position(CenterX, CenterY)
        self.car.set_pos(StartX, StartY)

    def reset(self):
        pass

    def step(self, action: Action) -> bool:
        self.car.control(action)
        self.car.move(self.screen, self.road_map)
        return self.car.damaged

    def rollback_step(self):
        self.car.rollback_state()

    def get_observation_space(self) -> list[int]:
        info = self.car.get_observation_info()
        return convert_obj_to_space(info)

    def get_observation_info(self) -> CarState:
        info = self.car.get_observation_info()
        return info

    def render(self):
        screen = self.screen
        road_map = self.road_map
        car = self.car

        screen.render_start()
        road_map.pos = Position(CenterX - car.state.x, CenterY - car.state.y)
        road_map.render(screen)
        car.move(screen, road_map)
        car.render(screen)
        screen.render_end()


if __name__ == '__main__':
    game = SelfDriveCarGame(silent_mode=False)
    while True:
        game.render()
        pygame.time.wait(1)
