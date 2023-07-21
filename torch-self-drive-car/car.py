import math
from enum import Enum
import copy

from constants import FrameWidth, FrameHeight, CenterX, CenterY, StartX, StartY
from radar import Radar
from roads import EmptyRoad, RoadMap
from math_utils import Position, update_coordinates, Line, rotate_rectangle
from pygameGraphic import IGraphic, PygameController


class Action(Enum):
    NONE = 0
    UP = 1
    LEFT = 2
    RIGHT = 3
    DOWN = 4


class CarState:
    def __init__(self):
        self.speed = 0
        self.angle = 0
        self.radar_lines = []

    def to_dict(self):
        return self.__dict__

    def clone(self):
        return copy.copy(self)


class Car:
    pos = Position(0, 0)
    controller = PygameController()
    x = 0
    y = 0
    prev_x = 0
    prev_y = 0
    state = CarState()
    prev_state = CarState()

    def __init__(self):
        self.radar = Radar()
        self.controller.create()
        self.acceleration = 0.3
        self.max_speed = 4
        self.friction = 0.02
        self.damaged = False

    def reset(self):
        self.x = StartX
        self.y = StartY
        self.state.speed = 0
        self.state.angle = 0
        self.damaged = False

    def render(self, ctx: IGraphic):
        state = self.state
        radar = self.radar
        ctx.draw_image("./assets/car1.png", self.pos, state.angle)
        radar.pos = self.pos
        radar.car_xy = Position(self.x, self.y)
        radar.car_angle = state.angle
        radar.render(ctx)
        bound_line = self.get_frame_lines()
        for line in bound_line:
            start = Position(line.start.x, line.start.y)
            end = Position(line.end.x, line.end.y)
            ctx.draw_line(Line(start, end), color="yellow", thickness=5)
        self.controller.render()

    def get_observation_info(self) -> CarState:
        info = self.state.clone()
        info.radar_lines = self.radar.get_observation_info()
        return info

    def control(self, action: Action):
        if action == Action.UP:
            self.controller.forward = True
        if action == Action.DOWN:
            self.controller.reverse = True
        if action == Action.LEFT:
            self.controller.left = True
        if action == Action.RIGHT:
            self.controller.right = True
        if action == Action.NONE:
            self.controller.forward = False
            self.controller.reverse = False
            self.controller.left = False
            self.controller.right = False

    def rollback_state(self):
        self.state = self.prev_state

    def move(self, ctx: IGraphic, road_map: RoadMap):
        state = self.state
        if self.controller.forward:
            state.speed += self.acceleration
        elif self.controller.reverse:
            state.speed -= self.acceleration

        if state.speed > self.max_speed:
            state.speed = self.max_speed
        elif state.speed < -self.max_speed / 2:
            state.speed = -self.max_speed / 2

        if state.speed > 0:
            state.speed -= self.friction
        elif state.speed < 0:
            state.speed += self.friction
        if abs(state.speed) < self.friction:
            state.speed = 0

        if self.controller.left:
            state.angle += 1
        elif self.controller.right:
            state.angle -= 1

        new_pos = update_coordinates(Position(self.x, self.y), state.angle, state.speed)
        self.x = new_pos.x
        self.y = new_pos.y

        self.collide(ctx, road_map)

    def collide(self, ctx: IGraphic, road_map: RoadMap):
        self.radar.collide(ctx, road_map)
        bound_lines = self.get_bound_lines()
        roads = road_map.roads
        for ix in range(len(roads)):
            for iy in range(len(roads[ix])):
                road = roads[ix][iy]
                collide_points = road.collide(ctx, bound_lines)
                if len(collide_points) > 0:
                    # print(f"BOOM")
                    self.x = self.prev_x
                    self.y = self.prev_y
                    self.state = self.prev_state.clone()
                    self.state.speed = 0
                    self.damaged = True
                    road.render_damaged(ctx)
                    return [road, collide_points]
        self.prev_x = self.x
        self.prev_y = self.y
        self.prev_state = self.state.clone()
        self.prev_state.radar_lines = self.radar.get_observation_info()
        return [EmptyRoad(), []]

    def get_bound_lines(self) -> list[Line]:
        [left_top, right_top, right_bottom, left_bottom] = self.get_bound_points()
        return [
            Line(left_top, right_top),
            Line(right_top, right_bottom),
            Line(right_bottom, left_bottom),
            Line(left_bottom, left_top)
        ]

    def get_bound_points(self):
        state = self.state
        left_top = Position(self.x - FrameWidth / 2, self.y - FrameHeight / 2)
        right_bottom = Position(left_top.x + FrameWidth, left_top.y + FrameHeight)
        return rotate_rectangle(left_top, right_bottom, state.angle)

    def get_frame_lines(self):
        left_top = Position(self.pos.x - FrameWidth / 2, self.pos.y - FrameHeight / 2)
        right_bottom = Position(left_top.x + FrameWidth, left_top.y + FrameHeight)
        [left_top, right_top, right_bottom, left_bottom] = rotate_rectangle(left_top, right_bottom, self.state.angle)
        return [
            Line(left_top, right_top),
            Line(right_top, right_bottom),
            Line(right_bottom, left_bottom),
            Line(left_bottom, left_top)
        ]
