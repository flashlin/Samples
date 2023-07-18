import math

from game import RadarColor, CarHeight, RadarLineLength
from math_utils import Position, rotate_points, Line
from pygameGraphic import IGraphic


class RadarLine:
    def __init__(self):
        self.pos = Position(0, 0)
        self.car_xy = Position(0, 0)
        self.car_angle = 0
        self.angle = 0
        self.distance = 0

    def render(self, ctx: IGraphic):
        ctx.draw_line(self.get_draw_line(), color=RadarColor, thickness=3)

    def render_damaged(self, ctx: IGraphic, point: Position):
        start, end = self.get_bound_line()
        distance = round(math.sqrt((start.x - point.x) ** 2 + (start.y - point.y) ** 2))
        self.distance = distance
        draw_line = self.get_draw_line()
        end_pos = Position(draw_line.start.x, draw_line.start.y - distance)
        rotated_points = rotate_points(self.pos, self.car_angle + self.angle, [draw_line.start, end_pos])
        radar_line = Line(rotated_points[0], rotated_points[1])
        ctx.draw_line(radar_line, color="yellow", thickness=3)

    def get_draw_line(self):
        start = Position(self.pos.x, self.pos.y - CarHeight / 2 + 20)
        end = Position(start.x, start.y - RadarLineLength)
        return Line(start, end)

    def get_bound_line(self):
        start = Position(self.car_xy.x, self.car_xy.y - CarHeight / 2 + 20)
        end = Position(start.x, start.y - RadarLineLength)
        rotated_points = rotate_points(self.car_xy, self.car_angle + self.angle, [start, end])
        return rotated_points[0], rotated_points[1]

    def is_my(self, point: Position):
        start, end = self.get_bound_line()
        slope = (end.y - start.y) / (end.x - start.x)
        test_line = Line(start, point)
        test_slope = (test_line.end.y - test_line.start.y) / (test_line.end.x - test_line.start.x)
        if self.compare_with_tolerance(slope, test_slope):
            return True
        self.distance = RadarLineLength
        return False

    @staticmethod
    def compare_with_tolerance(a, b):
        return abs(a - b) < 0.0000001
    