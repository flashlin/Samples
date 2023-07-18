import math

from game import RadarColor, CarHeight, RadarLineLength, RadarLineCount
from math_utils import Position, rotate_points, Line
from pygameGraphic import IGraphic
from roads import RoadMap


class RadarLine:
    def __init__(self):
        self.pos = Position(0, 0)
        self.car_xy = Position(0, 0)
        self.car_angle = 0
        self.angle = 0
        self.distance = 0

    def render(self, ctx: IGraphic):
        ctx.draw_line(self.get_draw_line(True), color=RadarColor, thickness=3)

    def render_damaged(self, ctx: IGraphic, point: Position):
        start, end = self.get_bound_line()
        distance = round(math.sqrt((start.x - point.x) ** 2 + (start.y - point.y) ** 2))
        self.distance = distance
        draw_line = self.get_draw_line(False)
        end_pos = Position(draw_line.start.x, draw_line.start.y + distance)
        rotated_points = rotate_points(self.pos, self.car_angle + self.angle, [draw_line.start, end_pos])
        radar_line = Line(rotated_points[0], rotated_points[1])
        ctx.draw_line(radar_line, color="red", thickness=5)

    def get_draw_line(self, rotate: bool) -> Line:
        start = Position(self.pos.x, self.pos.y + CarHeight / 2 - 20)
        end = Position(start.x, start.y + RadarLineLength)
        if rotate:
            rotated_points = rotate_points(self.pos, self.car_angle + self.angle, [start, end])
            return Line(rotated_points[0], rotated_points[1])
        return Line(start, end)

    def get_bound_line(self) -> Line:
        start = Position(self.car_xy.x, self.car_xy.y + CarHeight / 2 - 20)
        end = Position(start.x, start.y + RadarLineLength)
        rotated_points = rotate_points(self.car_xy, self.car_angle + self.angle, [start, end])
        return Line(rotated_points[0], rotated_points[1])

    def is_my(self, point: Position) -> bool:
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


def generate_angles(count: int, angle: int) -> list[int]:
    angles = []
    start_angle = 0
    counter = 0
    if count % 2 != 0:
        angles.append(start_angle)
        counter = 1

    for i in range(1, count):
        if i % 2 == 1:
            angles.append(start_angle - counter * angle)
        else:
            angles.append(start_angle + counter * angle)
            counter += 1

    return angles


class Radar:
    radar_lines: list[RadarLine] = []

    def __init__(self):
        self.pos = Position(0, 0)
        self.car_angle = 0
        self.car_xy = Position(0, 0)
        for radar_angle in generate_angles(RadarLineCount, 30):
            radar_line = RadarLine()
            radar_line.angle = radar_angle
            self.radar_lines.append(radar_line)

    def render(self, ctx: IGraphic):
        for radar_line in self.radar_lines:
            radar_line.pos = self.pos
            radar_line.car_xy = self.car_xy
            radar_line.car_angle = self.car_angle
            radar_line.render(ctx)

    def collide(self, ctx: IGraphic, road_map: RoadMap):
        for index, radar_line in enumerate(self.get_bound_lines()):
            radar = self.radar_lines[index]
            road, collide_radar_points = road_map.collide(ctx, [radar_line])
            if collide_radar_points:
                radar.render_damaged(ctx, collide_radar_points[0])
            else:
                radar.distance = RadarLineLength

    def get_bound_lines(self) -> list[Line]:
        bound_lines = []
        for radar_line in self.radar_lines:
            radar_line.pos = self.pos
            radar_line.car_xy = self.car_xy
            radar_line.car_angle = self.car_angle
            bound_lines.append(radar_line.get_bound_line())
        return bound_lines

    def render_damaged(self, ctx: IGraphic, point: Position):
        for radar_line in self.radar_lines:
            radar_line.pos = self.pos
            radar_line.car_xy = self.car_xy
            radar_line.car_angle = self.car_angle
            if radar_line.is_my(point):
                radar_line.render_damaged(ctx, point)
