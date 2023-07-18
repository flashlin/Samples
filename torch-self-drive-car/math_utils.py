import math
from typing import NamedTuple


class Position(NamedTuple):
    x: int
    y: int


class Line(NamedTuple):
    start: Position
    end: Position


class Arc(NamedTuple):
    centre: Position
    radius: int
    start_angle: int
    end_angle: int


def find_two_lines_intersection(line1: Line, line2: Line):
    x1, y1 = line1.start.x, line1.start.y
    x2, y2 = line1.end.x, line1.end.y
    x3, y3 = line2.start.x, line2.start.y
    x4, y4 = line2.end.x, line2.end.y

    slope = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))
    if slope == 0:
        return None

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / slope
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / slope

    if 0 <= t <= 1 and 0 <= u <= 1:
        return Position(
            x=int(x1 + t * (x2 - x1)),
            y=int(y1 + t * (y2 - y1)),
        )
    return None


def get_arc_lines(arc: Arc) -> list[Line]:
    if arc.end_angle == 0 and arc.end_angle < arc.start_angle:
        arc.end_angle = 360

    points = []
    for angle in range(int(arc.start_angle), int(arc.end_angle) + 1, 3):
        randi = angle * (math.pi / 180)
        x = int(arc.centre.x + arc.radius * math.cos(randi))
        y = int(arc.centre.y + arc.radius * math.sin(randi))
        points.append(Position(x, y))

    lines = []
    for index in range(1, len(points)):
        start = points[index - 1]
        end = points[index]
        lines.append(Line(start=start, end=end))

    return lines
