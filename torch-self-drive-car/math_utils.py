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
        arc = arc._replace(end_angle=360)
    points = []
    for angle in range(arc.start_angle, arc.end_angle + 1, 3):
        randi = angle * (math.pi / 180)
        x = round(arc.centre.x + arc.radius * math.cos(randi))
        y = round(arc.centre.y + arc.radius * math.sin(randi))
        points.append(Position(x, y))

    lines = []
    for index in range(1, len(points)):
        start = points[index - 1]
        end = points[index]
        lines.append(Line(start=start, end=end))

    return lines


def update_coordinates(pos: Position, angle: int, distance: int) -> Position:
    theta = -angle * math.pi / 180
    x = round(pos.x + distance * math.sin(theta))
    y = round(pos.y - distance * math.cos(theta))
    return Position(x, y)


def rotate_points(center: Position, angle: int, points: list[Position]) -> list[Position]:
    theta = -angle * (math.pi / 180)
    rotated_points = []
    for point in points:
        x = point.x - center.x
        y = point.y - center.y
        rotated_x = round(x * math.cos(theta) + y * math.sin(theta) + center.x)
        rotated_y = round(x * math.sin(theta) - y * math.cos(theta) + center.y)
        rotated_points.append(Position(rotated_x, rotated_y))
    return rotated_points


def rotate_rectangle(left_top: Position, right_bottom: Position, angle: int) -> list[Position]:
    x1, y1 = left_top.x, left_top.y
    x2, y2 = right_bottom.x, right_bottom.y

    # Calculate the center of the rectangle
    center = Position(round((x1 + x2) / 2), round((y1 + y2) / 2))

    # Define the four vertices of the rectangle
    points = [
        Position(x1, y1),
        Position(x2, y1),
        Position(x2, y2),
        Position(x1, y2)
    ]

    return rotate_points(center, angle, points)
