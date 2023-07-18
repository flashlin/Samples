import math
from abc import ABC, abstractmethod
from enum import Enum
from typing import NamedTuple, Dict, Callable, Tuple

from game import RoadColor, RoadMargin, RoadWidth, pos_info, DamagedColor, CanvasWidth, CanvasHeight
from math_utils import Position, Line, find_two_lines_intersection, Arc, get_arc_lines
from pygameGraphic import IGraphic


class IRoad(ABC):
    @abstractmethod
    def __init__(self, ix: int, iy: int, pos: Position):
        self.ix = ix
        self.iy = iy
        self.pos = pos

    @abstractmethod
    def render(self, ctx: IGraphic):
        pass

    @abstractmethod
    def collide(self, ctx: IGraphic, lines: list[Line]):
        pass

    @abstractmethod
    def render_damaged(self, ctx: IGraphic):
        pass

    @abstractmethod
    def get_bound_lines(self) -> list[Line]:
        pass


class EmptyRoad(IRoad):
    def __init__(self):
        self.ix = 0
        self.iy = 0
        self.pos = Position(x=0, y=0)

    def render(self, ctx):
        pass

    def collide(self, ctx: IGraphic, bound_lines: list[Line]):
        return []

    def render_damaged(self, ctx: IGraphic):
        pass

    def get_bound_lines(self):
        return []


class VerticalRoad(IRoad):
    ix = 0
    iy = 0
    pos = Position(0, 0)
    lineDamaged = ""

    def __init__(self):
        pass

    def render(self, ctx: IGraphic):
        x = self.pos.x
        y = self.pos.y
        color = RoadColor
        ctx.draw_line(Line(
            start=Position(x=x + RoadMargin, y=y),
            end=Position(x=x + RoadMargin, y=y + RoadWidth),
        ), color=color, thickness=7)
        ctx.draw_line(Line(
            start=Position(x=x + RoadWidth - RoadMargin, y=y),
            end=Position(x=x + RoadWidth - RoadMargin, y=y + RoadWidth)
        ), color=color, thickness=7)
        line1, line2 = self.get_bound_lines()
        # ctx.draw_text(Position(x=x + RoadMargin, y=y), f"{pos_info(line1.start)}", color=(0xf, 0xf, 0xf))

    def render_damaged(self, ctx: IGraphic):
        x = self.pos.x
        y = self.pos.y
        color = "red"
        if self.lineDamaged == "line1":
            ctx.draw_line(Position(x + RoadMargin, y), Position(x + RoadMargin, y + RoadWidth),
                          color=color, thickness=7)
        elif self.lineDamaged == "line2":
            ctx.draw_line(Position(x + RoadWidth - RoadMargin, y), Position(x + RoadWidth - RoadMargin, y + RoadWidth),
                          color=color, thickness=7)

    def get_bound_lines(self) -> list[Line]:
        x = self.ix * RoadWidth
        y = self.iy * RoadWidth
        line1 = Line(
            start=Position(
                x=x + RoadMargin,
                y=y
            ),
            end=Position(
                x=x + RoadMargin,
                y=y + RoadWidth
            )
        )

        line2 = Line(
            start=Position(
                x=x + RoadWidth - RoadMargin,
                y=y
            ),
            end=Position(
                x=x + RoadWidth - RoadMargin,
                y=y + RoadWidth,
            )
        )
        return [line1, line2]

    def collide(self, ctx, lines):
        line1, line2 = self.get_bound_lines()
        points = []
        for line in lines:
            point1 = find_two_lines_intersection(line1, line)
            if point1 is not None:
                self.lineDamaged = "line1"
                points.append(point1)
            point2 = find_two_lines_intersection(line2, line)
            if point2 is not None:
                self.lineDamaged = "line2"
                points.append(point2)
        self.lineDamaged = ""
        return points


class HorizontalRoad(IRoad):
    ix = 0
    iy = 0
    pos = Position(x=0, y=0)
    lineDamaged = ""

    def __init__(self):
        pass

    def render(self, ctx: IGraphic):
        x = self.pos.x
        y = self.pos.y
        ctx.draw_line(Line(
            start=Position(x, y + RoadMargin),
            end=Position(x + RoadWidth, y + RoadMargin)
        ), color=RoadColor, thickness=7)
        ctx.draw_line(Line(
            start=Position(x, y + RoadWidth - RoadMargin),
            end=Position(x + RoadWidth, y + RoadWidth - RoadMargin)
        ), color=RoadColor, thickness=7)

    def collide(self, ctx: IGraphic, lines: list[Line]):
        line1, line2 = self.get_bound_lines()
        points = []
        for line in lines:
            point1 = find_two_lines_intersection(line1, line)
            if point1 is not None:
                self.lineDamaged = "line1"
                points.append(point1)
            point2 = find_two_lines_intersection(line2, line)
            if point2 is not None:
                self.lineDamaged = "line2"
                points.append(point2)
        self.lineDamaged = ""
        return points

    def render_damaged(self, ctx: IGraphic):
        x = self.pos.x
        y = self.pos.y
        color = "red"
        if self.lineDamaged == "line1":
            ctx.draw_line(Line(
                start=Position(x, y + RoadMargin),
                end=Position(x + RoadWidth, y + RoadMargin)
            ), color=color, thickness=7)
        elif self.lineDamaged == "line2":
            ctx.draw_line(Line(
                start=Position(x, y + RoadWidth - RoadMargin),
                end=Position(x + RoadWidth, y + RoadWidth - RoadMargin)
            ), color=color, thickness=7)

    def get_bound_lines(self):
        x, y = self.pos
        line1 = Line(
            start=Position(x, y + RoadMargin),
            end=Position(x + RoadWidth, y + RoadMargin)
        )

        line2 = Line(
            start=Position(x, y + RoadWidth - RoadMargin),
            end=Position(x + RoadWidth, y + RoadWidth - RoadMargin)
        )
        return [line1, line2]


class CurveType(Enum):
    Empty = 0
    Outer = 1
    Inner = 2


CurveRadius = {
    CurveType.Outer: RoadWidth - RoadMargin,
    CurveType.Inner: RoadMargin
}


class CurveRoadType(Enum):
    LeftTop = 0
    RightTop = 1
    RightBottom = 2
    LeftBottom = 3


class CurveAngle(NamedTuple):
    start_angle: float
    end_angle: float


CurveAngles = {
    CurveRoadType.LeftTop: CurveAngle(start_angle=90, end_angle=180),
    CurveRoadType.RightTop: CurveAngle(0, 90),
    CurveRoadType.RightBottom: CurveAngle(270, 0),
    CurveRoadType.LeftBottom: CurveAngle(180, 270),
}


class CurveRoad:
    pos = Position(0, 0)
    type: CurveRoadType
    angles: CurveAngles

    def __init__(self, type: CurveRoadType):
        self.type = type
        self.angles = CurveAngles[type]

    def render(self, ctx: IGraphic, color: str):
        self.render_curve(ctx, CurveType.Outer, color)
        self.render_curve(ctx, CurveType.Inner, color)

    def render_curve(self, ctx: IGraphic, curve_type: CurveType, color: str):
        arc_xy = self.get_arc_xy()
        if curve_type == CurveType.Outer:
            start_angle, end_angle = self.angles
            ctx.draw_arc(Arc(
                centre=arc_xy,
                radius=CurveRadius[CurveType.Outer],
                start_angle=start_angle,
                end_angle=end_angle),
                color=color,
                thickness=7)

        if curve_type == CurveType.Inner:
            start_angle, end_angle = self.angles
            ctx.draw_arc(Arc(arc_xy,
                             radius=CurveRadius[CurveType.Inner],
                             start_angle=start_angle,
                             end_angle=end_angle),
                         color=color,
                         thickness=7)

    def get_bound_lines(self, curve_type: CurveType) -> list[Line]:
        arc_xy = self.get_arc_xy()
        radius = CurveRadius[curve_type]
        start_angle, end_angle = self.angles
        lines = get_arc_lines(Arc(arc_xy, radius, start_angle, end_angle))
        return lines

    def get_all_bound_lines(self) -> list[Line]:
        lines1 = self.get_bound_lines(CurveType.Outer)
        lines2 = self.get_bound_lines(CurveType.Inner)
        return lines1 + lines2

    def collide(self, lines: list[Line]) -> (CurveType, list[Position]):
        lines1 = self.get_bound_lines(CurveType.Outer)
        for line in lines1:
            # draw_line(ctx, line, { stroke_style: 'yellow' })
            for bound_line in lines:
                points1 = find_two_lines_intersection(bound_line, line)
                if points1 is not None:
                    return CurveType.Outer, [points1]

        lines2 = self.get_bound_lines(CurveType.Inner)
        for line in lines2:
            for bound_line in lines:
                points1 = find_two_lines_intersection(bound_line, line)
                if points1 is not None:
                    return CurveType.Inner, [points1]
        return CurveType.Empty, []

    def get_arc_xy(self) -> Position:
        if self.type == CurveRoadType.LeftTop:
            return Position(self.pos.x + RoadWidth, self.pos.y + RoadWidth)
        if self.type == CurveRoadType.RightTop:
            return Position(self.pos.x, self.pos.y + RoadWidth)
        if self.type == CurveRoadType.RightBottom:
            return Position(self.pos.x, self.pos.y)
        return Position(self.pos.x + RoadWidth, self.pos.y)


class LeftTopCurve(IRoad):
    def __init__(self):
        self.ix = 0
        self.iy = 0
        self.pos = Position(0, 0)
        self.line_damaged = CurveType.Empty
        self.curve = CurveRoad(CurveRoadType.LeftTop)

    def render(self, ctx: IGraphic):
        curve = self.curve
        curve.pos = self.pos
        curve.render(ctx, RoadColor)

    def collide(self, ctx: IGraphic, bound_lines: list[Line]) -> list[Position]:
        curve = self.curve
        curve.pos = self.get_bound_pos()
        curve_type, points = curve.collide(bound_lines)
        self.line_damaged = curve_type
        return points

    def render_damaged(self, ctx: IGraphic):
        curve = self.curve
        curve.pos = self.pos
        curve.render_curve(ctx, self.line_damaged, DamagedColor)

    def get_bound_lines(self) -> list[Line]:
        curve = self.curve
        curve.pos = self.get_bound_pos()
        return curve.get_all_bound_lines()

    def get_bound_pos(self) -> Position:
        return Position(self.ix * RoadWidth, self.iy * RoadWidth)


class RightTopCurve(IRoad):
    def __init__(self):
        self.ix = 0
        self.iy = 0
        self.pos = Position(0, 0)
        self.line_damaged = CurveType.Empty
        self.curve = CurveRoad(CurveRoadType.RightTop)

    def render(self, ctx: IGraphic):
        curve = self.curve
        curve.pos = self.pos
        curve.render(ctx, RoadColor)

    def render_damaged(self, ctx: IGraphic):
        curve = self.curve
        curve.pos = self.pos
        curve.render_curve(ctx, self.line_damaged, DamagedColor)

    def collide(self, ctx: IGraphic, bound_lines: list[Line]) -> list[Position]:
        curve = self.curve
        curve.pos = self.get_bound_pos()
        curve_type, points = curve.collide(bound_lines)
        self.line_damaged = curve_type
        return points

    def get_bound_lines(self) -> list[Line]:
        curve = self.curve
        curve.pos = self.get_bound_pos()
        return curve.get_all_bound_lines()

    def get_bound_pos(self) -> Position:
        x = self.ix * RoadWidth
        y = self.iy * RoadWidth
        return Position(x, y)


class RightBottomCurve(IRoad):
    def __init__(self):
        self.ix = 0
        self.iy = 0
        self.pos = Position(0, 0)
        self.line_damaged = CurveType.Empty
        self.curve = CurveRoad(CurveRoadType.RightBottom)

    def render(self, ctx: IGraphic):
        curve = self.curve
        curve.pos = self.pos
        curve.render(ctx, RoadColor)

    def render_damaged(self, ctx: IGraphic):
        curve = self.curve
        curve.pos = self.pos
        curve.render_curve(ctx, self.line_damaged, DamagedColor)

    def collide(self, ctx: IGraphic, bound_lines: list[Line]) -> list[Position]:
        curve = self.curve
        curve.pos = self.get_bound_pos()
        curve_type, points = curve.collide(bound_lines)
        self.line_damaged = curve_type
        return points

    def get_bound_lines(self) -> list[Line]:
        curve = self.curve
        curve.pos = self.get_bound_pos()
        return curve.get_all_bound_lines()

    def get_bound_pos(self) -> Position:
        x = self.ix * RoadWidth
        y = self.iy * RoadWidth
        return Position(x, y)


class LeftBottomCurve(IRoad):
    def __init__(self):
        self.ix = 0
        self.iy = 0
        self.pos = Position(0, 0)
        self.lineDamaged = CurveType.Empty
        self.curve = CurveRoad(CurveRoadType.LeftBottom)

    def render(self, ctx: IGraphic):
        curve_road = self.curve
        curve_road.pos = self.pos
        curve_road.render(ctx, RoadColor)

    def render_damaged(self, ctx: IGraphic):
        curve_road = self.curve
        curve_road.pos = self.pos
        if self.lineDamaged == CurveType.Outer:
            curve_road.render_curve(ctx, CurveType.Outer, DamagedColor)
        if self.lineDamaged == CurveType.Inner:
            curve_road.render_curve(ctx, CurveType.Inner, DamagedColor)

    def collide(self, ctx: IGraphic, bound_lines: list[Line]) -> list[Position]:
        curve = self.curve
        curve.pos = self.get_bound_pos()
        curve_type, points = curve.collide(bound_lines)
        self.lineDamaged = curve_type
        return points

    def get_bound_lines(self) -> list[Line]:
        self.curve.pos = self.get_bound_pos()
        return self.curve.get_all_bound_lines()

    def get_bound_pos(self) -> Position:
        x = self.ix * RoadWidth
        y = self.iy * RoadWidth
        return Position(x, y)


def create2d_array(rows: int, cols: int) -> list[list[IRoad]]:
    return [[EmptyRoad()] * cols for _ in range(rows)]


def create_road(ch: str) -> IRoad:
    dict: Dict[str, Callable[[], IRoad]] = {
        '-': lambda: HorizontalRoad(),
        '|': lambda: VerticalRoad(),
        '/': lambda: LeftTopCurve(),
        '\\': lambda: RightTopCurve(),
        'L': lambda: LeftBottomCurve(),
        '+': lambda: RightBottomCurve(),
    }

    if ch not in dict:
        return EmptyRoad()

    return dict[ch]()


def read_map(map_content: str) -> list[list[IRoad]]:
    lines = map_content.split('\n')
    width = max(len(line) for line in lines)
    height = len(lines)
    road_map = create2d_array(width, height)

    for y in range(height):
        line = lines[y]
        for x in range(width):
            ch = line[x]
            road = create_road(ch)
            road.ix = x
            road.iy = y
            road_map[x][y] = road

    return road_map


def read_map_file(file: str):
    with open(file, 'r') as file:
        content = file.read()
    return read_map(content)


class RoadMap:
    roads: list[list[IRoad]]

    def __init__(self):
        self.pos = Position(x=0, y=0)
        self.roads = create2d_array(10, 10)
        self.roads = read_map_file("./assets/map.txt")

    def render(self, ctx: IGraphic):
        x = self.pos.x + CanvasWidth // 2
        y = self.pos.y + CanvasHeight // 2
        roads = self.roads
        for ix in range(len(roads)):
            for iy in range(len(roads[ix])):
                road = roads[ix][iy]
                if road is None:
                    continue
                road.pos = Position(x + ix * RoadWidth, y + iy * RoadWidth)
                road.render(ctx)

    def collide(self, ctx: IGraphic, bound_lines: list[Line]) -> Tuple[IRoad, list[Position]]:
        roads = self.roads
        for ix in range(len(roads)):
            for iy in range(len(roads[ix])):
                road = roads[ix][iy]
                collide_points = road.collide(ctx, bound_lines)
                if len(collide_points) > 0:
                    return road, collide_points
        return EmptyRoad(), []
