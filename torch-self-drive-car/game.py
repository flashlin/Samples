from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Callable
from pygameGraphic import IGraphic
from math_utils import Position, Line, find_two_lines_intersection

CarFrameMargin = 5
CarWidth = 75
CarHeight = 117
FrameWidth = CarWidth - CarFrameMargin * 2
FrameHeight = CarHeight - CarFrameMargin * 2
CarColor = "blue"
RoadWidth = 220
RoadColor = 'blue'
RoadMargin = 22
CanvasWidth = 800
CanvasHeight = 700
CenterX = CanvasWidth // 2 - CarWidth // 2
CenterY = CanvasHeight // 2 - CarHeight // 2
CarPos = {"x": CenterX, "y": CenterY}
StartX = 100
StartY = 400
DamagedColor = "red"
RadarLineLength = 150
RadarLineCount = 5
RadarCount = 3
RadarColor = 'gray'
UseBrain = True


def posInfo(pos: Position) -> str:
    x = round(pos.x)
    y = round(pos.y)
    return f"{x},{y}"


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
    def collide(self, ctx: IGraphic, lines: List[Line]):
        pass

    @abstractmethod
    def render_damaged(self, ctx: IGraphic):
        pass

    @abstractmethod
    def get_bound_lines(self) -> List[Line]:
        pass


class EmptyRoad(IRoad):
    def __init__(self):
        self.ix = 0
        self.iy = 0
        self.pos = Position(x=0, y=0)

    def render(self, ctx):
        pass

    def collide(self, ctx: IGraphic, bound_lines: List[Line]):
        return []

    def render_damaged(self, ctx: IGraphic):
        pass

    def get_bound_lines(self):
        return []


# 垂直線
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
        ctx.draw_text(Position(x=x + RoadMargin, y=y), f"{posInfo(line1.start)}", color=(0xf, 0xf, 0xf))

    def render_damaged(self, ctx):
        x = self.pos.x
        y = self.pos.y
        ctx.beginPath()
        if self.lineDamaged == "line1":
            ctx.moveTo(x + RoadMargin, y)
            ctx.lineTo(x + RoadMargin, y + RoadWidth)
        elif self.lineDamaged == "line2":
            ctx.moveTo(x + RoadWidth - RoadMargin, y)
            ctx.lineTo(x + RoadWidth - RoadMargin, y + RoadWidth)
        ctx.lineWidth = 7
        ctx.strokeStyle = "red"
        ctx.stroke()

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

    def collide(self, ctx: IGraphic, lines: List[Line]):
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


def create2dArray(rows: int, cols: int) -> List[List[Optional[IRoad]]]:
    return [[EmptyRoad()] * cols for _ in range(rows)]


def createRoad(ch: str) -> IRoad:
    dict: Dict[str, Callable[[], IRoad]] = {
        '-': lambda: HorizontalRoad(),
        '|': lambda: VerticalRoad(),
        # '/': lambda: LeftTopCurve(),
        # '\\': lambda: RightTopCurve(),
        # 'L': lambda: LeftBottomCurve(),
        # '+': lambda: RightBottomCurve(),
    }

    if ch not in dict:
        return EmptyRoad()

    return dict[ch]()


def read_map(map_content: str) -> List[List[Optional[IRoad]]]:
    lines = map_content.split('\n')
    width = max(len(line) for line in lines)
    height = len(lines)
    road_map = create2dArray(width, height)

    for y in range(height):
        line = lines[y]
        for x in range(width):
            ch = line[x]
            road = createRoad(ch)
            road.ix = x
            road.iy = y
            road_map[x][y] = road

    return road_map


def read_map_file(file: str):
    with open(file, 'r') as file:
        content = file.read()
    return read_map(content)


class RoadMap:
    roads: List[List[Optional[IRoad]]]

    def __init__(self):
        self.pos = Position(x=0, y=0)
        self.roads = create2dArray(10, 10)
        self.roads = read_map_file("./assets/map.txt")

    def render(self, ctx: IGraphic):
        x = self.pos.x
        y = self.pos.y
        roads = self.roads
        for ix in range(len(roads)):
            for iy in range(len(roads[ix])):
                road = roads[ix][iy]
                if road is None:
                    continue
                road.pos = Position(
                    x=x + ix * RoadWidth,
                    y=y + iy * RoadWidth,
                )
                road.render(ctx)

    def collide(self, ctx: IGraphic, bound_lines: List[Line]) -> Tuple[IRoad, List[Position]]:
        roads = self.roads
        for ix in range(len(roads)):
            for iy in range(len(roads[ix])):
                road = roads[ix][iy]
                collide_points = road.collide(ctx, bound_lines)
                if len(collide_points) > 0:
                    return road, collide_points
        return EmptyRoad(), []
