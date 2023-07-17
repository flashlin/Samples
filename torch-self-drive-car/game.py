from abc import ABC, abstractmethod
from typing import List
from pygameGraphic import Position, Line, IGraphic

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
    def renderDamaged(self, ctx: IGraphic):
        pass

    @abstractmethod
    def getBoundLines(self) -> List[Line]:
        pass


