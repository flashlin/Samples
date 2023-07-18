from math_utils import Position

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
CarPos = Position(CenterX, CenterY)
StartX = 100
StartY = 400
DamagedColor = "red"
RadarLineLength = 150
RadarLineCount = 5
RadarCount = 3
RadarColor = 'gray'
UseBrain = True


def pos_info(pos: Position) -> str:
    x = round(pos.x)
    y = round(pos.y)
    return f"{x},{y}"

# 垂直線


# def get_left_top_curve_centre(pos: Position):
#     return Position(pos.x + RoadWidth, pos.y + RoadWidth)
#
# def get_outer_curve_radius():
#     return RoadWidth - RoadMargin
#
# def get_left_top_curve_angles():
#     return CurveAngles(start_angle=90, end_angle=180)
#
# def get_left_top_curve_lines(pos: Position) -> list[Line]:
#     arc_xy = get_left_top_curve_centre(pos)
#     radius = get_outer_curve_radius()
#     angles = get_left_top_curve_angles()
#     lines = get_arc_lines(Arc(arc_xy, radius, angles.start_angle, angles.end_angle))
#     return lines



