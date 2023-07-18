from abc import ABC, abstractmethod
import pygame
from math_utils import Position, Line, Arc


class IGraphic(ABC):
    @abstractmethod
    def create(self, screen_size: (int, int)):
        pass

    @abstractmethod
    def render_start(self):
        pass

    @abstractmethod
    def render_end(self):
        pass

    @abstractmethod
    def quit(self):
        pass

    @abstractmethod
    def draw_text(self, pos: Position, text: str, color: (int, int, int)):
        pass

    @abstractmethod
    def draw_line(self, line: Line, color: (int, int, int), thickness: int):
        pass

    @abstractmethod
    def draw_arc(self, arc: Arc, color: (int, int, int), thickness: int):
        pass

    @abstractmethod
    def draw_image(self, image_asset_name: str, pos: Position, angle: int):
        pass


class PygameGraphic(IGraphic):
    cache = {}

    def __init__(self):
        self.font = None
        self.screen = None
        self.screen_size = (800, 600)

    def create(self, screen_size: (int, int)):
        self.screen_size = screen_size
        pygame.init()
        pygame.display.set_caption("Game")
        self.screen = pygame.display.set_mode(self.screen_size)
        self.font = pygame.font.Font(None, 36)

    def render_start(self):
        self.screen.fill((0, 0, 0))

    def render_end(self):
        pygame.display.flip()

    def quit(self):
        pygame.quit()

    def draw_text(self, pos: Position, text: str, color: (int, int, int)):
        text_obj = self.font.render(text, True, color)
        self.screen.blit(text_obj, (pos.x, pos.x))

    def draw_line(self, line: Line, color: (int, int, int), thickness: int):
        start = (line.start.x, line.start.y)
        end = (line.end.x, line.end.y)
        pygame.draw.line(self.screen, color, start, end, thickness)

    def draw_arc(self, arc: Arc, color: (int, int, int), thickness: int):
        cx = arc.centre.x
        cy = arc.centre.y
        radius = arc.radius
        pygame.draw.arc(self.screen, color, (cx - radius, cy - radius, radius * 2, radius * 2),
                        arc.start_angle, arc.end_angle, thickness)

    def draw_image(self, image_asset_name: str, pos: Position, angle: int):
        image = self.fetch_data(f"image_{image_asset_name}", lambda: pygame.image.load(image_asset_name))
        rotated_image = pygame.transform.rotate(image, angle)
        rotated_rect = rotated_image.get_rect()
        offset_x = pos.x - rotated_rect.width / 2
        offset_y = pos.y - rotated_rect.height / 2
        self.screen.blit(rotated_image, (offset_x, offset_y))

    def fetch_data(self, key: str, fetch: callable):
        if key in self.cache:
            return self.cache[key]
        result = fetch()
        self.cache[key] = result
        return result
