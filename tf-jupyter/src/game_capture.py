import os
import uuid
import numpy as np
import time
from mss import mss
import cv2
from pynput.keyboard import Key, Listener
from pynput import mouse

class GamePlay():
   def __init__(self):
      self.game_area = {
         "left": 0,
         "top": 0,
         "width": 800,
         "height": 600
      }
      self.is_exit = False
      self.capture = mss()
      self.current_keys = None
      self.key_listener = Listener(on_press=self.on_keypress, on_release=self.on_keyrelease)
      self.key_listener.start()
      self.current_mouse_click = None
      mouse_listener = mouse.Listener(on_click=self.on_mouse_click)
      mouse_listener.start()

   def collect_frames(self, filename):
      gamecap = np.array(self.capture.grab(self.game_area))
      cv2.imwrite(f'{filename}.png', gamecap)

   def collect_key_events(self, filename):
      mouse_x, mouse_y = -1, -1
      if self.current_mouse_click is not None:
         mouse_x, mouse_y = self.current_mouse_click
      key = 'None'
      if self.current_keys is not None:
         key = self.current_keys
      data = [ key, f'{mouse_x}, {mouse_y}' ]
      np.savetxt(f'{filename}.txt', data, fmt='%s', delimiter=',')

   def collect(self):
      filename = os.path.join('data', str(uuid.uuid1()))
      self.collect_frames(filename)
      self.collect_key_events(filename)

   def collect_loop(self):
      while not self.is_exit:
         time.sleep(1)

   def on_keypress(self, key):
      if key == Key.esc:
         print('Exiting')
         self.is_exit = True
         return
      self.current_keys = key
      print(self.current_keys)
      self.collect()

   def on_keyrelease(self, key):
      self.current_keys = None
      self.collect()
      if key == Key.esc:
         return False
      return True

   def on_mouse_click(self, x, y, button, pressed):
      if pressed:
         if button == mouse.Button.left:
            self.current_mouse_click = (x, y)
            print(f'mouse {x}, {y}')
      else:
         self.current_mouse_click = None
      self.collect()

if __name__ == '__main__':
   game = GamePlay()
   game.collect_loop()
