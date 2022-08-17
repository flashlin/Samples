import os
import uuid
import numpy as np
import time
from mss import mss
import cv2
from pynput.keyboard import Key, Listener

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
      self.key_listener = Listener(on_press=self.on_keypress, on_release=self.on_keyrelease)
      self.key_listener.start()

   def collect_frames(self):
      filename = os.path.join('data', str(uuid.uuid1()))
      gamecap = np.array(self.capture.grab(self.game_area))
      cv2.imwrite(f'{filename}.png', gamecap)

   def on_keypress(self, key):
      if key == Key.esc:
         print('Exiting')
         self.is_exit = True
         return
      self.current_keys = key
      print(self.current_keys)

   def on_keyrelease(self, key):
      self.current_keys = None
      if key == Key.esc:
         return False
      return True

if __name__ == '__main__':
   game = GamePlay()
   while not game.is_exit:
      time.sleep(1)
      game.collect_frames()
