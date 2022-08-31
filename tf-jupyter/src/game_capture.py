import os
import uuid
import numpy as np
import time
from mss import mss
import cv2
from pynput.keyboard import Key, Listener
from pynput import mouse
import functools

def synchronized(lock):
    """ Synchronization decorator """
    def wrap(f):
        @functools.wraps(f)
        def newFunction(*args, **kw):
            with lock:
                return f(*args, **kw)
        return newFunction
    return wrap


import threading
lock = threading.Lock()



class GamePlay():
   def __init__(self):
      self.game_area = {
         "left": 0,
         "top": 0,
         "width": 500, #800,
         "height": 800, #600
      }
      self.is_exit = False
      self.capture = mss()
      self.current_keys = None
      self.key_listener = Listener(on_press=self.on_keypress, on_release=self.on_keyrelease)
      self.key_listener.start()
      self.current_mouse_click = None
      mouse_listener = mouse.Listener(on_click=self.on_mouse_click)
      mouse_listener.start()

   def collect_frame(self, filename):
      gamecap = np.array(self.capture.grab(self.game_area))
      cv2.imwrite(f'{filename}.png', gamecap)

   def collect_key_event(self, filename):
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
      self.collect_frame(filename)
      self.collect_key_event(filename)

   def collect_loop1(self):
      while not self.is_exit:
         time.sleep(1000 / 30)

   def collect_loop(self):
      while not self.is_exit:
         time.sleep(1 / 25)
         self.collect_frame_without_key()

   #@synchronized(lock)
   def collect_frame_without_key(self):
      filename = os.path.join('data', str(uuid.uuid1()))
      self.collect_frame(filename)
      self.write_key_event(filename)

   #@synchronized(lock)
   def only_collect_key(self, current_mouse_click):
      filename = os.path.join('data', str(uuid.uuid1()))
      self.collect_frame(filename)
      mouse_x, mouse_y = current_mouse_click
      data = [ 'None', f'{mouse_x}, {mouse_y}' ]
      self.write_key_event(filename, data)

   def write_key_event(self, filename, data=[ 'None', '-1, -1' ]):
      np.savetxt(f'{filename}.txt', data, fmt='%s', delimiter=',')

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
            self.only_collect_key(self.current_mouse_click)
            print(f'mouse {x}, {y}')
      else:
         self.current_mouse_click = None
      #self.collect()

if __name__ == '__main__':
   game = GamePlay()
   game.collect_loop()
