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
         "width": 440, #800,
         "height": 830, #600
      }
      self.key_pressed = {
         Key.left: 0,
         Key.right: 0,
         Key.up: 0,
         Key.down: 0
      }
      #
      self.start_mouse_x = 36
      self.start_mouse_y = 94
      self.cell_width = 20
      self.cell_height = 20
      self.cell_width_len = 18
      self.cell_height_len = 31 + 4
      self.cell_total_len = (self.cell_width_len * self.cell_height_len)
      #
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
         time.sleep(1 / 20)
         #self.collect_frame_without_key()

   #@synchronized(lock)
   def collect_frame_without_key(self):
      filename = os.path.join('data', str(uuid.uuid1()))
      self.collect_frame(filename)
      self.write_key_event(filename)

   #@synchronized(lock)
   def only_collect_key(self, current_mouse_click):
      filename = os.path.join('data', str(uuid.uuid1()))
      mouse_x, mouse_y = current_mouse_click
      # 18 x 31 方格, mouse 36,94 開始, 49,106  一格方塊大約 4x4
      start_mouse_x = self.start_mouse_x
      start_mouse_y = self.start_mouse_y
      cell_width = self.cell_width
      cell_height = self.cell_height
      cell_width_len = self.cell_width_len
      cell_height_len = self.cell_height_len
      cell_total_len = self.cell_total_len
      if mouse_x < start_mouse_x:
         return
      if mouse_y < start_mouse_y:
         return
      if mouse_x > start_mouse_x + cell_width * cell_width_len:
         return
      if mouse_y > start_mouse_y + cell_height * cell_height_len:
         return
      x = (mouse_x - start_mouse_x) // cell_width
      y = (mouse_y - start_mouse_y) // cell_height
      index = x + y * cell_width_len
      if index >= cell_total_len:
         return
      self.collect_frame(filename)
      print(f'{mouse_x},{mouse_y} idx:{index} [{x},{y}]')
      #data = [ 'None', f'{mouse_x}, {mouse_y}' ]
      #self.write_key_event(filename, data)
      mouse_data = [0] * cell_total_len
      mouse_data[index] = 1
      str_mouse_data = ','.join(str(x) for x in mouse_data)
      #print(f'str={str_mouse_data}')
      data = [ f'{index}', str_mouse_data ]
      self.write_key_event(filename, data)

   def write_key_event(self, filename, data=[ 'None', '-1, -1' ]):
      np.savetxt(f'{filename}.txt', data, fmt='%s', delimiter=',')

   def on_keypress(self, key):
      self.key_pressed[key] = 1
      if key == Key.esc:
         print('Exiting')
         self.is_exit = True
         return
      self.current_keys = key
      #print(self.current_keys)
      # print(self.key_pressed)
      # self.collect()

   def on_keyrelease(self, key):
      self.key_pressed[key] = 0
      self.current_keys = None
      # self.collect()
      if key == Key.esc:
         return False
      return True

   def on_mouse_click(self, x, y, button, pressed):
      if pressed:
         if button == mouse.Button.left:
            self.current_mouse_click = (x, y)
            print(f'mouse {x}, {y}')
            self.only_collect_key(self.current_mouse_click)
      else:
         self.current_mouse_click = None
      #self.collect()

if __name__ == '__main__':
   game = GamePlay()
   game.collect_loop()
