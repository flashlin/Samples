# pip install -r requirements.txt 
import PySimpleGUI as sg
import win32gui
from threading import Timer
import datetime
from PIL import ImageGrab
import numpy as np
import cv2
from pynput import keyboard
import time
import threading

window_width = 1280
window_height = 720
period = 0.05
font_size = 16
font = 'simsun '+str(font_size)+' bold'
is_moving = True
is_start_recording = False
captured_video = None


def on_press(key):
   global is_start_recording
   print(f'{key} pressed')
   if key == keyboard.Key.esc:
      is_start_recording = False
      return False
   return True

def hot_key():
   with keyboard.Listener(on_press=on_press) as listener:
      listener.join()

def start_hot_key():
   thread = Timer(period, hot_key)
   thread.setDaemon(True) # 避免 main thread is not in main loop
   thread.start()


def create_video_stream():
   global captured_video
   timeStamp = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
   recordName = f'ScreenRecording-{timeStamp}.mp4'
   fourChar = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
   #fourChar = cv2.VideoWriter_fourcc('a', 'v', 'c', '1')
   fps = 10.0
   captured_video = cv2.VideoWriter(recordName, fourChar, fps, (window_width, window_height))
   #captured_video.write(frame)
   print(f"Start recording '{recordName}'")

def start_recording():
   global is_start_recording
   if is_start_recording:
      img = ImageGrab.grab(bbox=(0, 0, window_width, window_height))
      img_np = np.array(img)
      img_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
      #cv2.imshow("capture", img_img)
      captured_video.write(img_img)
   else:
      captured_video.release()
      cv2.destroyAllWindows()
      #if cv2.waitKey(1) & 0xFF == ord('q'):
      #   return
   thread = Timer(period, start_recording)
   thread.setDaemon(True) # 避免 main thread is not in main loop
   thread.start()

def take_recording():
   global is_start_recording
   while is_start_recording:
      img = ImageGrab.grab(bbox=(0, 0, window_width, window_height))
      img_np = np.array(img)
      img_img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
      #cv2.imshow("capture", img_img)
      captured_video.write(img_img)
      time.sleep(100)
   captured_video.release()

def start_take_recording():
   thread = threading.Thread(target=take_recording, args=())
   thread.start()


def text(string, key=None, size=(13,1)):
   return sg.Text(text=string, size=size, font=font, key=key,
                   text_color='black', background_color='white',
                   border_width=0, justification='center', pad=(0, 5),
                   auto_size_text=False)

def frame(frm, title):
   return sg.Frame(layout=frm, title=title, pad=(0,5), font=font,
                  element_justification='left', 
                  background_color='green',
                  border_width=2)

def draw_text(widget, text):
   item = widget.draw_text(text, (0, 0), font=font, color='green')
   #self.numbers1.append(item)


frame1 = [
   [text('pos', size=(4,1)), text('x,y',       key='position')]
]

layout = [
   [sg.Canvas(size=(window_width, window_height), key='canvas', pad=(0, 0))],
   #[frame(frame1, '滑鼠 - ←↑↓→')],
   # [sg.Graph(canvas_size=(400, 400), graph_bottom_left=(0, 0), graph_top_right=(400, 400),
   #    key="-GRAPH-", 
   #    change_submits=True,  # mouse click events
   #    background_color='lightblue',
   #    drag_submits=True), ],
   [sg.T('Change circle color to:'), sg.Button('Start'), sg.Button('Stop')],
   ]

window = sg.Window('ScreenRecoder', 
   layout,
   size=(window_width, window_height),
   keep_on_top=True,
   auto_size_buttons=False,
   no_titlebar=True,
   grab_anywhere=False,
   return_keyboard_events=False,
   alpha_channel=0.7,
   use_default_focus=False,
   transparent_color='red',
   margins=(0, 0),
   finalize=True)      
#window.bind("<Control-KeyPress-c>", "CTRL-C")
#window.bind("<Control-KeyPress-C>", "CTRL-C")               
window.bind("<Control-KeyPress-F11>", "CTRL-F11")
window.bind("<Control-KeyPress-F12>", "CTRL-F12")
window.bind("<Button-1>", "MouseLeft")
window.bind("<Button-2>", "MouseMiddle")
window.Finalize()      
#window.maximize()


def captureMouse():
   # Get cursor position, pixel color, and grab image on screen
   global mouseX, mouseY
   flags, hcursor, (x, y)  = win32gui.GetCursorInfo()  # Get cursor position
   mouseX, mouseY = x ,y
   #  w = int(width/scale)
   #  h = int(height/scale)
   #  im=ImageGrab.grab(bbox=(x-w, y-h, x+w+1, y+h+1))    # Grab image on screen
   #  color = im.getpixel((w, h))                         # Get pixel color
   #  if scale!=1:
   #      im=im.resize(graph_size, resample=Image.NEAREST)# resize
   #  im = ImageTk.PhotoImage(im)
   #  return im, x, y, color
   left = x - window_width // 2
   top = y - window_height // 2
   #window['position'].update(f'{left},{top}')
   if is_moving:
      window.move(left, top)
   thread = Timer(period, captureMouse)
   thread.setDaemon(True) # 避免 main thread is not in main loop
   thread.start()

captureMouse()

canvas = window['canvas']
#cir = canvas.TKCanvas.create_oval(50, 50, 100, 100)      
#x = 0
#y = 0
#width = 1280
#height = 720
rect = canvas.TKCanvas.create_rectangle(0, 0, window_width, window_height, outline='green', width=3)

while True:      
   event, values = window.read()
   print(f"event: {event}")
   if event == sg.WINDOW_CLOSED:
      break
   if event is None:      
      break
   if event == "MouseLeft":
      is_moving = False
   if event == "Escape:27":
      break
   if event == "CTRL-F12":
      if is_start_recording:
         is_start_recording = False
         break
      else:
         is_start_recording = True
         create_video_stream()
         start_recording()
         #start_take_recording()
      is_moving = False
   #if event == "CTRL-C":
   #if event == 'F12:123':
    #if event == 'Blue':      
        #canvas.TKCanvas.itemconfig(cir, fill="Blue")      
    #elif event == 'Red': 
    #    canvas.TKCanvas.itemconfig(cir, fill="Red")
window.close()

