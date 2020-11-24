import Video_Capture
import Main_Algorithm
from tkinter import *

Video_Capture.Start_Video(Main_Algorithm.main_path)
print(Main_Algorithm.Recognise(Main_Algorithm.main_path))

'''
Main_window = Tk()
Video_Frame = Canvas(Main_window,width = 200,height = 200)
Video_Frame.draw()
Main_window.mainloop()
'''




