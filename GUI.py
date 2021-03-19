import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
from PIL import Image,ImageTk
import subprocess
import cv2
import shutil
import tkinter.messagebox as tkm
from demoYUCmd import runDemo
import tkinter.font as tkFont

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.flag = 0
        self.create_widgets()
        self.tmpVidInPath = '/tmp/vidIn.mp4'
        self.tmpImgInPath = '/tmp/imgIn'

    def start(self):
        self.flag = 1
        self.video_get_In()

    def stop(self):
        self.flag = 0 if self.flag == 1 else 1
        self.video_loop_In()


    def start_res(self):
        self.flag = 1
        self.video_get_Res()

    def stop_res(self):
        self.flag = 0 if self.flag == 1 else 1
        self.video_loop_Res()

    def create_widgets(self):

        self.fileModel = tk.StringVar()
        self.getModel_bt = tk.Button(self.master, width=15, height=1,
                                      text="Open your model file", command=self._getModel)
        self.getModel_bt.grid(row=0, column=0, sticky=tk.W)

        self.modelPathText = tk.Label(self.master, width=60, height=1, text='model path', anchor=tk.NW)
        self.modelPathText.grid(row=1, column=0, sticky=tk.W)

        self.outputPath = tk.StringVar()
        self.getOutput_bt = tk.Button(self.master, width=15, height=1, text="Open your output file",
                                      state='disable', command=self._getOutFile)
        self.getOutput_bt.grid(row=2, column=0, sticky=tk.W)

        self.outPathText = tk.Label(self.master, width=50, height=1, text='Output path', anchor=tk.NW)
        self.outPathText.grid(row=3, column=0, sticky=tk.W)


        self.fileImgPath = tk.StringVar()
        self.getImg_bt = tk.Button(self.master, width=15, height=1, text="Open your image path", command=self._getFile, state='disable')
        self.getImg_bt.grid(row=4, column=0, sticky=tk.W)

        self.imgPathText = tk.Label(self.master, width = 50, height = 1, text='image path', anchor=tk.NW)
        self.imgPathText.grid(row=5, column=0, sticky=tk.W)


        self.movieLabel = tk.Label(self.master,text='Input video')
        self.movieLabel.grid(row=6, column=0, sticky=tk.W)

        self.start_button = tk.Button(self.master, text='Restart', width=20, command=self.start, state = 'disable')
        self.start_button.grid(row=7, column=0, sticky=tk.W)

        self.stop_button = tk.Button(self.master, text='Pause/Resume', width=20, command=self.stop, state = 'disable')
        self.stop_button.grid(row=8, column=0, sticky=tk.W)

        self.run_button = tk.Button(self.master, text='Run model on input', width=20, command=self.runOnInput, state = 'disable')
        self.run_button.grid(row=4, column=1, sticky=tk.W)

        self.movie3DLabel = tk.Label(self.master, text='Output video')
        self.movie3DLabel.grid(row=6, column=1, sticky=tk.W)

        self.start_button_res = tk.Button(self.master, text='Restart', width=20, command=self.start_res,
                                      state='disable')
        self.start_button_res.grid(row=7, column=1, sticky=tk.W)

        self.stop_button_res = tk.Button(self.master, text='Pause/Resume', width=20, command=self.stop_res,
                                     state='disable')
        self.stop_button_res.grid(row=8, column=1, sticky=tk.W)

        f1 = tkFont.Font(family='times', size=20, weight='bold')
        self.statusText = tk.Label(self.master, width=60, height=1, text='Step1: Select pretrained model', anchor=tk.NW, font=f1)
        self.statusText.grid(row=0, column=1, sticky=tk.W, rowspan=2)

    def runOnInput(self):
        self.statusText['text'] = 'Running! Please wait for the result!'
        self.statusText.update()
        runDemo(self.fileImgPath, self.outputPath, self.fileModel)
        self.outputVid = os.path.join(self.outputPath, 'vibe_result.mp4')
        self.video_get_Res()
        self.start_button_res['state'] = 'normal'
        self.stop_button_res['state'] = 'normal'
        self.statusText['text'] = 'Done!'
        self.statusText.update()


    def _getOutFile(self):
        default_dir = r""
        self.outputPath = tk.filedialog.askdirectory(title=u'select your file', initialdir=(os.path.expanduser(default_dir)))
        self.outPathText['text'] = self.outputPath
        self.outPathText.update()
        self.getImg_bt['state'] = 'normal'
        self.getImg_bt.update()
        self.statusText['text'] = 'Step3: Select image path'
        self.statusText.update()

    def _getFile(self):
        default_dir = r""
        self.fileImgPath = tk.filedialog.askdirectory(title=u'select your file', initialdir=(os.path.expanduser(default_dir)))
        self.imgPathText['text'] = self.fileImgPath
        self.imgPathText.update()
        # tkmBox = tkm.showwarning('Warning', 'Processing images for visualization! \nPlease wait!')
        if os.path.exists(self.tmpImgInPath):
            shutil.rmtree(self.tmpImgInPath)
        if os.path.exists(self.tmpVidInPath):
            os.remove(self.tmpVidInPath)
        os.mkdir(self.tmpImgInPath)
        self.statusText['text'] = 'Preprocessing, please wait!'
        self.statusText.update()
        for ind, file in enumerate(os.listdir(self.fileImgPath)):
            img = cv2.imread(os.path.join(self.fileImgPath, file))
            tgtfile = os.path.join(self.tmpImgInPath, str(ind).zfill(6) + '.png')
            cv2.imwrite(tgtfile, img)
        self._genVid()

        self.start_button['state'] = 'normal'
        self.start_button.update()
        self.stop_button['state'] = 'normal'
        self.stop_button.update()

        self.run_button['state'] = 'normal'
        self.run_button.update()

        self.video_get_In()
        self.statusText['text'] = 'Step4: Run the model!'
        self.statusText.update()


    def _getModel(self):
        default_dir = r""
        self.fileModel = tk.filedialog.askopenfilename(title=u'select your file',initialdir=(os.path.expanduser(default_dir)))
        self.modelPathText['text'] = self.fileModel
        self.modelPathText.update()
        self.getOutput_bt['state'] = 'normal'
        self.getOutput_bt.update()
        self.statusText['text'] = 'Step2: Select output path'
        self.statusText.update()


    def _genVid(self):
        command = [
            'ffmpeg', '-y', '-threads', '16', '-i', f'{self.tmpImgInPath}/%06d.png', '-profile:v', 'baseline',
            '-level', '3.0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-an', '-v', 'error', self.tmpVidInPath,
        ]
        print(f'Running \"{" ".join(command)}\"')
        subprocess.call(command)
        print('Done')

    def video_get_In(self):
        self.capIn = cv2.VideoCapture(self.tmpVidInPath)  # 获取视频
        self.wait_time = 500 / self.capIn.get(5)  # 视频频率
        self.video_loop_In()

    def video_loop_In(self):
        ret, frame = self.capIn.read()  # 读取照片
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            current_image = Image.fromarray(img).resize((480, 270)) # 将图像转换成Image对象
            imgtk = ImageTk.PhotoImage(image=current_image)
            self.movieLabel.imgtk = imgtk
            self.movieLabel.config(image=imgtk)
            self.movieLabel.update()
            if self.flag == 1:
                self.master.after(int(self.wait_time), lambda: self.video_loop_In())

    def video_get_Res(self):
        self.capRes = cv2.VideoCapture(self.outputVid)  # 获取视频
        self.wait_time = 500 / self.capRes.get(5)  # 视频频率
        self.video_loop_Res()

    def video_loop_Res(self):
        ret, frame = self.capRes.read()  # 读取照片
        if ret:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            current_image = Image.fromarray(img).resize((480, 270)) # 将图像转换成Image对象
            imgtk = ImageTk.PhotoImage(image=current_image)
            self.movie3DLabel.imgtk = imgtk
            self.movie3DLabel.config(image=imgtk)
            self.movie3DLabel.update()
            if self.flag == 1:
                self.master.after(int(self.wait_time), lambda: self.video_loop_Res())

root = tk.Tk()
root.title("End-to-end 3D human shape")
root.geometry("1000x600+50+50")
root.grid_columnconfigure(0, minsize=500)  # Here


app = Application(master=root)
app.mainloop()
