# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import torch
import numpy as np
import cupy as cp
import itertools
import os
from scipy.fftpack import fft
from scipy import signal
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# %matplotlib inline
from scipy.interpolate import make_interp_spline, BSpline
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.figure import Figure
from matplotlib import animation
import matplotlib.animation as anim
from matplotlib.animation import TimedAnimation
from matplotlib.lines import Line2D
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import time
import timeit
import threading
from queue import Queue
from datetime import datetime
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
if torch.cuda.is_available() :
    print('GPU found')
else:
    print("No GPU found")


class CustomMainWindow(QMainWindow): # about the main window
    def __init__(self):
        super(CustomMainWindow, self).__init__()
        # Define the geometry of the main window
        self.setGeometry(100, 100, 1300, 900)
        self.setWindowTitle("BSN Graph")
        
        self.samt = 3.0
        self.samn = 100
        self.graph_num = -1
        global timer # init timer
        timer = QTimer(self)
        
        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        
        # Create FRAME_B
        self.FRAME_B = QFrame(self)
        self.LAYOUT_B = QGridLayout()
        self.FRAME_B.setLayout(self.LAYOUT_B)
        self.FRAME_B.setFixedWidth(320)
        
         #Create and Set Layout
        self.hboxlayout = QHBoxLayout()
        self.hboxlayout.addWidget(self.FRAME_B)
        self.hboxlayout.addWidget(self.FRAME_A)
        self.widget = QWidget()
        self.widget.setLayout(self.hboxlayout)
        self.setCentralWidget(self.widget)
        
        # name & version
        self.label = QLabel(self)
        self.label.setText('Made By KDY, v_' + datetime.today().strftime("%Y%m%d"))
        self.label.resize(50, 30)
        self.label.setFont(QFont("맑은 고딕", 10)) #폰트,크기 조절
        self.label.setStyleSheet('QLabel {padding: 1px; color:grey; }')
        self.LAYOUT_B.addWidget(self.label, 0, 0, 1, 2)
        
        # set exit button
        self.button_exit = QPushButton('EXIT', self)
        self.button_exit.clicked.connect(self.fin)
        self.button_exit.setStyleSheet('QPushButton {background-color: #c9e7ff; color: red;}')
        self.button_exit.setFont(QFont("맑은 고딕"))
        self.button_exit.setMaximumHeight(30)
        self.LAYOUT_B.addWidget(self.button_exit, 0, 2)
        
        # Load Button
        self.button_load = QPushButton("LOAD")
        self.button_load.setMaximumHeight(30)
        self.button_load.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')
        self.button_load.setFont(QFont("맑은 고딕"))
        self.LAYOUT_B.addWidget(self.button_load, 1, 0)
        self.button_load.clicked.connect(self.load_click)
        
        self.show()
        return
    
    def load_click(self) :  
        # Reload Button
        self.button_reload = QPushButton("RELOAD")
        self.button_reload.setMaximumHeight(30)
        self.button_reload.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')
        self.button_reload.setFont(QFont("맑은 고딕"))
        self.LAYOUT_B.addWidget(self.button_reload, 1, 1)
        self.button_reload.clicked.connect(self.reload_click)
        
        # Run Button
        self.button_1 = QPushButton("RUN")
        self.button_1.setMaximumHeight(30)
        self.button_1.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')
        self.button_1.setFont(QFont("맑은 고딕"))
        self.LAYOUT_B.addWidget(self.button_1, 2, 0)
        
        # Stop Button
        self.button_2 = QPushButton("STOP")
        self.button_2.setMaximumHeight(30)
        self.button_2.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')
        self.button_2.setFont(QFont("맑은 고딕"))
        self.LAYOUT_B.addWidget(self.button_2, 2, 1)
        
        pa_list = []
        re_list = []
            
        f = open('bsn_param.csv', 'r')
        
        param_list = f.read().split(',')
        self.graph_num = int(param_list[0])
        
        for i in range(self.graph_num):
            globals()['box_input{}'.format(chr(i + 1 + 64))] = float(param_list[i + 1])
            pa_list.append(globals()['box_input{}'.format(chr(i + 1 + 64))])
            
        for i in range(self.graph_num):
            globals()['relt_input{}'.format(chr(i + 1))] = float(param_list[i + 1 + self.graph_num])
            re_list.append(globals()['relt_input{}'.format(chr(i + 1))])
                
        self.samt = float(param_list[len(param_list) - 2])
        self.samn = int(param_list[len(param_list) - 1])
            
        f.close() 
        
        re_t = tuple(re_list)
        pa_t = tuple(pa_list)
        
        # Place the matplotlib figure for output
        self.myFig1 = CustomFigCanvas1(self.graph_num)
        self.LAYOUT_A.addWidget(self.myFig1, 0, 1)
        # Add the callbackfunc to ..
        myDataLoop = threading.Thread(name = 'myDataLoop', target = dataSendLoop, daemon = True, args = (self.addData_callbackFunc,))
        myDataLoop.start()
        
        # Place the matplotlib figure for histogram
        self.confFig = ConfigurationCanvas(self.graph_num)
        self.LAYOUT_B.addWidget(self.confFig, 3, 0, 1, 3)
        self.myFig1.store_andCanvas(self.confFig)
        
        # Place the matplotlib figure for mean
        self.myFig2 = CustomFigCanvas2(self.graph_num)
        self.LAYOUT_A.addWidget(self.myFig2, 0, 2)
        self.myFig1.store_canvas2(self.myFig2)
        # 변경된 값 CustomFigCanvas1에 전달
        self.myFig1.change_input(self.samt, self.samn, *pa_t)
        self.myFig1.change_relt(*re_t)
        
        # set button event for start and stop control
        self.control = Control()
        self.control.store_two_canvas(self.myFig1, self.myFig2, self.confFig)
        self.button_1.clicked.connect(self.control.start)
        self.button_2.clicked.connect(self.control.stop)
        return
        
    def fin(self) : # exit window
        sys.exit(QApplication(sys.argv))
        
    def reload_click(self) : # reload and set parameter value
        pa_list = []
        re_list = []
            
        f = open('bsn_param.csv', 'r')
        
        param_list = f.read().split(',')
        self.graph_num = int(param_list[0])
        
        for i in range(self.graph_num):
            globals()['box_input{}'.format(chr(i + 1 + 64))] = float(param_list[i + 1])
            pa_list.append(globals()['box_input{}'.format(chr(i + 1 + 64))])
            
        for i in range(self.graph_num):
            globals()['relt_input{}'.format(chr(i + 1))] = float(param_list[i + 1 + self.graph_num])
            re_list.append(globals()['relt_input{}'.format(chr(i + 1))])
                
        self.samt = float(param_list[len(param_list) - 2])
        self.samn = float(param_list[len(param_list) - 1])
            
        f.close() 
        
        re_t = tuple(re_list)
        pa_t = tuple(pa_list)
         # 변경된 값 CustomFigCanvas1에 전달
        self.myFig1.change_input(self.samt, self.samn, *pa_t)
        self.myFig1.change_relt(*re_t)

    def addData_callbackFunc(self, value):
        self.myFig1.addData(value)
        return


class Control(): # start and stop control
    def __init__(self):
        self.myFig1 = None
        self.myFig2 = None
        self.confFig = None
        
    def store_two_canvas(self, f1, f2, f3) :
        print('control test')
        self.myFig1 = f1
        self.myFig2 = f2
        self.confFig = f3
        
    def start(self) :
        self.myFig1.start()
        self.myFig2.start()
        self.confFig.start()
        
    def stop(self) :
        self.myFig1.stop()
        self.myFig2.stop()
        self.confFig.stop()


class ConfigurationCanvas(FigureCanvas):
    def __init__(self, graph_num):
        print(matplotlib.__version__)
        self.graph_num = graph_num
        self.and_num = -1
        
        for i in range(self.graph_num) :
            globals()['s{}'.format(chr(i + 65))] = -1
        
        self.plus = cp.array([])
        
        self.ani_flag = 2
        self.call_flag = -1
        self.samn = 100
        
        self.fig = Figure(figsize=(10, 20), dpi=50)  #그래프 그릴 창 생성
        self.fig.suptitle('Bit Configuration Histogram', fontsize=20)
        self.fig.subplots_adjust(hspace=1)
        
        self.rg = 2 ** self.graph_num + 1
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel('state')
        self.ax.set_ylabel('Probability')
        self.hi, bins, self.patches = self.ax.hist([], bins=range(0, self.rg, 1), rwidth=0.8, color='#abffb6')
        
        FigureCanvas.__init__(self, self.fig)
        return
    
    def change_samn(self, samn):
        self.init_plus()
        self.samn = samn
    
    def update_bsn(self, *nums):
        for i in range(self.graph_num) :
            globals()['s{}'.format(chr(i + 65))] = nums[i]
            
        self.plus = cp.append(self.plus, self.cal_and())
        
        if self.ani_flag == 2 :
            self.ani = anim.FuncAnimation(self.fig, self.animate, interval=100, blit = True) # self.animate 실시간 호출
            self.ani_flag = 1
        return
    
    def init_plus(self) : 
        print('init_plus test')
        self.plus = cp.array([]) # input num이 변했을 때 array 초기화
    
    def animate(self, i):
        if self.ani_flag == 1 and len(self.plus) == self.samn : # FuncAnimation이 start 상태이고 sampling 개수가 채워진 상태이면
            # probability를 histogram으로 표시
            self.ax.cla()
            self.ax.set_xlabel('State (decimal)')
            self.ax.set_ylabel('Probability')
            self.hi, bins, self.patches = self.ax.hist(cp.asnumpy(self.plus), bins=range(0, self.rg, 1), rwidth=0.8, color='#abffb6', density=True)
            self.init_plus()
        return self.patches
    
    def cal_and(self) : # 2진수 -> 10진수
        and_sum = 0
        for i in range(self.graph_num) :
            and_sum += globals()['s{}'.format(chr(i + 65))] * (2 ** (self.graph_num - i - 1))
        
        return and_sum
    
    def start(self) :
        self.ani.event_source.start()
        self.ani_flag = 1 # Animation start
        
    def stop(self) :
        self.ani.event_source.stop()
        self.ani_flag = -1 # Animation stop


class CustomFigCanvas1(FigureCanvas, TimedAnimation): # for constant graph
    def __init__(self, graph_num):
        print(matplotlib.__version__)
        self.graph_num = graph_num
        self.xlim = 30 # x축 length
        
        self.samt = 3.0
        self.samn = 100
        self.samn_flag = 0
        
        self.queue_flag = -1
        self.cf_list = []
        
        for i in range(self.graph_num) :
            globals()['addedData{}_1'.format(chr(i + 65))] = cp.array([]) # for output graph
            globals()['{}_1'.format(chr(i + 97))] = cp.array([]) # for output graph
            globals()['input{}_1'.format(chr(i + 65))] = 0.0 # input value
            globals()['flag{}_1'.format(chr(i + 65))] = -10.0 # 슬라이더 값이 변경됐을 때 mean list 초기화를 위한 flag 변수
            globals()['queue{}'.format(chr(i + 65))] = Queue(self.samn) # Queue for calculate mean 
            globals()['xval{}'.format(i + 1)] = 30
            globals()['relt{}'.format(i + 1)] = 1.0 # t_r
            globals()['n{}'.format(i + 1)] = cp.linspace(0, self.xlim * 1000 - 1, self.xlim * 1000) * globals()['relt{}'.format(i + 1)]
        
        self.myFig = None # CustomFigCanvas2 object
        self.confFig = None # ConfigurationCanvas object
        
        self.ani_flag = 1 # Animation start
        
        self.cal_graph(self.xlim, 0, -1) # {}_1 list init
        timer.start(self.samt * 1000) # timer interval : self.samt * 1000ms
        timer.timeout.connect(self.timer_graph)
        
        self.fig = Figure(figsize=(10, 20), dpi=70)  #그래프 그릴 창 생성
        self.fig.suptitle('OUTPUT', fontsize=16)
        self.fig.subplots_adjust(hspace=1)
        
        for i in range(1, self.graph_num + 1) :
            globals()['ax{}_1'.format(i)] = self.fig.add_subplot(self.graph_num, 1, i)
            globals()['ax{}_1'.format(i)].set_xlabel('number of times')
            globals()['ax{}_1'.format(i)].set_ylabel('BSN')
            globals()['line{}_1'.format(i)] = Line2D([], [], color='blue', drawstyle='steps')
            globals()['ax{}_1'.format(i)].add_line(globals()['line{}_1'.format(i)])
            globals()['ax{}_1'.format(i)].set_xlim(0, self.xlim - 1)
            globals()['ax{}_1'.format(i)].set_ylim(-0.1, 1.1)
        
        FigureCanvas.__init__(self, self.fig)
        TimedAnimation.__init__(self, self.fig, interval = 100, blit = True)
        setattr(self, '_draw_was_started', True)
        return
    
    def cal_graph(self, num, state, g_n) :
        for i in range(self.graph_num) :
            for j in range(num) :
                input = globals()['input{}_1'.format(chr(i + 65))]
                p_input = 1.0 / (1.0 + cp.exp(-input)) # the probability given by the sigmoid activation function 
                p = torch.FloatTensor(p_input) # float -> tensor convert
                ran = torch.rand(p.size()) #uniform distribution에서 random하게 0~1 사이의 숫자를 얻음
                # probability가 uniform distribution random 값 보다 크면 1, 아니면 -1로 출력
                p_ = p - ran
                p_sign = torch.sign(p_) 
                globals()['s{}_1'.format(chr(i + 65))] = torch.nn.functional.relu(p_sign) # -1로 출력된 값들은 모두 0으로 처리                         
                
                if state == 0 : # init
                    globals()['{}_1'.format(chr(i + 97))] = cp.append(globals()['{}_1'.format(chr(i + 97))], globals()['s{}_1'.format(chr(i + 65))]) # BSN 저장 list에 저장
                elif state == 1 : # with Timer
                    if globals()['queue{}'.format(chr(i + 65))].full() : # queue가 full이면 get()
                        globals()['queue{}'.format(chr(i + 65))].get()
                    globals()['queue{}'.format(chr(i + 65))].put(globals()['s{}_1'.format(chr(i + 65))])
                    self.queue_flag = globals()['s{}_1'.format(chr(i + 65))]
                    
                    if self.confFig != None :
                        self.cf_list = []
                        for i in range(self.graph_num) :
                            self.cf_list.append(globals()['s{}_1'.format(chr(i + 65))])
                elif state == 2 : # for output graph
                    if self.queue_flag != -1 : # 이미 queue에 삽입되기 위해 생성된 neuron 값이 있으면
                        globals()['addedData{}_1'.format(chr(i + 65))] = cp.append(globals()['addedData{}_1'.format(chr(i + 65))], self.queue_flag)
                    else :
                        globals()['addedData{}_1'.format(chr(i + 65))] = cp.append(globals()['addedData{}_1'.format(chr(i + 65))], globals()['s{}_1'.format(chr(i + 65))])
                    self.queue_flag = -1
                elif state == 3 and i == g_n : # t_r < 1.0
                    globals()['{}_1'.format(chr(i + 97))] = cp.append(globals()['{}_1'.format(chr(i + 97))], 
                                                                      globals()['s{}_1'.format(chr(i + 65))])
                    
        if state == 1 :
            self.samn_flag += 1
            print(self.samn_flag)
            if self.cf_list != [] :
                cf_t = tuple(self.cf_list)
                self.confFig.update_bsn(*cf_t) # ConfigurationCanvas에 BSN 전달
    
    def store_andCanvas(self, f) :
        self.confFig = f
    
    def store_canvas2(self, f) : # CustomFigCanvas2 객체 저장 및 scatter initializing for BSN list 전달
        self.myFig = f
        
        pa_list = []
        for i in range(self.graph_num) :
            pa_list.append(globals()['{}_1'.format(chr(i + 97))])
        
        pa_t = tuple(pa_list)
        self.myFig.init_scatter(*pa_t)
        
    
    def change_input(self, samt, samn, *nums): # 슬라이더 값 변경에 따른 input 값 변경
        for i in range(len(nums)) :
            globals()['input{}_1'.format(chr(i + 1 + 64))] = nums[i]
            print(globals()['input{}_1'.format(chr(i + 1 + 64))])
            
        if self.samt != samt or self.samn != samn :
            self.samt = samt
            timer.setInterval(self.samt * 1000) # timer interval init
            self.samn = samn
            self.confFig.change_samn(samn)
            for i in range(len(nums)) : # queue init
                globals()['queue{}'.format(chr(i + 65))] = Queue(self.samn)
            self.samn_flag = 0
            
    def change_relt(self, *relt): # t_r change
        for i in range(len(relt)) :
            if globals()['relt{}'.format(i + 1)] != relt[i] :
                globals()['relt{}'.format(i + 1)] = relt[i]
                if globals()['relt{}'.format(i + 1)] < 1.0 : # time constant가 1.0 미만이면 output graph에 표시할 데이터 크기 증가
                    globals()['xval{}'.format(i + 1)] = round(1.0 / globals()['relt{}'.format(i + 1)] * self.xlim)
                    globals()['{}_1'.format(chr(i + 97))] = cp.array([])
                    self.cal_graph(globals()['xval{}'.format(i + 1)], 3, i) # a_1, b_1 ... init 후 다시 xval만큼 생성
                globals()['n{}'.format(i + 1)] = cp.linspace(0, self.xlim * 1000 - 1, self.xlim * 1000) * globals()['relt{}'.format(i + 1)]
            
    def new_frame_seq(self):
        return iter(range(globals()['n{}'.format(1)].size))

    def _init_draw(self):
        for i in range(1, self.graph_num + 1) :
            globals()['lines{}_1'.format(i)] = [globals()['line{}_1'.format(i)]]
            for l in globals()['lines{}_1'.format(i)] :
                l.set_data([], [])
        return
    
    def timer_graph(self) :
        self.cal_graph(1, 1, -1)

    def addData(self, value):
        if self.ani_flag == 1 : # if snimation start
            for i in range(self.graph_num) :
                # 이전 input 값과 현재 input 값이 다르면 mean list 초기화
                if globals()['flag{}_1'.format(chr(i + 65))] != globals()['input{}_1'.format(chr(i + 65))] :
                    print('test{}_1'.format(chr(i + 65)))
                    globals()['queue{}'.format(chr(i + 65))] = Queue(self.samn)
                    self.samn_flag = 0
                    self.confFig.init_plus()

                globals()['flag{}_1'.format(chr(i + 65))] = globals()['input{}_1'.format(chr(i + 65))]

            self.cal_graph(1, 2, -1)

            # mean update by sampling number interval
            if self.myFig != None and globals()['queue{}'.format(chr(i + 65))].full and self.samn_flag == self.samn :
                self.samn_flag = 0
                pa_list = []  
                for i in range(self.graph_num) :
                    pa_list.append(cp.array(globals()['queue{}'.format(chr(i + 65))].queue))
                for i in range(self.graph_num) :
                    pa_list.append(globals()['input{}_1'.format(chr(i + 65))])
                pa_t = tuple(pa_list)
                self.myFig.update_list(*pa_t)
        return

    def _step(self, *args):
        try:
            TimedAnimation._step(self, *args)
        except Exception as e:
            TimedAnimation._stop(self)
            pass
        return

    def _draw_frame(self, framedata):
        for i in range(self.graph_num) :
            while(len(globals()['addedData{}_1'.format(chr(i + 65))]) > 0) :
                globals()['{}_1'.format(chr(i + 97))] = cp.roll(globals()['{}_1'.format(chr(i + 97))], -1)
                globals()['{}_1'.format(chr(i + 97))][-1] = globals()['addedData{}_1'.format(chr(i + 65))][0]
                globals()['addedData{}_1'.format(chr(i + 65))] = globals()['addedData{}_1'.format(chr(i + 65))][1:]
                
        for i in range(self.graph_num) :
            globals()['line{}_1'.format(i + 1)].set_data(cp.asnumpy(globals()['n{}'.format(i + 1)][: globals()['xval{}'.format(i + 1)]]), cp.asnumpy(globals()['{}_1'.format(chr(i + 97))][: globals()['xval{}'.format(i + 1)]]))
            globals()['_drawn_artists{}'.format(i + 1)] = [globals()['line{}_1'.format(i + 1)]]
        return
    
    def start(self) :
        TimedAnimation.__init__(self, self.fig, interval = 100, blit = True)
        setattr(self, '_draw_was_started', True)
        self.ani_flag = 1 # Animation start
        timer.start(self.samt * 1000)
        
    def stop(self) :
        TimedAnimation._stop(self)
        self.ani_flag = -1 # Animation stop
        timer.stop()


class CustomFigCanvas2(FigureCanvas):
    def __init__(self, graph_num):
        print(matplotlib.__version__)
        self.graph_num = graph_num
        self.mae = [] # output 평균 저장 list       

        for i in range(self.graph_num) :
            globals()['{}_2'.format(chr(i + 97))] = cp.array([])
            globals()['input{}_2'.format(chr(i + 65))] = 0.0
        
        self.ani_flag = 2
        self.call_flag = -1
        
        for i in range(-6, 6) :
            list = [] # BSN 저장 list
            p_num = 1.0 / (1.0 + cp.exp(-i)) # the probability given by the sigmoid activation function
            p = torch.FloatTensor(p_num) # float -> tensor convert

            for j in range(1000) :
                ran = torch.rand(p.size()) #uniform distribution에서 random하게 0~1 사이의 숫자를 얻음
                # probability가 uniform distribution random 값 보다 크면 1, 아니면 -1로 출력
                p_ = p - ran
                p_sign = torch.sign(p_) 
                s = torch.nn.functional.relu(p_sign) # relu 함수를 이용해 -1로 출력된 값들은 모두 0으로 처리
                list.append(s.item())
            self.mae.append(cp.mean(cp.array(list)).item()) # output 평균 저장
            
        spl = make_interp_spline(np.arange(-6, 6, 1), self.mae)
        self.mae_new = spl(np.arange(-6, 6, 0.01))
        
        self.fig = Figure(figsize=(10, 20), dpi=70)  #그래프 그릴 창 생성
        self.fig.suptitle('MEAN', fontsize=16)
        self.fig.subplots_adjust(hspace=1)
        
        for i in range(1, self.graph_num + 1) :
            globals()['ax{}_2'.format(i)] = self.fig.add_subplot(self.graph_num, 1, i)
            globals()['ax{}_2'.format(i)].set_xlabel('input number')
            globals()['ax{}_2'.format(i)].set_ylabel('the mean of the output')
            globals()['line{}_2'.format(i)] = Line2D(np.arange(-6, 6, 0.01), self.mae_new, color='red') # drawing sigmoid function
            globals()['ax{}_2'.format(i)].add_line(globals()['line{}_2'.format(i)])
            globals()['ax{}_2'.format(i)].set_xlim(-5.1, 5.1)
            globals()['ax{}_2'.format(i)].set_ylim(-0.1, 1.1)
            
        FigureCanvas.__init__(self, self.fig)
        return
    
    def init_scatter(self, *nums): 
        for i in range(len(nums)) :
            globals()['{}_2'.format(chr(i + 97))] = nums[i]
            
        for i in range(self.graph_num) :
            globals()['scat{}'.format(i + 1)] = None
            globals()['scat_init{}'.format(i + 1)] = globals()['ax{}_2'.format(i + 1)].scatter(globals()['input{}_2'.format(chr(i + 65))], cp.mean(globals()['{}_2'.format(chr(i + 97))]).item(), s=250, color='red')
        
    def update_list(self, *nums):
        for i in range(self.graph_num) :
            globals()['{}_2'.format(chr(i + 97))] = nums[i] # store queue value
        
        for i in range(self.graph_num, self.graph_num * 2) :
            globals()['input{}_2'.format(chr(i - self.graph_num + 65))] = nums[i]
        
        self.call_flag = 0
        if self.ani_flag == 2 :
            print('ani_flag test')
            for i in range(1, self.graph_num + 1) :
                self.ani_flag = 1
                globals()['scat_init{}'.format(i)].remove()
                globals()['ani{}'.format(i)] = anim.FuncAnimation(self.fig, self.animate, fargs=(i,), interval=100, blit=True)
        return
    
    def animate(self, i, num):
        if self.ani_flag == 1 and self.call_flag < self.graph_num : # FuncAnimation이 start 상태이고 update_list가 호출된 상태이면
            if globals()['scat{}'.format(num)] != None :
                globals()['scat{}'.format(num)].remove() # 기존 scatter 삭제
            globals()['scat{}'.format(num)] = globals()['ax{}_2'.format(num)].scatter(globals()['input{}_2'.format(chr(num + 64))], cp.mean(globals()['{}_2'.format(chr(num + 96))]).item(), s=250, color='red') # scatter update
            self.call_flag += 1
        return globals()['scat{}'.format(num)],
    
    def start(self) :
        if self.call_flag != -1 :
            for i in range(1, self.graph_num + 1) :
                globals()['ani{}'.format(i)].event_source.start()
        self.ani_flag = 1
        
    def stop(self) :
        if self.call_flag != -1 :
            for i in range(1, self.graph_num + 1) :
                globals()['ani{}'.format(i)].event_source.stop()
        self.ani_flag = -1


# +
class Communicate(QObject):
    data_signal = pyqtSignal(float)

def dataSendLoop(addData_callbackFunc):
    # Setup the signal-slot mechanism.
    mySrc = Communicate()
    mySrc.data_signal.connect(addData_callbackFunc)

    # Simulate some data
    n = cp.linspace(0, 499, 500)
    y = 50 + 25*(cp.sin(n / 8.3)) + 10*(cp.sin(n / 7.5)) - 5*(cp.sin(n / 1.5))
    i = 0

    while(True):
        if(i > 499):
            i = 0
        time.sleep(0.1)
        mySrc.data_signal.emit(y[i]) # <- Here you emit a signal!
        i += 1


# -

if __name__== '__main__':
    app = QApplication(sys.argv)
    myGUI = CustomMainWindow()
    sys.exit(app.exec_())


