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

from __future__ import division
import numpy as np
import random
import itertools
import os
from scipy.fftpack import fft
from scipy import signal
from random import uniform
import torch
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
import threading
from datetime import datetime
from queue import Queue


class CustomMainWindow(QMainWindow): # about the main window
    def __init__(self):
        super(CustomMainWindow, self).__init__()
        # Define the geometry of the main window
        self.setGeometry(100, 100, 1300, 900)
        self.setWindowTitle("Binary Stochastic Neuron")
        
        self.input_flag = 0
        self.inc_flag = 1
        self.Variable_ea = 65
        
        self.constant = 1.0
        self.samt = 3.0
        self.samn = 100
        global timer 
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
        
        # set Graph Num
        self.label = QLabel(self)
        self.label.setText('Set Bits Num')
        self.label.resize(50, 30)
        self.label.setFont(QFont("맑은 고딕", 13)) #폰트,크기 조절
        self.label.setStyleSheet('QLabel {padding: 1px;}')
        self.LAYOUT_B.addWidget(self.label, 1, 0)
        
        self.textbox = QLineEdit(self)
        self.textbox.resize(140,30)
        self.textbox.setFont(QFont("맑은 고딕", 15)) #폰트,크기 조절
        self.textbox.setText('3')
        self.LAYOUT_B.addWidget(self.textbox, 1, 1)
        
        self.button = QPushButton('Set Bits Num', self)
        self.button.clicked.connect(self.init_graph)
        self.button.setStyleSheet('QPushButton {background-color: #c9e7ff; color: red;}')
        self.button.setFont(QFont("맑은 고딕"))
        self.button.setMaximumHeight(30)
        self.LAYOUT_B.addWidget(self.button, 1, 2)
        
        # set exit button
        self.button_exit = QPushButton('EXIT', self)
        self.button_exit.clicked.connect(self.fin)
        self.button_exit.setStyleSheet('QPushButton {background-color: #c9e7ff; color: red;}')
        self.button_exit.setFont(QFont("맑은 고딕"))
        self.button_exit.setMaximumHeight(30)
        self.LAYOUT_B.addWidget(self.button_exit, 0, 2)
        
        self.show()
        
    def fin(self) :
        sys.exit(QApplication(sys.argv))
    
    def init_graph(self) :
        self.graph_num = int(self.textbox.text())
        print(self.graph_num)
        
        self.LAYOUT_B.removeWidget(self.label)
        self.LAYOUT_B.removeWidget(self.textbox)
        self.LAYOUT_B.removeWidget(self.button)
        self.label.deleteLater()
        self.label = None
        self.textbox.deleteLater()
        self.textbox = None
        self.button.deleteLater()
        self.button = None
        
        # Auto Num CheckBox
        self.auto_check = QCheckBox('Random Input')
        self.auto_check.stateChanged.connect(self.auto_click)
        self.auto_check.setFont(QFont("맑은 고딕", 12)) #폰트,크기 조절
        self.LAYOUT_B.addWidget(self.auto_check, 1, 2)   
        
        # Add Input
        for i in range(self.graph_num) :
            self.add_inputbox()
            
        # Add Parameter
        self.add_parambox('con', 'T_r', '1.0')
        self.add_parambox('samt', 'T_s', '3.0')
        self.add_parambox('samn', 'N_s', '100')
        
        # Run Button
        self.button_1 = QPushButton("RUN")
        self.button_1.setMaximumHeight(30)
        self.button_1.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')
        self.button_1.setFont(QFont("맑은 고딕"))
        self.LAYOUT_B.addWidget(self.button_1, self.inc_flag + 2, 0)
        
        # Stop Button
        self.button_2 = QPushButton("STOP")
        self.button_2.setMaximumHeight(30)
        self.button_2.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')
        self.button_2.setFont(QFont("맑은 고딕"))
        self.LAYOUT_B.addWidget(self.button_2, self.inc_flag + 2, 1)
        
        # Auto Store CheckBox
        self.store_check = QCheckBox('Store Output')
        #self.store_check.stateChanged.connect(self.auto_click)
        self.store_check.setFont(QFont("맑은 고딕", 12)) #폰트,크기 조절
        self.LAYOUT_B.addWidget(self.store_check, self.inc_flag + 2, 2)   
           
        # Place the matplotlib figure for output
        self.myFig1 = CustomFigCanvas1(self.graph_num)
        self.LAYOUT_A.addWidget(self.myFig1, 0, 1)
        # Add the callbackfunc to ..
        myDataLoop = threading.Thread(name = 'myDataLoop', target = dataSendLoop, daemon = True, args = (self.addData_callbackFunc,))
        myDataLoop.start()
        
        # Place the matplotlib figure for histogram
        self.gateFig = AndGateCanvas(self.graph_num)
        self.LAYOUT_B.addWidget(self.gateFig, self.inc_flag + 1, 0, 1, 3)
        self.myFig1.store_andCanvas(self.gateFig)
        
        # Place the matplotlib figure for mean
        self.myFig2 = CustomFigCanvas2(self.graph_num)
        self.LAYOUT_A.addWidget(self.myFig2, 0, 2)
        self.myFig1.store_canvas2(self.myFig2)
        
        # set button event for start and stop control
        self.control = Control()
        self.control.store_two_canvas(self.myFig1, self.myFig2, self.gateFig)
        self.button_1.clicked.connect(self.control.start)
        self.button_2.clicked.connect(self.control.stop)
        return
    
    def add_inputbox(self) : # add input
        globals()['box_input{}'.format(chr(self.Variable_ea))] = 0.0 # input 변수 생성
        
        globals()['label{}'.format(self.inc_flag)] = QLabel(self)
        globals()['label{}'.format(self.inc_flag)].setText(chr(self.Variable_ea))
        globals()['label{}'.format(self.inc_flag)].resize(50, 30)
        globals()['label{}'.format(self.inc_flag)].setFont(QFont("맑은 고딕", 18)) #폰트,크기 조절
        globals()['label{}'.format(self.inc_flag)].setStyleSheet('QLabel {padding: 1px;}')
        self.LAYOUT_B.addWidget(globals()['label{}'.format(self.inc_flag)], self.inc_flag + 1, 0)
        
        globals()['textbox{}'.format(self.inc_flag)] = QLineEdit(self)
        globals()['textbox{}'.format(self.inc_flag)].resize(140,30)
        globals()['textbox{}'.format(self.inc_flag)].setFont(QFont("맑은 고딕", 15)) #폰트,크기 조절
        globals()['textbox{}'.format(self.inc_flag)].setText('0.0')
        self.LAYOUT_B.addWidget(globals()['textbox{}'.format(self.inc_flag)], self.inc_flag + 1, 1)
        
        globals()['button{}'.format(chr(self.Variable_ea))] = QPushButton('Change input' + chr(self.Variable_ea), self)
        globals()['button{}'.format(chr(self.Variable_ea))].clicked.connect(self.on_click)
        globals()['button{}'.format(chr(self.Variable_ea))].setStyleSheet('QPushButton {background-color: #c9e7ff; color: red;}')
        globals()['button{}'.format(chr(self.Variable_ea))].setFont(QFont("맑은 고딕"))
        globals()['button{}'.format(chr(self.Variable_ea))].setMaximumHeight(30)
        self.LAYOUT_B.addWidget(globals()['button{}'.format(chr(self.Variable_ea))], self.inc_flag + 1, 2)
        
        self.Variable_ea += 1
        self.inc_flag += 1
        self.input_flag += 1
        
    def add_parambox(self, var_name, text_name, init_num) : # add param
        globals()['label_{}'.format(var_name)] = QLabel(self)
        globals()['label_{}'.format(var_name)].setText(text_name)
        globals()['label_{}'.format(var_name)].setFont(QFont("맑은 고딕", 18)) #폰트,크기 조절
        globals()['label_{}'.format(var_name)].setStyleSheet('QLabel {padding: 1px;}')
        self.LAYOUT_B.addWidget(globals()['label_{}'.format(var_name)], self.inc_flag + 1, 0)
        
        globals()['textbox_{}'.format(var_name)] = QLineEdit(self)
        globals()['textbox_{}'.format(var_name)].resize(50,30)
        globals()['textbox_{}'.format(var_name)].setFont(QFont("맑은 고딕", 15)) #폰트,크기 조절
        globals()['textbox_{}'.format(var_name)].setText(init_num)
        self.LAYOUT_B.addWidget(globals()['textbox_{}'.format(var_name)], self.inc_flag + 1, 1)
        
        globals()['button_{}'.format(var_name)] = QPushButton('Change ' + text_name, self)
        globals()['button_{}'.format(var_name)].clicked.connect(self.on_click)
        globals()['button_{}'.format(var_name)].setStyleSheet('QPushButton {background-color: #c9e7ff; color: red;}')
        globals()['button_{}'.format(var_name)].setFont(QFont("맑은 고딕"))
        globals()['button_{}'.format(var_name)].setMaximumHeight(30)
        self.LAYOUT_B.addWidget(globals()['button_{}'.format(var_name)], self.inc_flag + 1, 2)
        
        self.inc_flag += 1
    
    def auto_click(self, state) : # auto random input checkbox event
        if state == Qt.Checked:
            pa_list = []
            
            for i in range(self.input_flag) :
                globals()['box_input{}'.format(chr(i + 1 + 64))] = round(uniform(-5.0, 5.0), 4) # 소수점 다섯 째자리에서 반올림
                globals()['textbox{}'.format(i + 1)].setText(str(globals()['box_input{}'.format(chr(i + 1 + 64))]))
                pa_list.append(globals()['box_input{}'.format(chr(i + 1 + 64))])
            
            pa_t = tuple(pa_list)
            self.myFig1.auto_num(*pa_t)
        else:
            self.on_click()
        
    def on_click(self) : # textbox 값 저장 & text 값 변경  
        pa_list = []
        
        for i in range(self.input_flag) :
            if globals()['textbox{}'.format(i + 1)].text() != "" :
                globals()['box_input{}'.format(chr(i + 1 + 64))] = float(globals()['textbox{}'.format(i + 1)].text())
                pa_list.append(globals()['box_input{}'.format(chr(i + 1 + 64))])
          
        if textbox_con.text() != "" :
            self.constant = float(textbox_con.text())
        if textbox_samt.text() != "" :
            self.samt = float(textbox_samt.text())
        if textbox_samn.text() != "" :
            self.samn = float(textbox_samn.text())
        
        pa_t = tuple(pa_list)
        self.myFig1.change_input(self.constant, self.samt, self.samn, *pa_t) # 변경된 값 CustomFigCanvas1에 전달

    def addData_callbackFunc(self, value):
        self.myFig1.addData(value)
        return


class Control(): # start and stop control
    def __init__(self):
        self.myFig1 = None
        self.myFig2 = None
        self.gateFig = None
        
    def store_two_canvas(self, f1, f2, f3) :
        print('control test')
        self.myFig1 = f1
        self.myFig2 = f2
        self.gateFig = f3
        
    def start(self) :
        self.myFig1.start()
        self.myFig2.start()
        self.gateFig.start()
        
    def stop(self) :
        self.myFig1.stop()
        self.myFig2.stop()
        self.gateFig.stop()


class AndGateCanvas(FigureCanvas):
    def __init__(self, graph_num):
        print(matplotlib.__version__)
        self.graph_num = graph_num
        self.and_num = -1
        
        for i in range(self.graph_num) :
            globals()['s{}'.format(chr(i + 65))] = -1
        
        self.plus = []
        
        self.ani_flag = 2
        self.call_flag = -1
        self.samn = 100
        
        self.fig = Figure(figsize=(10, 20), dpi=50)  #그래프 그릴 창 생성
        self.fig.suptitle('AND Gate Histogram', fontsize=20)
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
            
        self.plus.append(self.cal_and())
        t_a = torch.tensor(self.plus)
        self.arr = list(t_a.numpy())
        if self.ani_flag == 2 :
            self.ani = anim.FuncAnimation(self.fig, self.animate, interval=100, blit = True) # self.animate 실시간 호출
            self.ani_flag = 1
        return
    
    def init_plus(self) : 
        print('init_plus test')
        self.plus = [] # input num이 변했을 때 array 초기화
    
    def animate(self, i):
        if self.ani_flag == 1 and len(self.plus) == self.samn : # FuncAnimation이 start 상태이고 sampling 개수가 채워진 상태이면
            # probability를 histogram으로 표시
            self.ax.cla()
            self.ax.set_xlabel('state')
            self.ax.set_ylabel('Probability')
            self.hi, bins, self.patches = self.ax.hist(self.arr, bins=range(0, self.rg, 1), rwidth=0.8, color='#abffb6', density=True)
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
        self.xval = 30 # x축에 표시할 데이터 크기
        self.constant = 1.0
        self.n = np.linspace(0, self.xlim * 1000 - 1, self.xlim * 1000) * self.constant
        
        self.samt = 3.0
        self.samn = 100
        self.samn_flag = 0
        
        self.queue_flag = -1
        self.gf_list = []
        
        for i in range(self.graph_num) :
            globals()['addedData{}_1'.format(chr(i + 65))] = [] # for output graph
            globals()['{}_1'.format(chr(i + 97))] = [] # for output graph
            globals()['input{}_1'.format(chr(i + 65))] = 0.0 # input value
            globals()['flag{}_1'.format(chr(i + 65))] = -10.0 # 슬라이더 값이 변경됐을 때 mean list 초기화를 위한 flag 변수
            globals()['queue{}'.format(chr(i + 65))] = Queue(self.samn) # Queue for calculate mean 
        
        self.myFig = None # CustomFigCanvas2 object
        self.gateFig = None # AndGateCanvas object
        
        self.ani_flag = 1 # Animation start
        
        self.cal_graph(self.xlim, 0) # {}_1 list init
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
    
    def cal_graph(self, num, state) :
        for i in range(self.graph_num) :
            for j in range(num) :
                input = globals()['input{}_1'.format(chr(i + 65))]
                p_input = 1.0 / (1.0 + np.exp(-input)) # the probability given by the sigmoid activation function 
                p = torch.FloatTensor([p_input]) # float -> tensor convert
                ran = torch.rand(p.size()) #uniform distribution에서 random하게 0~1 사이의 숫자를 얻음
                # probability가 uniform distribution random 값 보다 크면 1, 아니면 -1로 출력
                p_ = p - ran
                p_sign = torch.sign(p_) 
                globals()['s{}_1'.format(chr(i + 65))] = torch.nn.functional.relu(p_sign) # relu 함수를 이용해 -1로 출력된 값들은 모두 0으로 처리                         
                
                if state == 0 : # init
                    globals()['{}_1'.format(chr(i + 97))].append(globals()['s{}_1'.format(chr(i + 65))]) # BSN 저장 list에 저장
                elif state == 1 : # with Timer
                    if globals()['queue{}'.format(chr(i + 65))].full() : # queue가 full이면 get()
                        globals()['queue{}'.format(chr(i + 65))].get()
                    globals()['queue{}'.format(chr(i + 65))].put(globals()['s{}_1'.format(chr(i + 65))])
                    self.queue_flag = globals()['s{}_1'.format(chr(i + 65))]
                    
                    if self.gateFig != None :
                        self.gf_list = []
                        for i in range(self.graph_num) :
                            self.gf_list.append(globals()['s{}_1'.format(chr(i + 65))])
                        
                elif state == 2 : # for output graph
                    if self.queue_flag != -1 : # 이미 queue에 삽입되기 위해 생성된 neuron 값이 있으면
                        globals()['addedData{}_1'.format(chr(i + 65))].append(self.queue_flag)
                    else :
                        globals()['addedData{}_1'.format(chr(i + 65))].append(globals()['s{}_1'.format(chr(i + 65))])
                    self.queue_flag = -1
                    
        if state == 1 :
            self.samn_flag += 1
            print(self.samn_flag)
            if self.gf_list != [] :
                gf_t = tuple(self.gf_list)
                self.gateFig.update_bsn(*gf_t) # AndGateCanvas에 BSN 전달
    
    def store_andCanvas(self, f) :
        self.gateFig = f
    
    def store_canvas2(self, f) : # CustomFigCanvas2 객체 저장 및 scatter initializing for BSN list 전달
        self.myFig = f
        
        pa_list = []
        for i in range(self.graph_num) :
            pa_list.append(globals()['{}_1'.format(chr(i + 97))])
        
        pa_t = tuple(pa_list)
        self.myFig.init_scatter(*pa_t)
        
    def auto_num(self, *nums) : # auto num 받아서 값 저장
        for i in range(len(nums)) :
            globals()['input{}_1'.format(chr(i + 1 + 64))] = nums[i]
            print(globals()['input{}_1'.format(chr(i + 1 + 64))])
    
    def change_input(self, constant, samt, samn, *nums): # 슬라이더 값 변경에 따른 input 값 변경
        for i in range(len(nums)) :
            globals()['input{}_1'.format(chr(i + 1 + 64))] = nums[i]
            print(globals()['input{}_1'.format(chr(i + 1 + 64))])
        
        self.constant = constant
        if self.constant < 1.0 : # time constant가 1.0 미만이면 output graph에 표시할 데이터 크기 증가
            self.xval = round(1.0 / self.constant * self.xlim)
            for i in range(self.graph_num) :
                globals()['{}_1'.format(chr(i + 97))] = []
            self.cal_graph(self.xval, 0) # a_1, b_1 ... init 후 다시 xval만큼 생성
        self.n = np.linspace(0, self.xlim * 1000 - 1, self.xlim * 1000) * self.constant
            
        if self.samt != samt or self.samn != samn :
            self.samt = samt
            timer.setInterval(self.samt * 1000) # timer interval init
            self.samn = samn
            self.gateFig.change_samn(samn)
            for i in range(len(nums)) : # queue init
                globals()['queue{}'.format(chr(i + 65))] = Queue(self.samn)
            self.samn_flag = 0

    def new_frame_seq(self):
        return iter(range(self.n.size))

    def _init_draw(self):
        for i in range(1, self.graph_num + 1) :
            globals()['lines{}_1'.format(i)] = [globals()['line{}_1'.format(i)]]
            for l in globals()['lines{}_1'.format(i)] :
                l.set_data([], [])
        return
    
    def timer_graph(self) :
        self.cal_graph(1, 1)

    def addData(self, value):
        if self.ani_flag == 1 : # if snimation start
            for i in range(self.graph_num) :
                # 이전 input 값과 현재 input 값이 다르면 mean list 초기화
                if globals()['flag{}_1'.format(chr(i + 65))] != globals()['input{}_1'.format(chr(i + 65))] :
                    print('test{}_1'.format(chr(i + 65)))
                    globals()['queue{}'.format(chr(i + 65))] = Queue(self.samn)
                    self.samn_flag = 0
                    self.gateFig.init_plus()

                globals()['flag{}_1'.format(chr(i + 65))] = globals()['input{}_1'.format(chr(i + 65))]

            self.cal_graph(1, 2)

            # mean update by sampling number interval
            if self.myFig != None and globals()['queue{}'.format(chr(i + 65))].full and self.samn_flag == self.samn :
                self.samn_flag = 0
                pa_list = []  
                for i in range(self.graph_num) :
                    pa_list.append(globals()['queue{}'.format(chr(i + 65))].queue)
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
                globals()['{}_1'.format(chr(i + 97))] = np.roll(globals()['{}_1'.format(chr(i + 97))], -1)
                globals()['{}_1'.format(chr(i + 97))][-1] = globals()['addedData{}_1'.format(chr(i + 65))][0]
                del(globals()['addedData{}_1'.format(chr(i + 65))][0])
                
        for i in range(self.graph_num) :
            globals()['line{}_1'.format(i + 1)].set_data(self.n[: self.xval], globals()['{}_1'.format(chr(i + 97))][: self.xval])
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
            globals()['{}_2'.format(chr(i + 97))] = []
            globals()['input{}_2'.format(chr(i + 65))] = 0.0
        
        self.ani_flag = 2
        self.call_flag = -1
        
        for i in range(-6, 6) :
            list = [] # BSN 저장 list
            p_num = 1.0 / (1.0 + np.exp(-i)) # the probability given by the sigmoid activation function
            p = torch.FloatTensor([p_num]) # float -> tensor convert

            for j in range(1000) :
                ran = torch.rand(p.size()) #uniform distribution에서 random하게 0~1 사이의 숫자를 얻음
                # probability가 uniform distribution random 값 보다 크면 1, 아니면 -1로 출력
                p_ = p - ran
                p_sign = torch.sign(p_) 
                s = torch.nn.functional.relu(p_sign) # relu 함수를 이용해 -1로 출력된 값들은 모두 0으로 처리
                list.append(s)
            self.mae.append(np.mean(list)) # output 평균 저장
            
        x_new = np.arange(-6, 6, 0.01)
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
            globals()['scat_init{}'.format(i + 1)] = globals()['ax{}_2'.format(i + 1)].scatter(globals()['input{}_2'.format(chr(i + 65))], np.mean(globals()['{}_2'.format(chr(i + 97))]), s=250, color='red')
        
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
            globals()['scat{}'.format(num)] = globals()['ax{}_2'.format(num)].scatter(globals()['input{}_2'.format(chr(num + 64))], np.mean(globals()['{}_2'.format(chr(num + 96))]), s=250, color='red') # scatter update
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
    n = np.linspace(0, 499, 500)
    y = 50 + 25*(np.sin(n / 8.3)) + 10*(np.sin(n / 7.5)) - 5*(np.sin(n / 1.5))
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




