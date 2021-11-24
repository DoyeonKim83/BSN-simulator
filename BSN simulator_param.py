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
import random
import os
from random import uniform
import tensorflow as tf
import sys
import csv
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
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
        self.setGeometry(100, 100, 700, 900)
        self.setWindowTitle("BSN Parameter Setting")
        
        self.input_flag = 0
        self.inc_flag = 1
        self.Variable_ea = 65
        
        # Create FRAME_A
        self.FRAME_A = QFrame(self)
        self.LAYOUT_A = QGridLayout()
        self.FRAME_A.setLayout(self.LAYOUT_A)
        self.FRAME_A.setFixedWidth(350)
        
        # Create FRAME_B
        self.FRAME_B = QFrame(self)
        self.LAYOUT_B = QGridLayout()
        self.FRAME_B.setLayout(self.LAYOUT_B)
        self.FRAME_B.setFixedWidth(350)
        
        #Create and Set Layout
        self.hboxlayout = QHBoxLayout()
        self.hboxlayout.addWidget(self.FRAME_A)
        self.hboxlayout.addWidget(self.FRAME_B)
        self.widget = QWidget()
        self.widget.setLayout(self.hboxlayout)
        self.setCentralWidget(self.widget)
        
        # name & version
        self.label = QLabel(self)
        self.label.setText('Made By KDY, v_' + datetime.today().strftime("%Y%m%d"))
        self.label.resize(50, 30)
        self.label.setFont(QFont("맑은 고딕", 10)) #폰트,크기 조절
        self.label.setStyleSheet('QLabel {padding: 1px; color:grey; }')
        self.LAYOUT_A.addWidget(self.label, 0, 0, 1, 2)
        
        # set Graph Num
        self.label = QLabel(self)
        self.label.setText('Set Bits Num')
        self.label.resize(50, 30)
        self.label.setFont(QFont("맑은 고딕", 13)) #폰트,크기 조절
        self.label.setStyleSheet('QLabel {padding: 1px;}')
        self.LAYOUT_A.addWidget(self.label, 1, 0)
        
        self.textbox = QLineEdit(self)
        self.textbox.resize(140,30)
        self.textbox.setFont(QFont("맑은 고딕", 15)) #폰트,크기 조절
        self.textbox.setText('3')
        self.LAYOUT_A.addWidget(self.textbox, 1, 1)
        
        self.button = QPushButton('Set Bits Num', self)
        self.button.clicked.connect(self.init_graph)
        self.button.setStyleSheet('QPushButton {background-color: #c9e7ff; color: red;}')
        self.button.setFont(QFont("맑은 고딕"))
        self.button.setMaximumHeight(30)
        self.LAYOUT_A.addWidget(self.button, 1, 2)
        
        # set exit button
        self.button_exit = QPushButton('EXIT', self)
        self.button_exit.clicked.connect(self.fin)
        self.button_exit.setStyleSheet('QPushButton {background-color: #c9e7ff; color: red;}')
        self.button_exit.setFont(QFont("맑은 고딕"))
        self.button_exit.setMaximumHeight(30)
        self.LAYOUT_A.addWidget(self.button_exit, 0, 2)
        
        self.show()
        
    def fin(self) : # exit window
        sys.exit(QApplication(sys.argv))
    
    def init_graph(self) :
        self.graph_num = int(self.textbox.text())
        print(self.graph_num)
        
        self.LAYOUT_A.removeWidget(self.label)
        self.LAYOUT_A.removeWidget(self.textbox)
        self.LAYOUT_A.removeWidget(self.button)
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
        self.LAYOUT_A.addWidget(self.auto_check, 1, 2)   
        
        # Add Input
        for i in range(self.graph_num) :
            self.add_inputbox()
        
        # Save Button
        self.button_1 = QPushButton("SAVE")
        self.button_1.setMaximumHeight(30)
        self.button_1.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')
        self.button_1.setFont(QFont("맑은 고딕"))
        self.LAYOUT_A.addWidget(self.button_1, self.inc_flag + 2, 0)
        
        # Load Button
        self.button_2 = QPushButton("LOAD")
        self.button_2.setMaximumHeight(30)
        self.button_2.setStyleSheet('QPushButton {background-color: #A3C1DA; color: red;}')
        self.button_2.setFont(QFont("맑은 고딕"))
        self.LAYOUT_A.addWidget(self.button_2, self.inc_flag + 2, 1)
        
        self.inc_flag = 1
        
        # Add relaxion time
        for i in range(1, self.graph_num + 1) :
            self.add_parambox('relt' + str(i), 'T_r' + str(i), '1.0')
        
        # Add Parameter
        self.add_parambox('samt', 'T_s', '3.0')
        self.add_parambox('samn', 'N_s', '100')
        
        self.button_1.clicked.connect(self.save_click)
        self.button_2.clicked.connect(self.load_click)
        
        return
    
    def add_inputbox(self) : # add input
        globals()['label{}'.format(self.inc_flag)] = QLabel(self)
        globals()['label{}'.format(self.inc_flag)].setText(chr(self.Variable_ea))
        globals()['label{}'.format(self.inc_flag)].resize(50, 30)
        globals()['label{}'.format(self.inc_flag)].setFont(QFont("맑은 고딕", 18)) #폰트,크기 조절
        globals()['label{}'.format(self.inc_flag)].setStyleSheet('QLabel {padding: 1px;}')
        self.LAYOUT_A.addWidget(globals()['label{}'.format(self.inc_flag)], self.inc_flag + 1, 0, 1, 2)
        
        globals()['textbox{}'.format(self.inc_flag)] = QLineEdit(self)
        globals()['textbox{}'.format(self.inc_flag)].resize(140,30)
        globals()['textbox{}'.format(self.inc_flag)].setFont(QFont("맑은 고딕", 15)) #폰트,크기 조절
        globals()['textbox{}'.format(self.inc_flag)].setText('0.0')
        self.LAYOUT_A.addWidget(globals()['textbox{}'.format(self.inc_flag)], self.inc_flag + 1, 1, 1, 2)
        
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
        
        self.inc_flag += 1
        
    def save_click(self, state) : # save parameter value
        f = open('bsn_param.csv', 'w')
        wr = csv.writer(f)
        
        pa_list = []
        pa_list.append(self.graph_num)
        
        for i in range(self.graph_num):
            if globals()['textbox{}'.format(i + 1)].text() != "" :
                pa_list.append(globals()['textbox{}'.format(i + 1)].text())
                               
        for i in range(self.graph_num):
            if globals()['textbox_relt{}'.format(i + 1)].text() != "" :
                pa_list.append(globals()['textbox_relt{}'.format(i + 1)].text())
                               
        pa_list.append(textbox_samt.text())
        pa_list.append(textbox_samn.text())
        
        wr.writerow(pa_list) # csv 파일에 저장
        print('save parameter number')
        f.close()
        
    def load_click(self, state) : # load parameter value
        f = open('bsn_param.csv', 'r')
        
        pa_list = f.read().split(',')
        gr_n = int(pa_list[0])
        
        if gr_n == self.graph_num :
            for i in range(gr_n):
                globals()['textbox{}'.format(i + 1)].setText(pa_list[i + 1])
            for i in range(gr_n):
                globals()['textbox_relt{}'.format(i + 1)].setText(pa_list[i + 1 + gr_n])  
                
            textbox_samt.setText(pa_list[len(pa_list) - 2])
            textbox_samn.setText(pa_list[len(pa_list) - 1])
        else :
            print('bits num incorrect')
        f.close()
        
    def auto_click(self, state) : # auto random input checkbox event
        if state == Qt.Checked:
            for i in range(self.graph_num) :
                globals()['textbox{}'.format(i + 1)].setText(str(round(uniform(-5.0, 5.0), 4)))       


if __name__== '__main__':
    app = QApplication(sys.argv)
    myGUI = CustomMainWindow()
    sys.exit(app.exec_())
