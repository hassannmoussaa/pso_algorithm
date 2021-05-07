# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 15:23:25 2021

@author: hassa
"""

import numpy as np 
from math import *

signals = []
for i in range(0 , 200* 1000) :
    root = 200 - i * float(((200) /  1000))
    if(root == 0):
        break
    signals.append(root)
    
        
        
        
    
        
