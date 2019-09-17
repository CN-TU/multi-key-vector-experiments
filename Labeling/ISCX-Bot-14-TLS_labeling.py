# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2019, Institute of Telecommunications, TU Wien
#
# Name        : ISCX-Bot-14-TLS_labeling.py
# Description : label the TLS version of the multi-key feature vector
# Author      : Fares Meghdouri
#
#******************************************************************************

import pandas as pd
import numpy as np
import datetime
import sys

data = pd.DataFrame()

def read_data(day):
	global data
	print(">> Labeling {}".format(day))
	print(">> Loading the data")
	data = pd.read_csv(day)
	data['Attack'] = 'Normal'
	data['Label']  = '0'

def save_data(day):
	global data
	print(">> Saving {}".format(day))
	data.to_csv("{}.csv".format(day), index= False)
	del data
	print("#"*20)

def setting_label(ip1="", ip2="", attack=""):
	
	print(">> Adding {}".format(attack))

	if ip1 and ip2:
		data['Attack'] = np.where((data['sourceIPAddress'] == ip1) & 
								  (data['destinationIPAddress'] == ip2),
								   attack, data['Attack'])
		data['Attack'] = np.where((data['sourceIPAddress'] == ip2) & 
								  (data['destinationIPAddress'] == ip1),
								   attack, data['Attack'])
		return

	if ip1:
		data['Attack'] = np.where((data['sourceIPAddress'] == ip1) | 
								  (data['destinationIPAddress'] == ip1),
								   attack, data['Attack'])
		return

# here start ######################################################################

print(">> Welcome to the labeling script <<")
###################################################################
read_data(sys.argv[1])

setting_label(ip1="192.168.2.112", ip2="131.202.243.84", 
	          attack="IRC")
setting_label(ip1="192.168.5.122", ip2="198.164.30.2", 
	          attack="IRC")
setting_label(ip1="192.168.2.110", ip2="192.168.5.122", 
	          attack="IRC")
setting_label(ip1="192.168.4.118", ip2="192.168.5.122", 
	          attack="IRC")
setting_label(ip1="192.168.2.113", ip2="192.168.5.122", 
	          attack="IRC")
setting_label(ip1="192.168.1.103", ip2="192.168.5.122", 
	          attack="IRC")
setting_label(ip1="192.168.4.120", ip2="192.168.5.122", 
	          attack="IRC")
setting_label(ip1="192.168.2.112", ip2="192.168.2.110", 
	          attack="IRC")
setting_label(ip1="192.168.2.112", ip2="192.168.4.120", 
	          attack="IRC")
setting_label(ip1="192.168.2.112", ip2="192.168.1.103", 
	          attack="IRC")
setting_label(ip1="192.168.2.112", ip2="192.168.2.113", 
	          attack="IRC")
setting_label(ip1="192.168.2.112", ip2="192.168.4.118", 
	          attack="IRC")
setting_label(ip1="192.168.2.112", ip2="192.168.2.109", 
	          attack="IRC")
setting_label(ip1="192.168.2.112", ip2="192.168.2.105", 
	          attack="IRC")
setting_label(ip1="192.168.1.105", ip2="192.168.5.122", 
	          attack="IRC")

setting_label(ip1="147.32.84.180", 
	          attack="Neris")

setting_label(ip1="147.32.84.170", 
	          attack="RBot")

setting_label(ip1="147.32.84.150", 
	          attack="Menti")

setting_label(ip1="147.32.84.140", 
	          attack="Sogou")

setting_label(ip1="147.32.84.130", 
	          attack="Murlo")

setting_label(ip1="147.32.84.160", 
	          attack="Virut")

setting_label(ip1="10.0.2.15", 
	          attack="IRCbot and black hole1")

setting_label(ip1="192.168.106.141", 
	          attack="Black hole 2")

setting_label(ip1="192.168.106.131", 
	          attack="Black hole 3")

setting_label(ip1="172.16.253.130", 
	          attack="TBot")

setting_label(ip1="172.16.253.131", 
	          attack="TBot")

setting_label(ip1="172.16.253.129", 
	          attack="TBot")

setting_label(ip1="172.16.253.240", 
	          attack="TBot")

setting_label(ip1="74.78.117.238", ip2="158.65.110.24", 
	          attack="Weasel")

setting_label(ip1="192.168.3.35", 
	          attack="Zeus")

setting_label(ip1="192.168.3.25", 
	          attack="Zeus")

setting_label(ip1="192.168.3.65", 
	          attack="Zeus")

setting_label(ip1="172.29.0.116", 
	          attack="Zeus")

setting_label(ip1="172.29.0.109", 
	          attack="Osx_trojan")

setting_label(ip1="172.16.253.132", 
	          attack="Zero access")

setting_label(ip1="192.168.248.165", 
	          attack="Zero access")

setting_label(ip1="10.37.130.4", 
	          attack="Smoke bot")

data['Label'] = np.where(data['Attack'] == "Normal", 0, 1)

save_data(sys.argv[1])
###################################################################