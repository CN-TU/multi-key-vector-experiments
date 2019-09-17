# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2019, Institute of Telecommunications, TU Wien
#
# Name        : train_test_split.py
# Description : split the data into training and testing sets
# Author      : Fares Meghdouri
#
#******************************************************************************

seed      = 2019
test_size = 0.3
#############################################################################

import pandas as pd
from sklearn.model_selection import train_test_split

# read the data
print("Reading the data")
data = pd.read_csv("mega_ohe.csv").fillna(0)

# drop unecessary features
print("Drop key features")
X = data.drop(["flowStartMilliseconds", "sourceIPAddress","destinationIPAddress", "mode(destinationIPAddress)", "mode(sourceTransportPort)_x", "mode(destinationTransportPort)_x", "mode(sourceIPAddress)", "mode(sourceTransportPort)_y", "mode(destinationTransportPort)_y", "__NTAFlowID", "__NTAPorts", "Label"], axis=1)

y = data["Label"]
s = data["Attack"]

# split
print("Start spliting")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=s)

# concatenate teh data and the labels
testing = pd.concat([X_test, y_test], axis=1)
training = pd.concat([X_train, y_train], axis=1)

# export
print("Exporting")
testing.to_csv('mega_ipsec_testing.csv', index=False)
training.to_csv('mega_ipsec_training.csv', index=False)
