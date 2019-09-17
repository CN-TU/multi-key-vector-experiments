# -*- coding: utf-8 -*-

#******************************************************************************
#
# Copyright (C) 2019, Institute of Telecommunications, TU Wien
#
# Name        : join.py
# Description : create the multi-key feature vector from sub vectors
# Author      : Fares Meghdouri
#
#******************************************************************************

import pandas as pd

TA_w_As = pd.merge_asof (
							pd.read_csv('TA_bidirectional.csv').fillna(0).sort_values(by='flowStartMilliseconds'),
							pd.read_csv('AGM_s.csv').fillna(0).sort_values(by='flowStartMilliseconds'),
							on='flowStartMilliseconds',
							direction='backward',
							by='sourceIPAddress'
							)

TAAs_w_Ad = pd.merge_asof (
						TA_w_As,
						pd.read_csv('AGM_d.csv').fillna(0).sort_values(by='flowStartMilliseconds'),
						on='flowStartMilliseconds',
						direction='backward',
						by='destinationIPAddress'
						)

TAAs_w_Ad.to_csv('mega_unlabeled.csv', index=False)
