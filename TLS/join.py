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

CC_w_As = pd.merge_asof (
						pd.read_csv('CAIA_Consensus.csv').fillna(0).sort_values(by='flowStartMilliseconds'),
						pd.read_csv('AGM_s.csv').fillna(0).sort_values(by='flowStartMilliseconds'),
						on='flowStartMilliseconds',
						direction='backward',
						by='sourceIPAddress'
						)
CCAs_w_Ad = pd.merge_asof (
						CC_w_As,
						pd.read_csv('AGM_d.csv').fillna(0).sort_values(by='flowStartMilliseconds'),
						on='flowStartMilliseconds',
						direction='backward',
						by='destinationIPAddress'
						)

TA_w_CCAsd = pd.merge_asof (
							CCAs_w_Ad,
							pd.read_csv('TA_bidirectional.csv').fillna(0).sort_values(by='flowStartMilliseconds'),
							on='flowStartMilliseconds',
							direction='backward',
							by=['sourceIPAddress','destinationIPAddress']
							)

TA_w_CCAsd.to_csv('mega_unlabeled.csv', index=False)