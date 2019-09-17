# Multi-Key Vector Experiments

This repository contains scripts used to carry out experiments shown in "Multi-Key Flows for Attack Detection in Encrypted
Communications"

# Structure
```console
.
├── IPsec                                 # IPsec version scripts
│   ├── FlowSpecifications                ## json specification files used with Go-flows (1)
│   │   ├── AGM_d.json                    ### AGM aggregation by destination host
│   │   ├── AGM_s.json                    ### AGM aggregation by source host
│   │   └── TA.json                       ### Time Activity
│   ├── join.py                           ## Join feature vectors and construct the IPsec version (2)
│   ├── ML_paper.py                       ## Parameters tunning and classification based on RF
│   └── train_test_split                  ## Split the dataset into training and testing sets
├── TLS                                   # TLS version scripts
│   ├── FlowSpecifications                ## json specification files used with Go-flows
│   │   ├── AGM_d.json                    ### AGM aggregation by destination host
│   │   ├── AGM_s.json                    ### AGM aggregation by source host
│   │   ├── CAIA_Consensus.json           ### CAIA and Consensus
│   │   └── TA.json                       ### Time Activity
│   ├── join.py                           ## Join feature vectors and construct the TLS version
│   ├── ML_paper.py                       ## Parameters tunning and classification based on RF
│   └── train_test_split                  ## Split the dataset into training and testing sets
├── Labeling                              # Directory that contains labeling scripts
│   └── ...
├── freq_tables                           # Construct frequency tables for all feaures and perform OHE
├── LICENSE                               # License file
└── README.md                             # This file
```

# Requirements

* Python 3
	* Pandas
	* Numpy
	* sklearn
	* scipy
	* evolutionary_search

* [Go-flows](https://github.com/CN-TU/go-flows)

# Steps for reproducibility

#### Extraction ####
The first step is converting the PCAP files of each dataset into csv files containing network flows. To this end we use GO-flows in addition to the 'Flow Specification' files (1).
For each specification we extract flows using the following command:

```console
meghdouri@TUWien:~$ go-flows run features {flowSpec.json} export csv {outputFile.csv} source libpcap {sourcePCAP.pcap}
```

We repeat this step for both TLS and IPsec variants and for all specifications.

#### Construction ####
The `join.py` (2) script is used to group the previously extracted feature vectors together. Please make sure to change the file names inside the script to your personalized ones and run it without any arguments.

```console
meghdouri@TUWien:~$ python join.py
```

#### Labeling ####
To label the data, you can either use your own scripts or use the scripts provided under `{source}/Labeling/X_labeling.py`. The script needs only to be run with the correct source files (input: raw csv data, output: labeled csv data).
Note that if you want to use your own scripts, two columns are produced in this step: `Attack` and `Label`

#### One Hot Encoding (not mandatory) ####
`freq_tables.py` alows to extract frequency tables for statistical analysis and also OHE.
The script takes the raw labeled data and converts all features that the user chooses into binary dummies. For further instructions on how to use the script run `python freq_tables.py`.

#### Split ####
The `train_test_split.py` allows to both delete irrelevant features (such as IP addresses) and split the data into 70% training and 30% testing (the proportions can be set in the script). The input should be the labeled data and the output is wo files representing both a test and a training sets.

#### Analysis ####
The last step is the analysis step where our prepared data is fed into ML.

The script `ML_paper.py` contains a complete analysis framework for preprocessing, tuning, training and predicting classes based on the two files (training and testing) provided.
The script is run with the correct input files names and will output the following:

* `feature_importance_without_tuning_.csv`             
* `feature_importance_without_tuning_after_pca.csv`
* `feature_selection_with_tuning_.csv`
* `feature_selection_with_tuning_after_pca.csv`
* `pca_results.csv`
* `RF_classification_report.txt`
* `RF_DT_best_parameters.txt`
* `testing_performance.csv`
* `training_performance.csv`

The purpose of each file is described by its name, the last two files contain original labels and predicted labels in addition to the attack name for both test and training sets.

Moreover, more configurations that were not needed in the paper can be set inside (this page will be updated with further instructions)

# Contact
For further information or questions, please contact: `fares.meghdouri@tuwien.ac.at`