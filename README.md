# EEG classfier

## Overview

The repo is an exploration of ML methods to classify EEG signals with siezures present/absent.

Dataset - CHB-MIT Scalp EEG Database
+ 23 pediatric epilespy patients (1 subject has two datasets)
+ Contious EEG recordings 1-4 hours long
+ 198 labeled seizures
+ 256 hz @ 16 bit resolution
+ location: https://physionet.org/content/chbmit/1.0.0/


## Result from artifically balanced dataset
This is the result of a Random Forest model trained on a balanced data set, where siezures and non seizure epochs equally represented. Various statstics (peak-to-peak,variance,mean,zero-crossings) from each channel served as features. Just a quick sanity test to get up and running.

![Confusion matrix](img/balanced_hist.png)

**Figure 2.** Histogram of raw data from each class. Qualitative differences suggest a model should be able to classify.

![Confusion matrix](img/balanced_cm.png)

**Figure 2.** Confusion matrix. The model performs quite well on the balanced data.
