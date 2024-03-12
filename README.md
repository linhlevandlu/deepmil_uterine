# Deep-Attention Multiple Instance learning for classification and progonostication of Uterus cancer

This repository contains the source code of Deep-Attention Multiple Instance Learning (Deep-Attention MIL)
for classification and prognostication of Gynaecologic Smooth Muscle Tumors (Uterus cancer)

## Overview of the framework to work with Whole Slide Images (WSIs) and proposed system
To work with WSIs, a framework consisting of three phases has been used: (1) getting tiles (patches) from the WSIs of patients; (2) extracting the features from the extracted tiles by using an encoder, i.e., pre-trained <a href="https://arxiv.org/abs/1512.03385" target="blank">ResNet50</a>; (3) deep learning model to perform the goals of the study. 

![overview](docs/framework.png)

The proposed system in this study was stayed at the third phase of the <a href = "https://inria.hal.science/hal-04235077/document" target = "blank">framework </a> to classify and predict the survival of uterine cancer patients.

![overview](docs/model.png)

## The source code:
1. Classification task: GSMT_src_classification folder.
	a. main_cv.py: The main file to run the cross-validation test
	b. model_pl.py: The model
	c. bergonie_dataloader_survival_wsi.py: The file to load and create the Dataset, Dataloader
	
2. Prognostication task: GSMT_src_survival folder.