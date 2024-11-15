# Heart Attack Survival Prediction

<p align="center">
<img src="assets/Heart-attack-1-1000x600.jpg" width="600" height="300">
</p>

This repository shows a predictive model for a one year survival after a heart attack from the Echocardiogram dataset from UC Irvine :shipit:

### Table of Contents
- [Required Tools and Packages](#Required-Tools-and-Packages)
- [Data Analysis](#Data-Analysis)
  - [HeartAttack Data Set](#Echocardiogram-Data-Set)
    - [Dataset_Information](#Dataset-Information) 
  - [Data Preprocessing](#Data-Preprocessing)
  - [Data Exploration](#Data-Exploration)
- [Data Model](#Data-Model)
  - [Running the classifier](#Running-the-classifier)

<a name="Required-Tools-and-Packages"></a>
### Required Tools and Packages

* python (used version 3.10.12)
  - python libraries
    - sys
    - numpy
    - pandas
    - matplotlib
    - seaborn
    - sklearn
    - xgboost

<a name="Dataset-Analysis"></a>
### Data Analysis

<a name="Echocardiogram-Data-Set"></a>
##### Echocardiogram Data Set
All the patients suffered heart attacks at some point in the past. Some are still alive and some are not.  The survival and still-alive variables, when taken together, indicate whether a patient survived for at least one year following the heart attack.  

The problem addressed by past researchers was to predict from the other variables whether or not the patient will survive at least one year.  The most difficult part of this problem is correctly predicting that the patient will NOT survive.  (Part of the difficulty seems to be the size of the data set.)

<a name="Dataset-Information"></a>
##### Variables
The following variables are taken into consideration into the survival prediction:

   1. survival -- the number of months patient survived (has survived, if patient is still alive).  Because all the patients had their heart attacks at different times, it is possible that some patients have survived less than one year but they are still alive.  Check the second variable to confirm this.  Such patients cannot be used for the prediction task mentioned above.
   2. still-alive -- a binary variable.  0=dead at end of survival period, 1 means still alive 
   3. age-at-heart-attack -- age in years when heart attack occurred
   4. pericardial-effusion -- binary. Pericardial effusion is fluid around the heart.  0=no fluid, 1=fluid
   5. fractional-shortening -- a measure of contracility around the heart lower numbers are increasingly abnormal
   6. epss -- E-point septal separation, another measure of contractility.  Larger numbers are increasingly abnormal.
   7. lvdd -- left ventricular end-diastolic dimension.  This is a measure of the size of the heart at end-diastole. Large hearts tend to be sick hearts.
   8. wall-motion-score -- a measure of how the segments of the left ventricle are moving
   9. wall-motion-index -- equals wall-motion-score divided by number of segments seen.  Usually 12-13 segments are seen in an echocardiogram.  Use this variable INSTEAD of the wall motion score.
   10. mult -- a derivate var which can be ignored
   11. name -- the name of the patient (I have replaced them with "name")
   12. group -- meaningless, ignore it
   13. alive-at-1 -- Boolean-valued. Derived from the first two attributes. 0 means patient was either dead after 1 year or had been followed for less than 1 year.  1 means patient was alive at 1 year.

<a name="Data-Preprocessing"></a>
##### Data Preprocessing

In order to study the relationship between the available variables, some preprocessing steps were taken. These steps can be found in the script `scripts/dataPreprocessing.py`, and the can be summarized as follows:

1. Removed rows where survival data was missing.
2. Changed the variable types to the corresponging defined `dtypes`.

The ouput from running `python scripts/exploratory.py` will generate the input to the classifier (see [Running the classifier](#Running-the-classifier)). 

<a name="Data-Exploration"></a>
##### Data Exploration

At first glance the survival after 1 year showed a positive correlation with all variables (not considering survival and still alive) except with the fractional-shortening where we see a negative correlation.

<p align="center">
<img src="assets/corr_matrix.png">
</p>

The overall distributions of the variables portray different distributions according to the alive-at-1 label.

<p align="center">
<img src="assets/dist_matrix.png">
</p>

> [!NOTE]
> All of the below described results can be reproduced by using the `scripts/exploratory.py` file.

<a name="Data-Model"></a>
### Data Model

This repository provides a simple classifier that allows the prediction of survival for a certain person given the described variabled (see [Dataset_Information](#Dataset-Information))

<a name="Running-the-classifier"></a>
##### Running the classifier

The current version with the classifier details can be found at `scripts/classifier.py`. To run the classifier and reproduce the results simply run:

`python classifier.py [analysis_arguments]`

Where analysis_arguments can take the following arguments:

1. Data file to be used.
2. Variables to be used in the prediction. Currently need to choose from the described variabled (see [Dataset_Information](#Dataset-Information) ))
3. Classifier Architecture to be used:
  * `Logistic Regressor`
  * `Random Forest`
  * `XGBoost`
  * `SVM`
  * `KNN`

Running the command `python classifier.py input.csv all log_reg` gives an accuracy of 0.96 (macro avg f1-score of 0.95) (see below for results).

<p align="center">
<img src="assets/4_plots_grid.png">
</p>

Together our exploratory results and simple classifier show the capability of predicting the survival probability of a person that has had a heart attack 🎆
