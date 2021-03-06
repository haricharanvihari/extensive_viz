## Objective
#### 1.	To use Session 5 code and run the model for 25 epochs 
#### To implement below combinations
  •	without L1/L2 with BN
  •	without L1/L2 with GBN
  •	with L1 with BN
  •	with L1 with GBN
  •	with L2 with BN
  •	with L2 with GBN
  •	with L1 and L2 with BN
  •	with L1 and L2 with GBN

#### 2.	To Plot the accuracy changes and validation losses in individual graphs for all the models.

#### 3.	To Plot 25 misclassified images for "without L1/L2 with BN" AND "without L1/L2 with GBN" model

## Results 

#### Loss change graph for all the above 8 jobs
![alt text](https://github.com/haricharanvihari/extensive_viz/blob/master/S6_DNN/Images/Loss%20Curves.png)
<!-- <img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S6_DNN/Images/Loss%20Curves.png" width="650" title="Loss Change Graph"> -->

#### Validation Accuracy change graph for all 8 jobs
![alt text](https://github.com/haricharanvihari/extensive_viz/blob/master/S6_DNN/Images/Accuracy%20Curves.png)

#### Misclassified Images Without L1/L2 BN
![alt text](https://github.com/haricharanvihari/extensive_viz/blob/master/S6_DNN/Images/Without_L1L2_BN.png)

#### Misclassified Images Without L1/L2 GBN
![alt text](https://github.com/haricharanvihari/extensive_viz/blob/master/S6_DNN/Images/Without_L1l2_GBN.png)

#### Observation of L1 and L2's performance in the regularization
Accuracy is slightly higher when L2 is used and compared to using L1. In Loss Change graph with usage of L2 the loss is slightly less, compared to the run when L2 is not used. 
However with usage of L1/L2 some changes in accuracy is also seen.

