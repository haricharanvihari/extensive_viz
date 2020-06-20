
### Super-Convergence
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/S11.png" width="750" title="Que"> 

### Results:
#### Arranging file structure

-----|models </br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|---  __init.py__ <br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|--- *Resnet* Files , resnetbase <br>
------|dataloader.py <br>
------|datamodel.py </br>
------|gradcam.py <br>
------|lrfinder.py  <br>
------|cycliclr.py  <br>
------|optimizer.py <br>
------|transformation.py <br>

### 1.	Cyclic Curve
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/CYCLIC_TRIANGLE.png" width="550" title="Cyclic">

####  o	LR Max - 0.07
####  o	Accuracy Train - 99.69% Test - 93.24%
####  o	Accuracy at Epoch 5 : 92.87%%

### 2.	Train Test Acc Curve
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/train_test_acc_change.png" width="550" title="Train_test">

### 3.	Loss Acc Change Curve
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/loss_acc_change.png" width="550" title="Train_test">

### First 5 Gradcam Images
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/gradcam_incorrect_0_cat.png" width="550" title="cat">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/gradcam_incorrect_1_ship.png" width="550" title="ship">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/gradcam_incorrect_2_dog.png" width="550" title="dog">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/gradcam_incorrect_3_cat.png" width="550" title="cat">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/gradcam_incorrect_4_dog.png" width="550" title="dog">

