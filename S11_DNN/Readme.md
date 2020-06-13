
### Super-Convergence
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/S11.png" width="550" title="Que"> 

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
####  o	Best Accuracy Train - 94.26% Test - 90.46% (Epoch 5)
####  o	Target Accuracy : 88.51%

### 2.	Train Test Acc Curve
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/train_test_acc_change.png" width="550" title="Train_test">

### 3.	Loss Acc Change Curve
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/images/loss_acc_change.png" width="550" title="Train_test">

### Gradcam Images
<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/Images/gradcam_incorrect_0_dog.png" width="550" title="car">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/Images/gradcam_incorrect_1_plane.png" width="550" title="truck">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/Images/gradcam_incorrect_2_cat.png" width="550" title="cat">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/Images/gradcam_incorrect_3_ship.png" width="550" title="dog">

<img src="https://github.com/haricharanvihari/extensive_viz/blob/master/S11_DNN/Images/gradcam_incorrect_4_dog.png" width="550" title="dog">

