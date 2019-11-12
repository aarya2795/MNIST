﻿<h1>Fashion MNIST</h1>
<h2> Dataset </h2>
<ul type = "disc">
<li> Fashion MNIST dataset within Keras/Tensorflow </li>
<li> Originally, this dataset had 60,000 training examples of images from Zalando’s in 10 categories </li>
<li> The categories being- 
	<ol type="1">
	<li>T-shirt/top</li>
	<li>Trouser</li>
	<li>Pullover</li>
	<li>Dress</li>
	<li>Coat</li>
	<li>Sandal</li>
	<li>Shirt</li>
	<li>Sneaker</li>
	<li>Bag</li>
	<li>Ankle boot</li>
	</ol>
<li><b>Modification</b> - limited the dataset to 20,000 training images</li>
<li> There are 10,000 test images </li>
</ul>
<h2>Model</h2>
<ol type="1">
<li>Initially, imported Dense, Dropout, Conv2D, MaxPooling2D</li>
<li>The training and test images are reshaped into 4 dimensions. So, here added a Convolutional layer and 1 Max pooling layer</li>
<li>After the convolution layer, the dimensions of the output becomes 26 X 26 X 32 and after applying max pool the dimensions become 13 X 13 X 32 which is equal to 5408; hence the value mentioned in the dense function</li>
<li>Added a 0.2 Dropout</li>
<li> Model Summary</li>	
Model: "sequential"<br>
_______________________________________________________________________________________<br>
Layer (type)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;Output Shape&emsp;&emsp;&emsp;&emsp;Param #   <br>
====================================================<br>
conv2d (Conv2D)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(None, 26, 26, 32)&emsp;&emsp;&emsp;&ensp;320       <br>
_______________________________________________________________________________________<br>
max_pooling2d(MaxPooling2D)&emsp;&emsp;(None, 13, 13, 32)&emsp;&emsp;&emsp;&ensp;&ensp;0<br>
_______________________________________________________________________________________<br>
dropout (Dropout)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(None, 13, 13, 32)&emsp;&emsp;&emsp;&emsp;0<br>
_______________________________________________________________________________________<br>
flatten (Flatten)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(None, 5408)&emsp;&emsp;&emsp;&emsp;&emsp;0<br>
_______________________________________________________________________________________<br>
dense (Dense)&emsp;&emsp;&emsp;&emsp;&emsp;&ensp;&emsp;&emsp;&emsp;&emsp;&emsp;(None, 5408)&emsp;&emsp;&emsp;&ensp;29251872  <br>
_______________________________________________________________________________________<br>
dense_1 (Dense)&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;(None,10)&emsp;&emsp;&emsp;&emsp;&emsp;54090     <br>
====================================================<br>
Total params: 29,306,282<br>
Trainable params: 29,306,282<br>
Non-trainable params: 0<br>
_______________________________________________________________________________________
</ol>
<h2> Accuracy </h2>
<li>The accuracy(performance) achieved  is 89.94%</li>
