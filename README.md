# Steering-Angle-Prediction
Neural network which predicts the steering angle (given a road image) for a self-driving car application which is inspired by Udacity’s Behavior Cloning Project

Important Note : Keep steering folder (having img0) and both the python scripts in the same directory.

The main file containing code is assign3.py and it creates different neural network models for different experiments listed in l3.pdf. Whereas the file assign3_extra_layer.py is for adding one more hidden layer(for detailed explanation please refer to the report).  

Two optimisation techniques were tried to improve the accuracy of current model : adam optimisation and adding extra layer.
		 
How to run the program:
	
	With extra hidden layer:
		python assign3_extra_layer.py
		
	For other experiments :
		python assign3.py X
		
where X is choosen according to below given experiments :

X= 1 for A plot of sum of squares error on the training and validation set as a function of training iterations (for 5000 epochs) with a learning rate of 0.01. (no dropout,minibatch size of 64).

X= 2 for A plot of sum of squares error on the training and validation set as a function of training iterations (for 1000 epochs) with a fixed learning rate of 0.01 and minibatch size – 32.

X=3 for A plot of sum of squares error on the training and validation set as a function of training iterations (for 1000 epochs) with a fixed learning rate of 0.01 and minibatch size – 64.

X=4 for A plot of sum of squares error on the training and validation set as a function of training iterations (for 1000 epochs) with a fixed learning rate of 0.01 and minibatch size – 128.

X=5 for A plot of sum of squares error on the training and validation set as a function of training iterations (for 1000 epochs) with a learning rate of 0.001, and dropout probability of 0.5 for the first, second and third layers.

X=6 for A plot of sum of squares error on the training and validation set as a function of training iterations (for 1000 epochs) with learning rate 0.001(no drop out, minibatch size – 64)

X=7 for A plot of sum of squares error on the training and validation set as a function of training iterations (for 1000 epochs) with learning rate 0.005(no drop out, minibatch size – 64)

X=8 for A plot of sum of squares error on the training and validation set as a function of training iterations (for 1000 epochs) with a fixed learning rate of 0.001 and minibatch size – 128 and adam optimisation

