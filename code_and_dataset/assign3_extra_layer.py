from PIL import Image
import random
import math
import numpy as np
from scipy.special import expit
import pickle
import os
import matplotlib.pyplot as plt

class NeuralNetwork:
	def __init__(self,layer_size,alpha):
    
		# initialise weights for layers
		self.W1 = np.linspace(-0.01,0.01,layer_size[0]*layer_size[1])
		self.W1 = np.reshape(self.W1, (-1, layer_size[1]))
		zero_row = np.zeros([1,layer_size[1]])
		self.W1 = np.vstack((zero_row,self.W1))


		self.W2 = np.linspace(-0.01,0.01,layer_size[1]*layer_size[2])
		self.W2 = np.reshape(self.W2, (-1, layer_size[2]))
		zero_row1 = np.zeros([1,layer_size[2]])
		self.W2 = np.vstack((zero_row1,self.W2))


		self.W3 = np.linspace(-0.01,0.01,layer_size[2]*layer_size[3])
		self.W3 = np.reshape(self.W3, (-1, layer_size[3]))
		zero_row2 = np.zeros([1,layer_size[3]])
		self.W3 = np.vstack((zero_row2,self.W3))
		
		self.W4 = np.linspace(-0.01,0.01,layer_size[3])
		self.W4 = np.reshape(self.W4, (-1, 1))
		zero_row3 = np.zeros([1,1])
		self.W4 = np.vstack((zero_row3,self.W4))
		
		self.dropout = 0											# no dropout
		self.alpha = alpha											# alpha is the learning rate 
		
	def neural_dropout(self,layer_size,dropout_prob):
	
		self.active_I1 = np.ones([1,layer_size[0]])				# matrix of zeros and ones for switching off I1 nodes during dropout
		self.active_I2 = np.ones([1,layer_size[1]])				# matrix of zeros and ones for switching off I2 nodes during dropout
		self.active_I3 = np.ones([1,layer_size[2]])				# matrix of zeros and ones for switching off I3 nodes during dropout
		self.active_I4 = np.ones([1,layer_size[3]])				# matrix of zeros and ones for switching off I4 nodes during dropout

		inactive_I1_nodes = int(layer_size[0] * dropout_prob)
		inactive_I1_indices = sorted(random.sample(range(0,layer_size[0]),inactive_I1_nodes))
		for i in inactive_I1_indices:
			self.active_I1[0,i] = 0

		inactive_I2_nodes = int(layer_size[1] * dropout_prob)
		inactive_I2_indices = sorted(random.sample(range(0,layer_size[1]),inactive_I2_nodes))
		for i in inactive_I2_indices:
			self.active_I2[0,i] = 0
	
		inactive_I3_nodes = int(layer_size[2] * dropout_prob)
		inactive_I3_indices = sorted(random.sample(range(0,layer_size[2]),inactive_I3_nodes))
		for i in inactive_I3_indices:
			self.active_I3[0,i] = 0
			
		inactive_I4_nodes = int(layer_size[3] * dropout_prob)
		inactive_I4_indices = sorted(random.sample(range(0,layer_size[3]),inactive_I4_nodes))
		for i in inactive_I4_indices:
			self.active_I4[0,i] = 0
			
		self.dropout = 1
		self.dropout_prob = dropout_prob
    	
	def neural_IO(self,X,Y,isTest):
    	self.X = X														# input
    	self.Y = Y														# output
    	self.isTest =  isTest											# test instance or not
	
	def forwardprop(self):
		# while training for dropout
		if(self.dropout==1 and self.isTest==0):
			self.I1 = self.X
			self.I1 = np.multiply(self.active_I1,self.I1)
			self.I1 = np.hstack((np.ones([len(self.I1),1]),self.I1))
			self.O1 = np.dot(self.I1,self.W1)								   # output of first layer
			self.O1 = expit(self.O1)
			self.O1 = np.multiply(self.active_I2,self.O1)
			
			self.I2 = self.O1
			self.I2 = np.hstack((np.ones([len(self.I2),1]),self.I2))
			self.O2 = np.dot(self.I2,self.W2)									# output of second layer
			self.O2 = expit(self.O2)
			self.O2 = np.multiply(self.active_I3,self.O2)
		
			self.I3 = self.O2
			self.I3 = np.hstack((np.ones([len(self.I3),1]),self.I3))
			self.O3 = np.dot(self.I3,self.W3)									# output of third layer
			self.O3 = expit(self.O3)
			self.O3 = np.multiply(self.active_I4,self.O3)
			
			self.I4 = self.O3
			self.I4 = np.hstack((np.ones([len(self.I4),1]),self.I4))
			self.O4 = np.dot(self.I4,self.W4)									# output of fourth layer
			
		# for all other cases
		else:
			self.I1 = self.X
			if(self.dropout==1 and self.isTest==1):
				self.I1 = self.I1 * (1-self.dropout_prob)
			self.I1 = np.hstack((np.ones([len(self.I1),1]),self.I1))
			self.O1 = np.dot(self.I1,self.W1)
			self.O1 = expit(self.O1)
			
			self.I2 = self.O1
			if(self.dropout==1 and self.isTest==1):
				self.I2 = self.I2 * (1-self.dropout_prob)
			self.I2 = np.hstack((np.ones([len(self.I2),1]),self.I2))
			self.O2 = np.dot(self.I2,self.W2)
			self.O2 = expit(self.O2)
			
			self.I3 = self.O2
			if(self.dropout==1 and self.isTest==1):
				self.I3 = self.I3 * (1-self.dropout_prob)
			self.I3 = np.hstack((np.ones([len(self.I3),1]),self.I3))
			self.O3 = np.dot(self.I3,self.W3)
			self.O3 = expit(self.O3)
			
			self.I4 = self.O3
			if(self.dropout==1 and self.isTest==1):
				self.I4 = self.I4 * (1-self.dropout_prob)
			self.I4 = np.hstack((np.ones([len(self.I4),1]),self.I4))
			self.O4 = np.dot(self.I4,self.W4)
			
	def backwardprop(self):

		# derivatives that flow backwards
		D4 = (self.O4-self.Y).transpose()
		D3 = np.multiply(np.dot(self.W4[1:,:],D4),np.multiply(self.O3,1-self.O3).transpose())
		D2 = np.multiply(np.dot(self.W3[1:,:],D3),np.multiply(self.O2,1-self.O2).transpose())
		D1 = np.multiply(np.dot(self.W2[1:,:],D2),np.multiply(self.O1,1-self.O1).transpose())

		# update weights
		self.W4 = self.W4 - self.alpha * ((np.dot(D4,self.I4)).transpose())
		self.W3 = self.W3 - self.alpha * ((np.dot(D3,self.I3)).transpose())
		self.W2 = self.W2 - self.alpha * ((np.dot(D2,self.I2)).transpose())
		self.W1 = self.W1 - self.alpha * ((np.dot(D1,self.I1)).transpose())
		
	def compute_err(self):
	
		self.forwardprop()
		D = np.subtract(self.O4,self.Y)
		err =  ((np.dot(D.transpose(),D))/len(self.Y))[0,0]			# mean square error is computed for reporting accuracy of model
		return err



def StandardizeTrainData(data):
	mean_list = []
	std_list = []													# standardise the training data
	for i in range(len(data[0])):
		mean = data[:,i].mean()
		std = data[:,i].std()
		mean_list.append(mean)
		std_list.append(std)
		data[:,i] = data[:,i] - mean
		if std!=0:
		    data[:,i] = data[:,i] / std
	return data,mean_list,std_list
    

def StandardizeTestData(data,mean_list,std_list):
	for i in range(len(data[0])):
		data[:,i] = data[:,i] - mean_list[i]						# standardise test data
		if std_list[i]!=0:
		    data[:,i] = data[:,i] / std_list[i]
	return data

# creates neural network model according to the given parameters
def CreateNNModel(layer_size,pkl_file,alpha,minibatch_size,dropout,totalIter,graphName,graphTitle):
	if os.path.exists(pkl_file):
		NNfile = open(pkl_file,"rb")
		SteeringNN = pickle.load(NNfile)
	
	else:
		global training_Input,training_Output,validation_Input,validation_Output
		TrainErr = []
		TestErr = []												# for plotting graph
		Iter = []
	
		SteeringNN = NeuralNetwork(layer_size,alpha)
		if(dropout!=0):
			SteeringNN.neural_dropout(layer_size,0)
		SteeringNN.neural_IO(training_Input,training_Output,0)
		train_err = SteeringNN.compute_err()
		print("Training error with random weights "+str(train_err))			# iteration 0 (uniform weights)
		SteeringNN.neural_IO(validation_Input,validation_Output,1)
		test_err = SteeringNN.compute_err()
		print("Testing error with random weights "+str(test_err))
		Iter.append(0)
		TrainErr.append(train_err)
		TestErr.append(test_err)

		for k in range(totalIter):
			temp = np.hstack((training_Input,training_Output))				# shuffle the training data
			np.random.shuffle(temp)
			training_Input = temp[:,:-1].reshape(len(temp),len(temp[0])-1)
			training_Output = temp[:,-1].reshape(len(temp),1)
			count=0
			while(count<num):
				if(count+minibatch_size<=num):
					I1 = training_Input[range(count,count+minibatch_size), ]		# create mini batches
					Y = training_Output[range(count,count+minibatch_size), ]
				else:
					I1 = training_Input[range(count,num), ]
					Y = training_Output[range(count,num), ]
		
				if(dropout!=0):
					SteeringNN.neural_dropout(layer_size,dropout)
				SteeringNN.neural_IO(I1,Y,0)
				SteeringNN.forwardprop()
				SteeringNN.backwardprop()
				count = count+minibatch_size
	
			SteeringNN.neural_IO(training_Input,training_Output,0)	
			train_err = SteeringNN.compute_err()
			print("Training error --- Iteration "+str(k+1)+" : "+str(train_err))
			SteeringNN.neural_IO(validation_Input,validation_Output,1)	
			test_err = SteeringNN.compute_err()
			print("Testing error --- Iteration "+str(k+1)+" : "+str(test_err))
			Iter.append(k+1)
			TrainErr.append(train_err)
			TestErr.append(test_err)
		
		NNfile = open(pkl_file,"wb") 											# save the model
		pickle.dump(SteeringNN,NNfile)
		NNfile.close()
		
		plt.plot(Iter,TrainErr,label="Training Error")							# plot graph
		plt.plot(Iter,TestErr,label = "Testing Error")
		plt.ylim([0, 0.5])
		plt.xlabel('Number of Iterations')
		plt.ylabel('Error')
		plt.title(graphTitle)
		plt.legend()
		plt.savefig(graphName)



data_file = list(open('steering/data.txt'))
total_num = len(data_file)                             					# total_num is total number of examples
num = int(0.8*total_num)
o = []
for l in data_file:
	data = l.split()
	o.append(float(data[1]))

input_list = [i for i in range(total_num)]
random.shuffle(input_list)
Input = np.zeros([total_num,1024])  									# input image is 32X32 , 1 extra term for bias
Output = np.zeros([total_num,1])

j=0
for i in input_list:
	img_name = 'steering/img_'+str(i)+'.jpg'
	img = Image.open(img_name).convert('L')  							# convert image to 8-bit grayscale
	WIDTH, HEIGHT = img.size
	data = list(img.getdata())
	k = 0
	for d in data:
		Input[j,k] = d
		k = k+1
	Output[j,0]=o[i]
	j=j+1


training_Input = Input[0:num,:]
validation_Input = Input[num:total_num,:]
training_Output = Output[0:num,:]
validation_Output = Output[num:total_num,:]

training_Input,mean_list,std_list = StandardizeTrainData(training_Input)
validation_Input = StandardizeTestData(validation_Input,mean_list,std_list)


CreateNNModel([1024,512,256,64],"trainedNeuralNetwork_extraLayer.pkl",0.01,64,0,1000,'graph9.png','1000 iterations with learning rate= 0.01 and minibatch size= 64')

