import json # we need to use the JSON package to load the data, since the data is stored in JSON format
import numpy as np
import collections
from numpy.linalg import inv
import time
import math as m
with open("proj1_data.json") as fp:
    data = json.load(fp)
    
from numpy  import array




# Now the data is loaded.
# It a list of data points, where each datapoint is a dictionary with the following attributes:
# popularity_score : a popularity score for this comment (based on the number of upvotes) (type: float)
# children : the number of replies to this comment (type: int)
# text : the text of this comment (type: string)
# controversiality : a score for how "controversial" this comment is (automatically computed by Reddit)
# is_root : if True, then this comment is a direct reply to a post; if False, this is a direct reply to another comment


'''
*    Title: Negative words.txt/ list used for sentiment 
*    Author:  Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews." 
*    Date: ,Aug 22-25, 2004, Seattle, 
*    Code version: Discovery and Data Mining (KDD-2004)
*    Availability: https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html?fbclid=IwAR1hI4Rgqf1dxJAnOw8WzpAN_bM2xkUEC5b-h9UeZO4LDBvEQ5R8hgPEI2c#lexicon
*
'''

filepath = 'negative-words.txt'
negative = []  
with open(filepath,encoding = "ISO-8859-1") as fp:
	line = fp.readline()
	while line:
		negative.append(line.strip())
		line = fp.readline()

    
 

#Part 1: Data processing --------------------------------------------------------------------------------------------------------------------

#Helper method to get top 160 words
def process_words_into_list(data):
	x = 0 
	file= open('words.txt',"a")
	file.write("----------------Start of call to retrieve top 160 word--------------------")
	file.write('\n')
	#this first part is for the word processing you can ignore it for part 1
	word_matrix =[]
	for data_point in data[:10000]:
		text = data_point["text"].split()
		#here I split the data based on the whitespaces and then I lowercase each word and append it to the word_matrix list
		for word in text:
			word_matrix.append(word.lower())		
	#here I'm getting the top 160 words by using a counter on each unique word
	feature_words = []
	counter = collections.Counter(word_matrix)
	for word, number in counter.most_common():
		x = x + 1
		if(x == 161 ):
			break
		else:
			feature_words.append(word)
			file.write(word)
			file.write('\n')
	#in the feature_words are all 160 words that will be used for our feature 
	file.write("--------------------End of call------------------------------")
	file.write('\n')	
	file.close()
	return(feature_words)
	


# Method to retrieve the feature matrix and out vector with no text
def retrieve_q1_matrices(data,dataset = 3):
	x = 0
	feature_matrix = []
	output_vector = []

	for data_point in data: # select the first data point in the dataset
		# Now we print all the information about this datapoint
		output_vector.append(data_point["popularity_score"])
		#convert is_root boolean into true and false
		is_root = 0
		if(data_point["is_root"]):
			is_root = 1
		data = [1,data_point["controversiality"], is_root, data_point["children"]]
		feature_matrix.append(data)
		#print(data_point["controversiality"])
		x = x + 1
	   # for info_name, info_value in data_point.items():
	    #    print(info_name + " : " + str(info_value))
	#print(feature_matrix)

	if dataset == 0:
			return np.array(feature_matrix)[10000:11000,:], np.array(output_vector).T[10000:11000]
	if dataset == 1:
			return np.array(feature_matrix)[11000:12001,:], np.array(output_vector).T[11000:12001]
	else:
			return np.array(feature_matrix)[0:10000,:], np.array(output_vector).T[:10000]

def word_feature_row(text, word_list):
	feature_row = []
	for word in word_list:
		if word in text:
			feature_row.append(1)
		else:
			feature_row.append(0)
	return (feature_row)

def retrieve_word_matrix(data, word_list):
	word_matrix = []

	for data_point in data:
		text = data_point["text"].split()
		#here I split the data based on the whitespaces and then I lowercase each word and append it to the word_matrix list
		filtered_text = []
		for word in text:
			filtered_text.append(word.lower())
		word_matrix.append(word_feature_row(filtered_text, word_list))

def retrieve_word_row(text, word_list):
		#here I split the data based on the whitespaces and then I lowercase each word and append it to the word_matrix list

	filtered_text = []
	for word in text:
		filtered_text.append(word.lower())
	return word_feature_row(filtered_text, word_list)

	
	
#Method to retrieve top 60 and full feature matrix	
def retrieve_full_feature_matrix(data, dataset = 3, is_top_60 = False):
	x = 0
	feature_matrix = []
	output_vector = []
	word_list =process_words_into_list(data)

	for data_point in data: # select the first data point in the dataset
		# Now we print all the information about this datapoint
		#convert is_root boolean into true and false
		output_vector.append(data_point["popularity_score"])
		is_root = 0
		if(data_point["is_root"]):
			is_root = 1
		data = [1,data_point["controversiality"], is_root, data_point["children"]]
		data2 = retrieve_word_row(data_point["text"].split(), word_list)
		feature_matrix.append(data + data2)
		#print(data_point["controversiality"])

		x = x + 1
	#print(x)
	if is_top_60:
		if dataset == 0:
			return np.array(feature_matrix)[10000:11000,0:64], np.array(output_vector).T[10000:11000]
		if dataset == 1:
			return np.array(feature_matrix)[11000:12001,0:64], np.array(output_vector).T[11000:12001]
		else:
			return np.array(feature_matrix)[0:10000,0:64], np.array(output_vector).T[:10000]
	else:

		if dataset == 0:
			return np.array(feature_matrix)[10000:11000,:], np.array(output_vector).T[10000:11000]
		if dataset == 1:
			return np.array(feature_matrix)[11000:12001,:], np.array(output_vector).T[11000:12001]
		else:
			return np.array(feature_matrix)[0:10000,:], np.array(output_vector).T[:10000]


# Method to get negativity score of a word for the additional feature
def getTextNegativityScore(x):
	z = 0 
	for word in negative:
		if(word in x):
			z = z + 1
	return m.log(z+1)

def get_column(data):
	s = []
	for data_point in data:
		s.append(getTextNegativityScore(data_point["text"]))
	print(s)



# Method to get matrix with additonal two features
def retrieve_extra_feature_matrix(data, dataset = 3):
	x = 0
	feature_matrix = []
	output_vector = []

	for data_point in data: # select the first data point in the dataset
		# Now we print all the information about this datapoint
		#convert is_root boolean into true and false
		output_vector.append(data_point["popularity_score"])
		count=len(data_point["text"].split())
		is_root = 0
		if(data_point["is_root"]):
			is_root = 1
		data = [1,data_point["controversiality"], is_root, data_point["children"],getTextNegativityScore(data_point["text"]),count]
		feature_matrix.append(data)
		#print(data_point["controversiality"])

		x = x + 1
	#print(x)

	if dataset == 0:
		return np.array(feature_matrix)[10000:11000,:], np.array(output_vector).T[10000:11000]
	if dataset == 1:
		return np.array(feature_matrix)[11000:12001,:], np.array(output_vector).T[11000:12001]
	else:
		return np.array(feature_matrix)[0:10000,:], np.array(output_vector).T[:10000]



#Part 2: closed form and gradient descent  weight computation -----------------------------------------------------------------------------

 
#Closed form weight computation
#x is the feature matrix (training data set)
#y is the output matrix  (output for training data)
def weight(x, y):
    t = time.time()
    x_prod_inv = np.linalg.pinv(np.matmul(x.T, x))
    x_prod_y = np.matmul(x.T, y)
    return np.matmul( x_prod_inv, x_prod_y),t




#Gradient descent weight computation with a decaying rate 
#x is the feature matrix (training data set)
#y is the output matrix  (output for training data)
#w weight
#b controls the speed of the decay 
#n is the initial learning rate
#alpha is the decaying learning rate 
#eps is the precision hyperparameter 
def gradient_descent(x,y, w, b, n,eps):
   t = time.time()
   XTX=np.matmul(x.T, x) # Precompute to make faster
   XTY=np.matmul(x.T, y) 
   cur_w=w
   while True:
     alpha= n/(1+b)
     prev_w= cur_w
     XTX_prev_w=np.matmul(XTX, prev_w)
     cur_w= np.subtract(prev_w,2*alpha*np.subtract( XTX_prev_w, XTY))
     # check if it failed condition
     threshold = np.subtract(cur_w,prev_w)
     b = b + 1
     if np.linalg.norm(threshold) < eps:
          break
   return prev_w,t
   
#Used to get error rate -duplicated due to timing accuracy issues for the first one
def gradient_descent_debug(x,y, w, b, n,eps):
   t = time.time()
   XTX=np.matmul(x.T, x) # To make computations faster
   XTY=np.matmul(x.T, y)
   i=0
   cur_w=w
   while True:
     alpha= n/(1+b)
     prev_w= cur_w
     XTX_prev_w=np.matmul(XTX, prev_w)
     cur_w= np.subtract(prev_w,2*alpha*np.subtract( XTX_prev_w, XTY))
     # check if it failed condition
     b = b + 1
     i=i+1
     threshold = np.subtract(cur_w,prev_w)
     error=np.linalg.norm(threshold)
     print ("The error for the %d iteration is %f" % (i, error))
     if error < eps:
          print ("The error for the %d iteration is smaller than eps  %f  " %(i, eps))
          print ("End gradient descent")
          break
   return prev_w,t


#Least square error calculation
#Y_prime is the predicted output 
#Y is actual output 
def lse (Y_prime,Y):
   result= np.power(np.subtract(Y_prime, Y), 2)
   return result


#Part 3: Testing and perfomance computations  ------------------------------------------------------------------------------------------------



#Slicing features into training, validation and testing sets for matrix with no text features 

trainX,trainY = retrieve_q1_matrices(data)
validX,validY = retrieve_q1_matrices(data,0)
testX,testY = retrieve_q1_matrices(data,1)
print ("Slicing features into training, validation and testing sets for matrix with no text features... ")

#Slicing features into training, validation and testing sets for matrix with top 60 text features

trainX_60,trainY_60 = retrieve_full_feature_matrix(data,3,True)
validX_60,validY_60 = retrieve_full_feature_matrix(data,0,True)
testX_60,testY_60 = retrieve_full_feature_matrix(data,1,True)
print ("Slicing features into training, validation and testing sets for matrix top 60 text features...")


#Slicing features into training, validation and testing sets for matrix with full 160 text features

trainX_160,trainY_160 = retrieve_full_feature_matrix(data)
validX_160,validY_160 = retrieve_full_feature_matrix(data,0)
testX_160,testY_160 = retrieve_full_feature_matrix(data,1)
print ("Slicing features into training, validation and testing sets for matrix full 160 text features...")


#Slicing features into training, validation and testing sets for matrix with full 160 text features
trainX_add2,trainY_add2 = retrieve_extra_feature_matrix(data)
validX_add2,validY_add2 = retrieve_extra_feature_matrix(data,0)
testX_add2,testY_add2 = retrieve_extra_feature_matrix(data,1)
print ("Slicing features into training, validation and testing sets for matrix with no text and 2 additional features...")

print ("..............................................Runtime .................................................... ")
#Closed form solution with no text feature
w_closed_form,t1 = weight(trainX, trainY)
Y_closed_form= np.matmul(trainX, w_closed_form)
end_t1= time.time()
Y_closed_form_valid= np.matmul(validX, w_closed_form)
Y_closed_form_test= np.matmul(testX, w_closed_form)
print ("Closed form solution  with no text feature run time : ", end_t1-t1)

#Gradient descent solution no text feature
w_gradient_descent,t2 = gradient_descent(trainX,trainY, np.zeros((4)), 0, 0.0005,0.001)
Y_gradient_descent = np.matmul(trainX, w_gradient_descent)
end_t2= time.time()
Y_gradient_descent_valid = np.matmul(validX, w_gradient_descent)
Y_gradient_descent_test = np.matmul(testX, w_gradient_descent)
print ("Gradient with no text feature run time : ", end_t2-t2)


#Closed form solution with top 60 words
w_closed_form_60,t3 = weight(trainX_60, trainY_60)
Y_closed_form_60= np.matmul(trainX_60, w_closed_form_60)
end_t3= time.time()
Y_closed_form_valid_60= np.matmul(validX_60, w_closed_form_60)
Y_closed_form_test_60= np.matmul(testX_60, w_closed_form_60)
print ("Closed form solution  with top 60 words run time : ", end_t3-t3)

#Gradient descent solution with top 60 words

w_gradient_descent_60,t4 = gradient_descent(trainX_60,trainY_60, np.zeros((64)), 0, 0.0005,0.001)
Y_gradient_descent_60 = np.matmul(trainX_60, w_gradient_descent_60 )
end_t4= time.time()
Y_gradient_descent_valid_60 = np.matmul(validX_60, w_gradient_descent_60)
Y_gradient_descent_test_60 = np.matmul(testX_60, w_gradient_descent_60)
print ("Gradient descent with top 60 words run time : ", end_t4-t4)


#Closed form solution with top 160 words
w_closed_form_160,t5 = weight(trainX_160, trainY_160)
Y_closed_form_160= np.matmul(trainX_160, w_closed_form_160)
end_t5= time.time()
Y_closed_form_valid_160= np.matmul(validX_160, w_closed_form_160)
Y_closed_form_test_160= np.matmul(testX_160, w_closed_form_160)
print ("Closed form solution with top 160 words run time : ", end_t5-t5)

#Gradient descent solution with top 160 words
w_gradient_descent_160,t6= gradient_descent(trainX_160,trainY_160, np.zeros((164)), 0, 0.0005,0.005)
Y_gradient_descent_160= np.matmul(trainX_160, w_gradient_descent_160)
end_t6= time.time()
Y_gradient_descent_valid_160 = np.matmul(validX_160, w_gradient_descent_160)
Y_gradient_descent_test_160 = np.matmul(testX_160, w_gradient_descent_160)
print ("Gradient descent with top 160 words run time : ", end_t6-t6)


#Closed form solution no text but two additional features 
w_closed_form_add2,t7 = weight(trainX_add2, trainY_add2)
Y_closed_form_add2= np.matmul(trainX_add2, w_closed_form_add2)
end_t7= time.time()
Y_closed_form_valid_add2= np.matmul(validX_add2, w_closed_form_add2)
Y_closed_form_test_add2= np.matmul(testX_add2, w_closed_form_add2)
print ("Closed form with additional features (text word count + negative words) time) : ", end_t7-t7)

#Used for decaying error rate demonstration
print ("......................................................................................................................... ")
print ("Decaying error rate demonstration with gradient descent top 60 text features : ")
w_gradient_descent_60e,t9 = gradient_descent_debug(trainX_60,trainY_60, np.zeros((64)), 0, 0.0005,0.001)



#RSS and MSE calculations 

#RSS for (closed form) Y prime sets with no text feature 
train_lse = lse(Y_closed_form, trainY)
valid_lse = lse(Y_closed_form_valid, validY)
test_lse = lse(Y_closed_form_test, testY)


#RSS for (gradient descent) Y prime set with no text feature 
gtrain_lse = lse(Y_gradient_descent, trainY)
gvalid_lse = lse(Y_gradient_descent_valid, validY)
gtest_lse = lse(Y_gradient_descent_test, testY)

#RSS  for (closed form) Y prime sets with top 60 words 
train_lse_60 = lse(Y_closed_form_60, trainY_60)
valid_lse_60 = lse(Y_closed_form_valid_60, validY_60)
test_lse_60 = lse(Y_closed_form_test_60, testY_60)


#RSS for (gradient descent) Y prime sets with top 60 words  
gtrain_lse_60 = lse(Y_gradient_descent_60, trainY_60)
gvalid_lse_60 = lse(Y_gradient_descent_valid_60, validY_60)
gtest_lse_60 = lse(Y_gradient_descent_test_60, testY_60)


#RSS  for (closed form) Y prime sets with full 160 words 
train_lse_160 = lse(Y_closed_form_160, trainY_160)
valid_lse_160 = lse(Y_closed_form_valid_160, validY_160)
test_lse_160 = lse(Y_closed_form_test_160, testY_160)

#RSS for (gradient descent) Y prime sets with full 160 words  
gtrain_lse_160 = lse(Y_gradient_descent_160, trainY_160)
gvalid_lse_160 = lse(Y_gradient_descent_valid_160, validY_160)
gtest_lse_160 = lse(Y_gradient_descent_test_160, testY_160)


#RSS  for (closed form) Y prime sets with no text and 2 additional features
train_lse_add2 = lse(Y_closed_form_add2, trainY_add2)
valid_lse_add2 = lse(Y_closed_form_valid_add2, validY_add2)
test_lse_add2 = lse(Y_closed_form_test_add2, testY_add2)



#MSE calculations
print ("...............................................MSE/perfomance Calculations ................................ ")
 
# MSE  for (closed form) Y prime with no text feature 
train_mse = train_lse.mean()
valid_mse = valid_lse.mean()
test_mse = test_lse.mean()
print ("Training set with no text feature MSE (closed form solution)  : ", train_mse)
print ("Validation set with no text feature MSE (closed form solution) :" ,valid_mse)
print ("Testing set with no text feature MSE (closed form solution) : ", test_mse)
print ("........................................................................................................... ")

# MSE  for (gradient descent) Y prime sets with no text feature 
gtrain_mse = gtrain_lse.mean()
gvalid_mse =gvalid_lse.mean() 
gtest_mse = gtest_lse.mean()
print ("Training set with no text feature MSE (gradient descent) : ", gtrain_mse)
print ("Validation set with no text feature MSE (gradient descent) : ", gvalid_mse)
print ("Testing set with no text feature MSE (gradient descent) : ", gtest_mse)
print (".......................................................................................................... ")

# MSE  for (closed form) Y prime sets with top 60 words 
train_mse_60 = train_lse_60.mean()
valid_mse_60 = valid_lse_60.mean()
test_mse_60 = test_lse_60.mean()
print ("Training set with top 60 words MSE (closed form solution) : ", train_mse_60)
print ("Validation set with top 60 words MSE (closed form solution) : ", valid_mse_60)
print ("Testing set with top 60 words MSE (closed form solution) : ", test_mse_60)
print ("........................................................................................................... ")

# MSE  for (gradient descent) Y prime set with top 60 words 
gtrain_mse_60 = gtrain_lse.mean()
gvalid_mse_60 = gvalid_lse.mean()
gtest_mse_60 = gtest_lse.mean()
print ("Training set with top 60 words MSE  (gradient descent) : ", gtrain_mse_60)
print ("Validation set with top 60 words MSE  (gradient descent) : ", gvalid_mse_60)
print ("Testing set with  top 60 words MSE  (gradient descent) : ", gtest_mse_60)
print (".......................................................................................................... ")

# MSE for (closed form) Y prime  sets with full 160 words 
train_mse_160 = train_lse_160.mean()
valid_mse_160 = valid_lse_160.mean()
test_mse_160 = test_lse_160.mean()
print ("Training set with full 160 words MSE (closed form solution) : ", train_mse_160)
print ("Validation set with full 160 words MSE (closed form solution) : ", valid_mse_160)
print ("Testing set with full words MSE (closed form solution) : ", test_mse_160)
print (".......................................................................................................... ")

# MSE for (gradient descent) Y prime sets with full 160 words 
gtrain_mse_160 = gtrain_lse_160.mean()
gvalid_mse_160 = gvalid_lse_160.mean()    
gtest_mse_160 = gtest_lse_160.mean()
print ("Training set with  full 160 words MSE  (gradient descent) : ", gtrain_mse_160)
print ("Validation set with full  160 words MSE  (gradient descent) : ", gvalid_mse_160)
print ("Testing set with with  full  160 words MSE  (gradient descent) : ", gtest_mse_160)
print (".......................................................................................................... ")

train_mse_add2 = train_lse_add2.mean()
valid_mse_add2 = valid_lse_add2.mean()
test_mse_add2= test_lse_add2.mean()
print ("Training set with no text and two additional features words MSE (closed form solution) : ", train_mse_add2)
print ("Validation set with no text and two additional features words (closed form solution) : ", valid_mse_add2)
print ("Testing set with no text and two additional features words (closed form solution) : ", test_mse_add2)

print (".........................Best perfoming model :top 60 words (closed form solution) ..... ...................... ")
print ("Closed form solution  with top 60 words run time : ", end_t3-t3)
print ("Training set with top 60 words MSE (closed form solution) : ", train_mse_60)
print ("Validation set with top 60 words MSE (closed form solution) : ", valid_mse_60)
print ("Testing set with top 60 words MSE (closed form solution) : ", test_mse_60)


print ("...........................................Additional features improvements ..... ............................... ")

print (".....................Features added word count + sentiment score of words..... ..................................... ")
print ("Improvement using two addtional features for training set (closed form solution no text)  : ", train_mse-train_mse_add2 )
print ("Improvement using two addtional features for validation set (closed form solution no text)  : ",valid_mse-valid_mse_add2)
print ("Improvement using two addtional features for testing set(closed form solution no text)  ", test_mse-test_mse_add2 )


print ("Done")



