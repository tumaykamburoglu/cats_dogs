import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob2
import random

def load_images(path,num_px,batch):
    X = np.empty(((3*num_px*num_px),0))
    Y = np.zeros((2,batch))
    i = 0
    j = 0
    for pict in glob2.glob(path+"/cats/*.jpg"):
        
        image = np.array(Image.open(pict).resize((num_px, num_px)))
        image = image / 255.
        image = image.reshape((1, num_px * num_px * 3)).T
        X =np.append(X,image,axis=1)
        Y[0,i] = 1
        i = i + 1 
        if i == batch/2:
            break
    for pict in glob2.glob(path+"/dogs/*.jpg"):
        
        image = np.array(Image.open(pict).resize((num_px, num_px)))
        image = image / 255.
        image = image.reshape((1, num_px * num_px * 3)).T
        
        X =np.append(X,image,axis=1)
        Y[1,i+j] = 1
        j = j + 1
        if i+j == batch:
            break
    return X,Y
def relu(Z):
    A = np.maximum(0,Z)
    cache = Z 
    return A, cache

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    cache = Z
    return A, cache
def tanh(Z):
     A = np.tanh(Z)
     cache = Z
     return A,cache
def tanh_back(dA,cache):
    Z = cache
    s = np.tanh(Z)
    dZ = dA*(1-np.power(s,2))
    return dZ

def relu_back(dA,cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0   
    return dZ
def sigmoid_back(dA, cache):
    Z = cache   
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s) 
    return dZ
def random_init_parameters(layer_ns):
    parameters = {}
    L = len(layer_ns)
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(layer_ns[l],layer_ns[l-1])*0.005
        parameters["b"+str(l)] = np.zeros((layer_ns[l],1))
    return parameters



def linear_forward(A,W,b):
    Z = np.dot(W,A)+b
    cache = (A,W,b)
    return Z,cache

def linear_activation_forward(A_prev,W,b,type):
    if type == "relu":
        Z , linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = relu(Z)
    elif type=="sigmoid":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = sigmoid(Z)
    elif type=="tanh":
        Z, linear_cache = linear_forward(A_prev,W,b)
        A, activation_cache = tanh(Z)
    cache = (linear_cache, activation_cache)

    return A,cache

def forward_propagation(X,parameters):
    caches = []
    A = X
    L = int(len(parameters)/2)
    for l in range(1,L):
        A_prev = A
        A,cache = linear_activation_forward(A_prev,parameters["W"+str(l)],parameters["b"+str(l)],"tanh")
        caches.append(cache)
    AL,cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)
    return AL,caches

def calculate_cost(AL,Y):
    m = Y.shape[1]
    cost = (np.dot(Y,np.log(AL).T) + np.dot((1-Y),np.log(1-AL).T))/-m
    cost = np.diag(np.squeeze(cost))
    return cost

def linear_backward(dZ,cache):
    A_prev,W,b = cache
    m = A_prev.shape[1]
    dW = np.dot(dZ,A_prev.T)/m
    db = np.sum(dZ,axis=1,keepdims=True)/m
    dA_prev =np.dot(W.T,dZ)
    return dA_prev,dW,db

def linear_activation_backward(dA,cache,type):
    linear_cache, activation_cache = cache   
    if type == "relu":       
        dZ= relu_back(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    elif type == "sigmoid":
        dZ= sigmoid_back(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)
    elif type == "tanh":
        dZ= tanh_back(dA,activation_cache)
        dA_prev, dW, db = linear_backward(dZ,linear_cache)  
    return dA_prev, dW, db


def backward_propagation(AL,Y,caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L-1]
    dA_prev_temp,dW_temp,db_temp = linear_activation_backward(dAL,current_cache,"sigmoid")
    grads["dA" + str(L-1)] = dA_prev_temp
    grads["dW" + str(L)] = dW_temp
    grads["db" + str(L)] = db_temp   
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev_temp,current_cache,"tanh")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l+1)] = dW_temp
        grads["db" + str(l+1)] = db_temp
    return grads
def update_parameters(params,grads,learning_rate):
    parameters = params.copy()
    L = len(parameters) // 2 
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
    return parameters

def train_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    costs = []
    parameters = random_init_parameters(layers_dims)
    for i in range(0, num_iterations):
        AL, caches = forward_propagation(X,parameters)
        cost = calculate_cost(AL,Y)
        grads = backward_propagation(AL,Y,caches)
        parameters= update_parameters(parameters,grads,learning_rate)
        if print_cost and i % 100 == 0 or i == num_iterations - 1:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if i % 100 == 0 or i == num_iterations:
            costs.append(cost)
        if np.any(cost <= 0.035):
            break
     
    return parameters, costs

def predict(X,Y,parameters):
    AL,caches = forward_propagation(X,parameters)
    return AL.astype(float)

def train_and_check_acc(learning_rate,iter_n,layer_ns,n_p,train_size):
    path = "./images/dataset/training_set"
    X_train,Y_train = load_images(path,n_p,train_size)
    print(X_train.shape)
    print(Y_train.shape) 
    parameters,costs = train_model(X_train,Y_train,layer_ns,learning_rate,iter_n,True)
    pred = (predict(X_train,Y_train,parameters)>0.5).astype(int)
    print("accuracy on training set = %",100-(100*np.sum(abs(Y_train-pred),axis=1))/train_size)

    path = "./images/dataset/test_set"
    X_test,Y_test = load_images(path,n_p,200)
    pred = (predict(X_test,Y_test,parameters)>0.5).astype(int)
    print("accuracy on test set = %",100-(np.sum(abs(Y_test-pred),axis=1))/2)

    save_parameters(parameters)

def save_parameters(parameters):
    L = int(len(parameters)/2)
    for l in range(1,L+1):
        w_str = "W"+str(l)
        b_str = "b"+str(l)
        np.save(w_str+".npy",parameters[w_str])
        np.save(b_str+".npy",parameters[b_str])
    np.save("layer_size.npy",L)

def load_parameters():
    parameters = {}
    L = np.load("layer_size.npy")
    for l in range(1,L+1):
        w_str = "W"+str(l)
        b_str = "b"+str(l)
        parameters[w_str] = np.load(w_str+".npy")
        parameters[b_str] = np.load(b_str+".npy")
    np.save("layer_size.npy",L*2)
    return parameters

def serve_image(path,num_px,index,parameters):
    i = 0
    for pict in glob2.glob(path+"/*/*.jpg"):
        if i == index:
            image = np.array(Image.open(pict).resize((num_px, num_px)))
            plt.imshow(image)
            
            image = image / 255.
            image = image.reshape((1, num_px * num_px * 3)).T
            X = image
            Y = 1
            pred = predict(X,Y,parameters)
            if pred > 0.5:
                plt.title("Cat with %"+str(int(pred[0][0]*100))+" certanity")
            else:
                plt.title("Dog with %"+str(100-int(pred[0][0]*100))+" certanity")
            plt.show()
            break
        i = i + 1

def show_images():
    n_p=100
    path = "./images/dataset/test_set"
    parameters = load_parameters()
    while (1):
        index = random.randint(0,8000)
        serve_image(path,n_p,index,parameters)

n_p = 75
layer_ns = [3*n_p*n_p,20, 2] 
train_and_check_acc(0.005,35000,layer_ns,n_p,4000)
#show_images()