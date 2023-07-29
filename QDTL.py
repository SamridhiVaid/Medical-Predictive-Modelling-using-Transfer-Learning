

# -*- coding: utf-8 -*-


import keras
from keras.models import Sequential
import os
from keras.models import load_model, model_from_json
import csv
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import os
import pickle

import numpy as np
import random
import tensorflow as tf
import statistics

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)
# tf.random.set_random_seed(1)

os.getcwd()

import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
import os
from random import randint
import random
import numpy as np
import random
import numpy as np
import time
import random
import h5py 
from operator import add
import random

import time


import numpy as np
import random
import time



#Check similarity between two models to eliminate similar models
def check_similar_models(model_1, model_2):
  if(model_1 == model_2):
    return True
  
  return False

retain=0.4


#Function to remove duplicate models for quality population
def remove_duplicated_models(population, generation_num, model_accuracy):
  count=0
  if(generation_num==1):
    for i in range(0, len(population)-1):
      x=[]
      for j in range(i+1, len(population)):
        if(check_similar_models(model_accuracy[i], model_accuracy[j])):
          print("Both the Networks are Equal so remove it!!")
          print(model_accuracy[i], "<-->",  model_accuracy[j])
          x.append(j)
      count+=len(x)
      print("All the elemnts are duplicated at position: ", x)
      population=np.delete(population, x).tolist()
      model_accuracy=np.delete(model_accuracy, x).tolist()
    print("The population")
    print("The Number of networks deleted are: ", count)
    print("The size of population after deletion is:", len(population))    
    for i in range(0, count):
      child=mutation(population[0])
      print("..........The new models fitness is:.......")
      child_accuracy=fitness(child)
      print(child_accuracy)
      population.append(child)
      model_accuracy.append(child_accuracy)
  
  elif(generation_num>1):
    for i in range(0, len(population)-1):
      x=[]
      for j in range(i+1, len(population)):
        if(check_similar_models(model_accuracy[i], model_accuracy[j])):
          print("Both the Networks are Equal so remove it!!")
          print(model_accuracy[i], "<-->",  model_accuracy[j])
          x.append(j)
      count+=len(x)
      print("All the elemnts are duplicated at position: ", x)
      population=np.delete(population, x).tolist()
      model_accuracy=np.delete(model_accuracy, x).tolist()
    print("No of networks deleted in the population are: ", count)
    print("The length of population is after deleting is :", len(population))
    network_num=10
    for i in range(0, count):
      parent_networks=population[:4] #Top 40% of the population!! #Please change heere if the population size changes!!
      indices=np.arange(len(parent_networks)).tolist()
      print("The Indices Matrix are: ", indices)
      [male, female]=random.sample(indices, k=2)
      print("Two different networks are Male Position is: ", male,"The Female Position is: ", female)
      babies=crossover_and_mutation(parent_networks[male], parent_networks[female])
      for baby in babies:
        #Please change here if the population size changes!!
        if(len(population)<10):
          print("..................The fitness of the new model is:.............")
          child_accuracy=fitness(baby)
          print(child_accuracy)
          population.append(baby)
          model_accuracy.append(child_accuracy)
      #Change the population size if it chnages!!
      if(len(population)==10):
        break
      
    print("The overall population size is after adding new networks: ", len(population))
  
  return population, model_accuracy

retain=0.4




#Function to remove duplicate models for diversity population
def remove_duplicated_models_diversity(population, generation_num, model_accuracy):
  count=0
  if(generation_num==1):
    for i in range(0, len(population)-1):
      x=[]
      for j in range(i+1, len(population)):
        if(check_similar_models(model_accuracy[i], model_accuracy[j])):
          print("Both the Networks are Equal so remove it!!")
          print(model_accuracy[i], "<-->",  model_accuracy[j])
          x.append(j)
      count+=len(x)
      print("All the elemnts are duplicated at position: ", x)
      population=np.delete(population, x).tolist()
      model_accuracy=np.delete(model_accuracy, x).tolist()
    print("The population")
    print("The Number of networks deleted are: ", count)
    print("The size of population after deletion is:", len(population))    
    for i in range(0, count):
      child=mutation(population[0])
      print("..........The new models fitness is:.......")
      child_accuracy=fitness_diversity(child)
      print(child_accuracy)
      population.append(child)
      model_accuracy.append(child_accuracy)
  
  elif(generation_num>1):
    for i in range(0, len(population)-1):
      x=[]
      for j in range(i+1, len(population)):
        if(check_similar_models(model_accuracy[i], model_accuracy[j])):
          print("Both the Networks are Equal so remove it!!")
          print(model_accuracy[i], "<-->",  model_accuracy[j])
          x.append(j)
      count+=len(x)
      print("All the elemnts are duplicated at position: ", x)
      population=np.delete(population, x).tolist()
      model_accuracy=np.delete(model_accuracy, x).tolist()
    print("No of networks deleted in the population are: ", count)
    print("The length of population is after deleting is :", len(population))
    network_num=10
    for i in range(0, count):
      parent_networks=population[:4] #Top 40% of the population!! #Please change heere if the population size changes!!
      indices=np.arange(len(parent_networks)).tolist()
      print("The Indices Matrix are: ", indices)
      [male, female]=random.sample(indices, k=2)
      print("Two different networks are Male Position is: ", male,"The Female Position is: ", female)
      babies=crossover_and_mutation(parent_networks[male], parent_networks[female])
      for baby in babies:
        #Please change here if the population size changes!!
        if(len(population)<10):
          print("..................The fitness of the new model is:.............")
          child_accuracy=fitness_diversity(baby)
          print(child_accuracy)
          population.append(baby)
          model_accuracy.append(child_accuracy)
      #Change the population size if it chnages!!
      if(len(population)==10):
        break
      
    print("The overall population size is after adding new networks: ", len(population))
  
  return population, model_accuracy


#Loading the base model 
with open("PTB_NPTB_Fold5D_Train_X_normal.pickle", "rb") as fp:
    x_train_r = pickle.load(fp)

with open("PTB_NPTB_Fold5D_Train_Y_normal.pickle", "rb") as fp:
    y_train_r = pickle.load(fp)

with open("20_PTB_Fold5_Test_X_normal.pickle", "rb") as fp:
    x_test_r = pickle.load(fp)

with open("20_PTB_Fold5_Test_Y_normal.pickle", "rb") as fp:
    y_test_r = pickle.load(fp)

print(x_train_r.shape)
print(y_test_r.shape)
print(x_test_r.shape)

x_train = np.reshape(x_train_r, (53, 70,1))
y_train = np.reshape(np.array(y_train_r), (53,1))
x_test = np.reshape(x_test_r, (8, 70,1))
y_test = np.reshape(y_test_r, (8,1))




#Mutation functions for generating candidate solutions
### Multiplying alpha by random number in range [-2, 2] for a layer.

def mutation_1(model_1):
    model = model_1
    model_weights_1 = model.get_weights()
    model_weights = model_weights_1.copy() 
    base_alphas=[]
    for layer in model_weights_1:
      alpha_layer = np.ones(layer.shape)
      base_alphas.append(alpha_layer)
    model_alphas = base_alphas.copy()
    ## To get the ture values of the weights i.e. by dividing with alpha values.. Since we are multiplying the weights with alphas in the end
    for index in range(len(model_weights)):
        model_weights[index] = np.divide(model_weights[index], model_alphas[index])
    
    layer_index = random.randint(1, len(model_weights)-1)
    current_layer = model.get_weights()[layer_index]
    
    loaded_model = tf.keras.models.clone_model(model)
    
    print("Mutaion operation X where multiplying alpha with random range for a layer")
#     print("The model layer: ", layer_index)
    if len(current_layer.shape) ==2:
        x_index = current_layer.shape[0]
        y_index = current_layer.shape[1]
        x = np.random.uniform(-2, 2, size=(x_index, y_index))
        model_alphas[layer_index] = np.multiply(model_alphas[layer_index],x)
    elif len(current_layer.shape) == 1:
        x_index = current_layer.shape[0]
        x = np.random.uniform(-2, 2, size=x_index)
        model_alphas[layer_index] = np.multiply(model_alphas[layer_index],x)
    
    ### Multiply the weights with alpha values.
    for index in range(len(model_weights)):
        model_weights[index] = np.multiply(model_weights[index], model_alphas[index])
    
    
    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001,decay=0.0) 
    loaded_model.set_weights(model_weights)
    loaded_model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    return loaded_model

### Replaces f values of a random layer with a randomly selected f values
def mutation_2(model_1):
    model = model_1
    model_weights_1 = model.get_weights()
    model_weights = model_weights_1.copy()
    base_alphas=[]
    for layer in model_weights_1:
      alpha_layer = np.ones(layer.shape)
      base_alphas.append(alpha_layer) 
    model_alphas = base_alphas.copy()
    ## To get the ture values of the weights i.e. by dividing with alpha values.. Since we are multiplying the weights with alphas in the end
    for index in range(len(model_weights)):
        model_weights[index] = np.divide(model_weights[index], model_alphas[index])
    ### source layer index
#     layer_index = random.randint(0, len(model_weights)-1)
    source_index = random.choice([i for i in range(1, len(model_weights)-2) if len(model_weights[i].shape)==2])
    
    ### target layer index
    valid_indices= [i for i in range(1, len(model_weights)-2) if len(model_weights[i].shape)==2]
    valid_indices.remove(source_index)
    target_index = random.choice(valid_indices)
    print(f" Replacing values in target index {source_index} from the source index is {target_index}")
    
    ### Since all the weights are in same shape for the 2D data, we can directly replace the values. 
    model_weights[target_index] = model_weights[source_index]
    
    loaded_model = tf.keras.models.clone_model(model)
    
    ### Multiply the weights with alpha values.
    for index in range(len(model_weights)):
        model_weights[index] = np.multiply(model_weights[index], model_alphas[index])
    
    
    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001,decay=0.0) 
    loaded_model.set_weights(model_weights)
    loaded_model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    return loaded_model

### Add a random a and f to a random position in a random layer.

def mutation_3(model_1):
    model = model_1
    model_weights_1 = model.get_weights()
    model_weights = model_weights_1.copy()
    base_alphas=[]
    for layer in model_weights_1:
      alpha_layer = np.ones(layer.shape)
      base_alphas.append(alpha_layer) 
    model_alphas = base_alphas.copy() 
    
    ## To get the ture values of the weights i.e. by dividing with alpha values.. Since we are multiplying the weights with alphas in the end
    for index in range(len(model_weights)):
        model_weights[index] = np.divide(model_weights[index], model_alphas[index])
    ### source layer index
#     layer_index = random.randint(0, len(model_weights)-1)
    source_index = random.choice([i for i in range(1, len(model_weights)-2) if len(model_weights[i].shape)==2])
    ### target layer index
    valid_indices= [i for i in range(1, len(model_weights)-2) if len(model_weights[i].shape)==2]
    valid_indices.remove(source_index)
    target_index = random.choice(valid_indices)
#     target_index = random.choice([i for i in range(1, len(model_weights)-1) if len(model_weights[i].shape)==2].remove(source_index))
    print(f" Replacing values in Adding target index{source_index} to the source index is {target_index}")
    
    ### Since all the weights are in same shape for the 2D data, we can directly replace the values. 
    model_weights[target_index] = model_weights[target_index] + model_weights[source_index]
    model_alphas[target_index] = model_alphas[target_index] + model_alphas[source_index]
    
    loaded_model = tf.keras.models.clone_model(model)
    
    ### Multiply the weights with alpha values.
    for index in range(len(model_weights)):
        model_weights[index] = np.multiply(model_weights[index], model_alphas[index])
    
    
    #adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001,decay=0.0) 
    loaded_model.set_weights(model_weights)
    loaded_model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    return loaded_model

### Swapping two random layers with 'f' and 'a'

def mutation_4(model_1):
    model = model_1
    model_weights_1 = model.get_weights()
    model_weights = model_weights_1.copy()
    base_alphas=[]
    for layer in model_weights_1:
      alpha_layer = np.ones(layer.shape)
      base_alphas.append(alpha_layer) 
    model_alphas = base_alphas.copy() 
  
    ## To get the true values of the weights i.e. by dividing with alpha values.. Since we are multiplying the weights with alphas in the end
    for index in range(len(model_weights)):
        model_weights[index] = np.divide(model_weights[index], model_alphas[index])
    ### source layer index
#     layer_index = random.randint(0, len(model_weights)-1)
    source_index = random.choice([i for i in range(1, len(model_weights)-2) if len(model_weights[i].shape)==2])
    ### target layer index
    valid_indices= [i for i in range(1, len(model_weights)-2) if len(model_weights[i].shape)==2]
    valid_indices.remove(source_index)
    target_index = random.choice(valid_indices)
#     target_index = random.choice([i for i in range(1, len(model_weights)-1) if len(model_weights[i].shape)==2].remove(source_index))
    print(f" Swapping values at target index {source_index} and source index is {target_index}")
    
    ### Since all the weights are in same shape for the 2D data, we can directly swap without reshaping them.
    model_weights[source_index], model_weights[target_index] = model_weights[target_index],model_weights[source_index]
    
    model_alphas[source_index], model_alphas[target_index] = model_alphas[target_index], model_alphas[source_index]
    
    loaded_model = tf.keras.models.clone_model(model)
    
    ### Multiply the weights with alpha values.
    for index in range(len(model_weights)):
        model_weights[index] = np.multiply(model_weights[index], model_alphas[index])
    
    
  
    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001,decay=0.0) 
    loaded_model.set_weights(model_weights)
    loaded_model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    return loaded_model

# Add weights to a particular muataion index
from keras.models import model_from_json

def mutation_only_weights_1(model_1):
    #deifne weights
    print('Applying Weights only Mutation 1! - Adding to an index')
    model = model_1
    model_weights_1 = model.get_weights()
    model_weights = model_weights_1.copy()
    base_alphas=[]
    for layer in model_weights_1:
      alpha_layer = np.ones(layer.shape)
      base_alphas.append(alpha_layer) 
    model_alphas = base_alphas.copy()
    ## To get the true values of the weights i.e. by dividing with alpha values.. Since we are multiplying the weights with alphas in the end
    for index in range(len(model_weights)):
        model_weights[index] = np.divide(model_weights[index], model_alphas[index])
    
    weights_index = random.randint(0, len(model_weights)-1)
    current_layer = model.get_weights()[weights_index]
    
#     json_file = open('base_model.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)

    loaded_model = tf.keras.models.clone_model(model)
#     print(loaded_model.get_weights()[-1].shape)
    
    ###For weights
    if len(current_layer.shape) ==2:
        x_index = random.randint(0, current_layer.shape[0]-1)
        y_index = random.randint(0, current_layer.shape[1]-1)
        x = np.random.random()
        model_weights[weights_index][x_index][y_index] = model_weights[weights_index][x_index][y_index] + x
    elif len(current_layer.shape) ==1:
        x_index = random.randint(0, current_layer.shape[0]-1)
        x = np.random.random()
        model_weights[weights_index][x_index] = model_weights[weights_index][x_index] + x
    
    ### For alpha values
    if len(current_layer.shape) ==2:
        x_index = random.randint(0, current_layer.shape[0]-1)
        y_index = random.randint(0, current_layer.shape[1]-1)
        x = np.random.random()
        model_alphas[weights_index][x_index][y_index] = model_alphas[weights_index][x_index][y_index] + x
    elif len(current_layer.shape) ==1:
        x_index = random.randint(0, current_layer.shape[0]-1)
        x = np.random.random()
        model_alphas[weights_index][x_index] = model_alphas[weights_index][x_index] + x
    
    #### Multiply the weights with alpha values.
    for index in range(len(model_weights)):
        model_weights[index] = np.multiply(model_weights[index], model_alphas[index])
    
    #adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001,decay=0.0) 
    loaded_model.set_weights(model_weights)
    loaded_model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    return loaded_model

# Add weights to a particular muatation layer
from keras.models import model_from_json

def mutation_only_weights_2(model_1):
    #deifne weights
    print('Applying Weights only Mutation 2! - Adding to an layer')
    model = model_1
    model_weights_1 = model.get_weights()
    model_weights = model_weights_1.copy()
    base_alphas=[]
    for layer in model_weights_1:
      alpha_layer = np.ones(layer.shape)
      base_alphas.append(alpha_layer) 
    model_alphas = base_alphas.copy()

    ## To get the true values of the weights i.e. by dividing with alpha values.. Since we are multiplying the weights with alphas in the end
    for index in range(len(model_weights)):
        model_weights[index] = np.divide(model_weights[index], model_alphas[index])
    
    weights_index = random.randint(0, len(model_weights)-1)
    current_layer = model.get_weights()[weights_index]
  

    loaded_model = tf.keras.models.clone_model(model)

    
    ###For weights
    if len(current_layer.shape) ==2:
        x_index = current_layer.shape[0]
        y_index = current_layer.shape[1]
        x = np.random.random((x_index, y_index))
        model_weights[weights_index] = np.add(model_weights[weights_index], x)
    elif len(current_layer.shape) ==1:
        x_index = current_layer.shape[0]
        x = np.random.random(x_index)
        model_weights[weights_index] = np.add(model_weights[weights_index], x)
    
    ### For alpha values
    if len(current_layer.shape) ==2:
        x_index = current_layer.shape[0]
        y_index = current_layer.shape[1]
        x = np.random.random((x_index, y_index))
        model_alphas[weights_index] = np.add(model_alphas[weights_index], x)
    elif len(current_layer.shape) ==1:
        x_index = current_layer.shape[0]
        x = np.random.random(x_index)
        model_alphas[weights_index] = np.add(model_alphas[weights_index], x)
    
    #### Multiply the weights with alpha values.
    for index in range(len(model_weights)):
        model_weights[index] = np.multiply(model_weights[index], model_alphas[index])
    
    #adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001,decay=0.0) 
    loaded_model.set_weights(model_weights)
    loaded_model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    return loaded_model

# Subtracting weights to a particular muataion index
from keras.models import model_from_json

def mutation_only_weights_3(model_1):
    #deifne weights
    print('Applying Weights only Mutation 3! - Subtracting to an index')
    model = model_1
    model_weights_1 = model.get_weights()
    model_weights = model_weights_1.copy()
    base_alphas=[]
    for layer in model_weights_1:
      alpha_layer = np.ones(layer.shape)
      base_alphas.append(alpha_layer) 
    model_alphas = base_alphas.copy()
    ## To get the true values of the weights i.e. by dividing with alpha values.. Since we are multiplying the weights with alphas in the end
    for index in range(len(model_weights)):
        model_weights[index] = np.divide(model_weights[index], model_alphas[index])
    
    weights_index = random.randint(0, len(model_weights)-1)
    current_layer = model.get_weights()[weights_index]
    

    loaded_model = tf.keras.models.clone_model(model)

    
    ###For weights
    if len(current_layer.shape) ==2:
        x_index = random.randint(0, current_layer.shape[0]-1)
        y_index = random.randint(0, current_layer.shape[1]-1)
        x = np.random.random()
        model_weights[weights_index][x_index][y_index] = model_weights[weights_index][x_index][y_index] - x
    elif len(current_layer.shape) ==1:
        x_index = random.randint(0, current_layer.shape[0]-1)
        x = np.random.random()
        model_weights[weights_index][x_index] = model_weights[weights_index][x_index] - x
    
    ### For alpha values
    if len(current_layer.shape) ==2:
        x_index = random.randint(0, current_layer.shape[0]-1)
        y_index = random.randint(0, current_layer.shape[1]-1)
        x = np.random.random()
        model_alphas[weights_index][x_index][y_index] = model_alphas[weights_index][x_index][y_index] - x
    elif len(current_layer.shape) ==1:
        x_index = random.randint(0, current_layer.shape[0]-1)
        x = np.random.random()
        model_alphas[weights_index][x_index] = model_alphas[weights_index][x_index] - x
    
    #### Multiply the weights with alpha values.
    for index in range(len(model_weights)):
        model_weights[index] = np.multiply(model_weights[index], model_alphas[index])
    
 
    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001,decay=0.0) 
    loaded_model.set_weights(model_weights)
    loaded_model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    return loaded_model

# Subtracting weights to a particular muataion layer
from keras.models import model_from_json

def mutation_only_weights_4(model_1):
    #deifne weights
    print('Applying Weights only Mutation 4! - Subtracting to an layer')
    model = model_1
    model_weights_1 = model.get_weights()
    model_weights = model_weights_1.copy()
    base_alphas=[]
    for layer in model_weights_1:
      alpha_layer = np.ones(layer.shape)
      base_alphas.append(alpha_layer) 
    model_alphas = base_alphas.copy()
    ## To get the true values of the weights i.e. by dividing with alpha values.. Since we are multiplying the weights with alphas in the end
    for index in range(len(model_weights)):
        model_weights[index] = np.divide(model_weights[index], model_alphas[index])
    
    weights_index = random.randint(0, len(model_weights)-1)
    current_layer = model.get_weights()[weights_index]
    

    loaded_model = tf.keras.models.clone_model(model)
#     print(loaded_model.get_weights()[-1].shape)
    
    ###For weights
    if len(current_layer.shape) ==2:
        x_index = current_layer.shape[0]
        y_index = current_layer.shape[1]
        x = np.random.random((x_index, y_index))
        model_weights[weights_index] = np.subtract(model_weights[weights_index], x)
    elif len(current_layer.shape) ==1:
        x_index = current_layer.shape[0]
        x = np.random.random(x_index)
        model_weights[weights_index] = np.subtract(model_weights[weights_index], x)
    
    ### For alpha values
    if len(current_layer.shape) ==2:
        x_index = current_layer.shape[0]
        y_index = current_layer.shape[1]
        x = np.random.random((x_index, y_index))
        model_alphas[weights_index] = np.subtract(model_alphas[weights_index], x)
    elif len(current_layer.shape) ==1:
        x_index = current_layer.shape[0]
        x = np.random.random(x_index)
        model_alphas[weights_index] = np.subtract(model_alphas[weights_index], x)
    
    #### Multiply the weights with alpha values.
    for index in range(len(model_weights)):
        model_weights[index] = np.multiply(model_weights[index], model_alphas[index])

    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001, decay=0.0) 
    loaded_model.set_weights(model_weights)
    loaded_model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    return loaded_model

# Multiplying weights to a particular muataion index
from keras.models import model_from_json

def mutation_only_weights_5(model_1):
    #deifne weights
    print('Applying Weights only Mutation 5! - Multiplying to an index')
    model = model_1
    model_weights_1 = model.get_weights()
    model_weights = model_weights_1.copy()
    base_alphas=[]
    for layer in model_weights_1:
      alpha_layer = np.ones(layer.shape)
      base_alphas.append(alpha_layer) 
    model_alphas = base_alphas.copy()
    ## To get the true values of the weights i.e. by dividing with alpha values.. Since we are multiplying the weights with alphas in the end
    for index in range(len(model_weights)):
        model_weights[index] = np.divide(model_weights[index], model_alphas[index])
    
    weights_index = random.randint(0, len(model_weights)-1)
    current_layer = model.get_weights()[weights_index]
    

    loaded_model = tf.keras.models.clone_model(model)
#     print(loaded_model.get_weights()[-1].shape)
    
    ###For weights
    if len(current_layer.shape) ==2:
        x_index = random.randint(0, current_layer.shape[0]-1)
        y_index = random.randint(0, current_layer.shape[1]-1)
        x = np.random.random()
        model_weights[weights_index][x_index][y_index] = model_weights[weights_index][x_index][y_index] * x
    elif len(current_layer.shape) ==1:
        x_index = random.randint(0, current_layer.shape[0]-1)
        x = np.random.random()
        model_weights[weights_index][x_index] = model_weights[weights_index][x_index] * x
    
    ### For alpha values
    if len(current_layer.shape) ==2:
        x_index = random.randint(0, current_layer.shape[0]-1)
        y_index = random.randint(0, current_layer.shape[1]-1)
        x = np.random.random()
        model_alphas[weights_index][x_index][y_index] = model_alphas[weights_index][x_index][y_index] * x
    elif len(current_layer.shape) ==1:
        x_index = random.randint(0, current_layer.shape[0]-1)
        x = np.random.random()
        model_alphas[weights_index][x_index] = model_alphas[weights_index][x_index] * x
    
    #### Multiply the weights with alpha values.
    for index in range(len(model_weights)):
        model_weights[index] = np.multiply(model_weights[index], model_alphas[index])

    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001, decay=0.0) 
    loaded_model.set_weights(model_weights)
    loaded_model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    return loaded_model

# Multiplying weights to a particular muataion layer
from keras.models import model_from_json

def mutation_only_weights_6(model_1):
    #deifne weights
    print('Applying Weights only Mutation 6! - Multiplying to an layer')
    model = model_1
    model_weights_1 = model.get_weights()
    model_weights = model_weights_1.copy()
    base_alphas=[]
    for layer in model_weights_1:
      alpha_layer = np.ones(layer.shape)
      base_alphas.append(alpha_layer) 
    model_alphas = base_alphas.copy()
    ## To get the true values of the weights i.e. by dividing with alpha values.. Since we are multiplying the weights with alphas in the end
    for index in range(len(model_weights)):
        model_weights[index] = np.divide(model_weights[index], model_alphas[index])
    
    weights_index = random.randint(0, len(model_weights)-1)
    current_layer = model.get_weights()[weights_index]
    

    loaded_model = tf.keras.models.clone_model(model)
#     print(loaded_model.get_weights()[-1].shape)
    
    ###For weights
    if len(current_layer.shape) ==2:
        x_index = current_layer.shape[0]
        y_index = current_layer.shape[1]
        x = np.random.random((x_index, y_index))
        model_weights[weights_index] = np.multiply(model_weights[weights_index], x)
    elif len(current_layer.shape) ==1:
        x_index = current_layer.shape[0]
        x = np.random.random(x_index)
        model_weights[weights_index] = np.multiply(model_weights[weights_index], x)
    
    ### For alpha values
    if len(current_layer.shape) ==2:
        x_index = current_layer.shape[0]
        y_index = current_layer.shape[1]
        x = np.random.random((x_index, y_index))
        model_alphas[weights_index] = np.multiply(model_alphas[weights_index], x)
    elif len(current_layer.shape) ==1:
        x_index = current_layer.shape[0]
        x = np.random.random(x_index)
        model_alphas[weights_index] = np.multiply(model_alphas[weights_index], x)
    
    #### Multiply the weights with alpha values.
    for index in range(len(model_weights)):
        model_weights[index] = np.multiply(model_weights[index], model_alphas[index])
    
    #adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001, decay=0.0) 
    loaded_model.set_weights(model_weights)
    loaded_model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    return loaded_model

# Divide weights to a particular muataion index
from keras.models import model_from_json

def mutation_only_weights_7(model_1):
    #deifne weights
    print('Applying Weights only Mutation 7! - Dividing to an index')
    model = model_1
    model_weights_1 = model.get_weights()
    model_weights = model_weights_1.copy()
    base_alphas=[]
    for layer in model_weights_1:
      alpha_layer = np.ones(layer.shape)
      base_alphas.append(alpha_layer) 
    model_alphas = base_alphas.copy()
    ## To get the true values of the weights i.e. by dividing with alpha values.. Since we are multiplying the weights with alphas in the end
    for index in range(len(model_weights)):
        model_weights[index] = np.divide(model_weights[index], model_alphas[index])
    
    weights_index = random.randint(0, len(model_weights)-1)
    current_layer = model.get_weights()[weights_index]


    loaded_model = tf.keras.models.clone_model(model)
#     print(loaded_model.get_weights()[-1].shape)
    
    ###For weights
    if len(current_layer.shape) ==2:
        x_index = random.randint(0, current_layer.shape[0]-1)
        y_index = random.randint(0, current_layer.shape[1]-1)
        x = np.random.random()
        model_weights[weights_index][x_index][y_index] = model_weights[weights_index][x_index][y_index] / x
    elif len(current_layer.shape) ==1:
        x_index = random.randint(0, current_layer.shape[0]-1)
        x = np.random.random()
        model_weights[weights_index][x_index] = model_weights[weights_index][x_index] / x
    
    ### For alpha values
    if len(current_layer.shape) ==2:
        x_index = random.randint(0, current_layer.shape[0]-1)
        y_index = random.randint(0, current_layer.shape[1]-1)
        x = np.random.random()
        model_alphas[weights_index][x_index][y_index] = model_alphas[weights_index][x_index][y_index] / x
    elif len(current_layer.shape) ==1:
        x_index = random.randint(0, current_layer.shape[0]-1)
        x = np.random.random()
        model_alphas[weights_index][x_index] = model_alphas[weights_index][x_index] / x
    
    #### Multiply the weights with alpha values.
    for index in range(len(model_weights)):
        model_weights[index] = np.multiply(model_weights[index], model_alphas[index])
    

    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001, decay=0.0) 
    loaded_model.set_weights(model_weights)
    loaded_model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    return loaded_model

# Divide weights to a particular muataion layer
from keras.models import model_from_json

def mutation_only_weights_8(model_1):
    #deifne weights
    print('Applying Weights only Mutation 8! - Dividing to an layer')
    model = model_1
    model_weights_1 = model.get_weights()
    model_weights = model_weights_1.copy()
    base_alphas=[]
    for layer in model_weights_1:
      alpha_layer = np.ones(layer.shape)
      base_alphas.append(alpha_layer) 
    model_alphas = base_alphas.copy()
    ## To get the true values of the weights i.e. by dividing with alpha values.. Since we are multiplying the weights with alphas in the end
    for index in range(len(model_weights)):
        model_weights[index] = np.divide(model_weights[index], model_alphas[index])
    
    weights_index = random.randint(0, len(model_weights)-1)
    current_layer = model.get_weights()[weights_index]
    

    loaded_model = tf.keras.models.clone_model(model)
#     print(loaded_model.get_weights()[-1].shape)
    
    ###For weights
    if len(current_layer.shape) ==2:
        x_index = current_layer.shape[0]
        y_index = current_layer.shape[1]
        x = np.random.random((x_index, y_index))
        if(model_weights[weights_index]>0):
            model_weights[weights_index] = np.divide(model_weights[weights_index], x)
    elif len(current_layer.shape) ==1:
        x_index = current_layer.shape[0]
        x = np.random.random(x_index)
        if(model_weights[weights_index]>0):
            model_weights[weights_index] = np.divide(model_weights[weights_index], x)
    
    ### For alpha values
    if len(current_layer.shape) ==2:
        x_index = current_layer.shape[0]
        y_index = current_layer.shape[1]
        x = np.random.random((x_index, y_index))
        if(model_weights[weights_index]>0):
            model_alphas[weights_index] = np.divide(model_alphas[weights_index], x)
    elif len(current_layer.shape) ==1:
        x_index = current_layer.shape[0]
        x = np.random.random(x_index)
        if(model_weights[weights_index]>0):
            model_alphas[weights_index] = np.divide(model_alphas[weights_index], x)
    
    #### Multiply the weights with alpha values.
    for index in range(len(model_weights)):
        model_weights[index] = np.multiply(model_weights[index], model_alphas[index])
    

    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001,decay=0.0) 
#model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    loaded_model.set_weights(model_weights)
    loaded_model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    return loaded_model

from keras.models import model_from_json

### Mutation functions

### Conceptual expansion mutation operations

def CE_mutation(model):
    x = random.randint(1, 4)
    if(x==1):
        model_1 = mutation_1(model)
    elif(x==2):
        model_1 = mutation_2(model)
    elif(x==3):
        model_1 = mutation_3(model)
    elif(x==4):
        model_1 = mutation_4(model)
    
    return model_1

### Mutation functions

### Weights only mutation operations


def weight_mutation(model):
    x = random.randint(1, 7)
    if(x==1):
        model_1 = mutation_only_weights_1(model)
    elif(x==2):
        model_1 = mutation_only_weights_2(model)
    elif(x==3):
        model_1 = mutation_only_weights_3(model)
    elif(x==4):
        model_1 = mutation_only_weights_4(model)
    elif(x==5):
        model_1 = mutation_only_weights_5(model)
    elif(x==6):
        model_1 = mutation_only_weights_6(model)
    elif(x==7):
        model_1 = mutation_only_weights_7(model)

    
    return model_1

### Mutation functions

### combined only mutation operations


def mutation(model):
    x = random.randint(1, 11)
    if(x==1):
        model_1 = mutation_only_weights_1(model)
    elif(x==2):
        model_1 = mutation_only_weights_2(model)
    elif(x==3):
        model_1 = mutation_only_weights_3(model)
    elif(x==4):
        model_1 = mutation_only_weights_4(model)
    elif(x==5):
        model_1 = mutation_only_weights_5(model)
    elif(x==6):
        model_1 = mutation_only_weights_6(model)
    elif(x==7):
        model_1 = mutation_only_weights_7(model)
    elif(x==8):
        model_1 = mutation_1(model)
    elif(x==9):
        model_1 = mutation_2(model)
    elif(x==10):
        model_1 = mutation_3(model)
    elif(x==11):
        model_1 = mutation_4(model)
    
    return model_1


# Arrange the dimensions of the weight layers

def arrange_dim(weights, desired_shape):
  shapex=weights[0].shape
  print("Change the shape",shapex,"->",desired_shape)

  count1=0
  count2=0

  for i in range(0, len(desired_shape)):
    if(weights[0].shape[i]< desired_shape[i]):
      count1+=1
    elif(weights[0].shape[i]>desired_shape[i]):
      count2+=1

  if(count2>0):
    if(len(shapex)==4):
      weights[0]= weights[0][:desired_shape[0], :desired_shape[1], :desired_shape[2], :desired_shape[3]]

    elif(len(shapex)==3):
      weights[0]= weights[0][:desired_shape[0], :desired_shape[1], :desired_shape[2]]
    
    elif(len(shapex)==2):
      weights[0]= weights[0][:desired_shape[0], :desired_shape[1]]
    
    elif(len(shapex)==1):
      weights[0]= weights[0][:desired_shape[0]]
    
  return weights, count1, count2

from keras.layers.core import Lambda
from keras import backend as K



#Crossover function for generating new child models 
mutate_chance=0.2

def crossover_and_mutation(network_1, network_2):

  print("CrossOver Operation took Place")
  childern=[]

  for _ in range(2):
    len_network_1=len(network_1.layers)
    len_network_2=len(network_2.layers)

    #Tracking all the positions of LSTM layers in the network -1
    network_1_LSTM_positions=[]
    i=0
    for layer in network_1.layers:
      if(isinstance(layer, tf.keras.layers.LSTM)):
        network_1_LSTM_positions.append(i)
      i+=1

    #Tracking all the positions of LSTM layers in the network -2
    network_2_LSTM_positions=[]
    i=0
    for layer in network_2.layers:
      if(isinstance(layer, tf.keras.layers.LSTM)):
        network_2_LSTM_positions.append(i)
      i+=1
    
    split_position_1=random.choice(network_1_LSTM_positions[1:])
    split_position_2=random.choice(network_2_LSTM_positions[1:])

    print("The position at which the model is splitted from network 1 is %d" %(split_position_1))

    print("The position at which the model is splitted from network 2 is %d" %(split_position_2))

    crossed_child=Sequential()

    #Crossing both the parents at split positions!!
    i=0
    for layer in network_1.layers:
      # print(layer.get_config()['name'])

      if(i==split_position_1):
        break
      else:
        if(("lstm" in layer.get_config()['name']) and i==0):
          crossed_child.add(tf.keras.layers.LSTM(units=layer.get_config()['units'], return_sequences=layer.get_config()['return_sequences'], input_shape=layer.get_config()['batch_input_shape'][1:]))
        elif(("lstm" in  layer.get_config()['name']) and i==split_position_1):
          break
        elif(("lstm" in  layer.get_config()['name'])):
          crossed_child.add(tf.keras.layers.LSTM(units=layer.get_config()['units'], return_sequences=layer.get_config()['return_sequences']))
        elif("dropout" in layer.get_config()['name']):
          crossed_child.add(tf.keras.layers.Dropout(rate=layer.get_config()['rate']))
        elif(("lstm" in  layer.get_config()['name'])):
          crossed_child.add(tf.keras.layers.LSTM(units=layer.get_config()['units'], return_sequences=layer.get_config()['return_sequences']))
        elif("dropout" in layer.get_config()['name']):
          crossed_child.add(tf.keras.layers.Dropout(rate=layer.get_config()['rate']))
        elif("dense" in layer.get_config()['name']):
          crossed_child.add(Dense(units=layer.get_config()['units']))
      i+=1
    
    i=0
    for layer in network_2.layers:
      # print(layer.get_config()['name'])
      if(i>=split_position_2):
        if(("lstm" in layer.get_config()['name']) and i==0):
          crossed_child.add(tf.keras.layers.LSTM(units=layer.get_config()['units'], return_sequences=layer.get_config()['return_sequences'], input_shape=layer.get_config()['batch_input_shape'][1:]))
        elif("dropout" in layer.get_config()['name']):
          crossed_child.add(tf.keras.layers.Dropout(rate=layer.get_config()['rate']))
        elif(("lstm" in  layer.get_config()['name'])):
          crossed_child.add(tf.keras.layers.LSTM(units=layer.get_config()['units'], return_sequences=layer.get_config()['return_sequences']))
        elif("dropout" in layer.get_config()['name']):
          crossed_child.add(tf.keras.layers.Dropout(rate=layer.get_config()['rate']))
        elif("dense" in layer.get_config()['name']):
          crossed_child.add(Dense(units=layer.get_config()['units']))
      i+=1
    

    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001 ,decay=0.0) 
   
    crossed_child.compile(optimizer = RMSprop, loss = 'mean_squared_error')


    for layer in crossed_child.layers:
      print(layer.get_config())
    
    #Assign the weights to the crossed child network

    for i in range(0, len(network_1.layers)):
      # model_layer.set_weights(network_layer)
      a=i
      b=i
      # if(i>=position):
      #   a=i
      #   b=i+2

      print(a, b)

      model_layer=crossed_child.layers[b]
      network_layer= network_1.layers[a]

      x=model_layer.get_config()
      y=network_layer.get_config()

      del x['name']
      del y['name']


      if(i==split_position_1):
        break
      else:
        if((x==y) and (len(network_layer.get_weights())>0)):
          desired_weights_shape=model_layer.get_weights()[0].shape
          print("CORRECT__1")
          weights, count1, count2=arrange_dim(network_layer.get_weights(),  desired_weights_shape)
          if(count2>=0 and count1==0):
            crossed_child.layers[b].set_weights(weights)
            print("Weights Matched and assigned!!")
          else:
            print("Default weights are assigned!!")
        else:
          print("The layers doesn't contain the weights!!")

    z=0
    for i in range(0, len(network_2.layers)):
      # model_layer.set_weights(network_layer)
      a=i
      b=i

      
      if(i>=split_position_2):
        a=i
        b=split_position_1+z

        print(a, b)
        
        model_layer=crossed_child.layers[b]
        network_layer= network_2.layers[a]

        x=model_layer.get_config()
        y=network_layer.get_config()

        del x['name']
        del y['name']

        if((x==y) and (len(network_layer.get_weights())>0)):
          desired_weights_shape=model_layer.get_weights()[0].shape
          print("CORRECT__2")
          weights, count1, count2=arrange_dim(network_layer.get_weights(),  desired_weights_shape)
          if(count2>=0 and count1==0):
            crossed_child.layers[b].set_weights(weights)
            print("Weights Matched and assigned!!")
          else:
            print("Default weights are assigned!!")
        else:
          print("The layers doesn't contain the weights!!")
        z+=1

      else:
        print(a, b)
        continue
    
    if mutate_chance>random.random():
      print("Crossover --> Mutation took place!!")
      crossed_child=mutation(crossed_child)

    childern.append(crossed_child)

  return childern


#Loading the base model to extract weights from each layer
from keras.models import model_from_json
json_file = open('Ed_L_fold5D_model_G2b.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
modelb = model_from_json(loaded_model_json)

modelb.load_weights("Ed_L_fold5D_model_G2b_weights.h5")

weights_base = []
for layer in  modelb.layers:
  
  if(len(layer.get_weights())==3):
    for i in range(len(layer.get_weights())):
      weights_base.extend(layer.get_weights()[i].reshape(1,-1).tolist()[0])
  elif(len(layer.get_weights())==2):
    for i in range(len(layer.get_weights())):
      weights_base.extend(layer.get_weights()[i].reshape(1,-1).tolist()[0])
  elif(len(layer.get_weights())==1):
    for i in range(len(layer.get_weights())):
      weights_base.extend(layer.get_weights()[i].reshape(1,-1).tolist()[0])


#Diversity fitness function
def fitness_diversity(model):
    weights_child=[]
    saved_weights_child=[]
    for layer in model.layers:
        if(len(layer.get_weights())==3):
            for i in range(len(layer.get_weights())):
                weights_child.extend(layer.get_weights()[i].reshape(1,-1).tolist()[0])
        elif(len(layer.get_weights())==2):
            for i in range(len(layer.get_weights())):
                weights_child.extend(layer.get_weights()[i].reshape(1,-1).tolist()[0])
        elif(len(layer.get_weights())==1):
            for i in range(len(layer.get_weights())):
                weights_child.extend(layer.get_weights()[i].reshape(1,-1).tolist()[0])
        
        
  
  
    diff= [abs(a - b) for a, b in zip(weights_base,weights_child)]
    print("diff",np.mean(diff, axis=0))

    return np.mean(diff, axis=0)


def fitness(model):
    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001 ,decay=0.0) 
    model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
    w=[]
    for i in range(40):
        w.append(model.evaluate(x_train, y_train, verbose=2))
    summ = statistics.mean(w)+ statistics.stdev(w)
    return summ

new_dir_path = 'Fold5D_Weight'
os.mkdir(new_dir_path)

#Training different child models 
def train_networks_and_accuracy(networks, population_names, model_accuracy, generation_num):

  early_stopper = EarlyStopping(patience=3)

  batch_size=15

  print("The current working directory is %s" %(os.getcwd()))
  #Path to folder where the models should be saved
  os.chdir('Experiments_Neurips/Final_PTB_Experiments/Fold5D_Weight') #change directory for diffrent folds 

  print("############################################################################################################")

  accuracy=0
  for i in range(0, len(networks)):
    print("Model number is", i+1)
    print("Model name is: ", population_names[i])
    print("Fitness", fitness(networks[i]))



    if(generation_num==20):
      print("Went through last generation backprop")
      RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001, decay=0.0) 
      networks[i].compile(optimizer = RMSprop, loss = 'mean_squared_error')

      score = networks[i].evaluate(x_test, y_test, verbose=2)

      train_score_z = fitness(networks[i])


        
      network_json = networks[i].to_json()
      filename_json=population_names[i]+".json"
      filename_h5= population_names[i]+".h5"

      with open(filename_json, "w") as json_file:
        json_file.write(network_json)

      networks[i].save_weights(filename_h5)
      print("Saved model to disk")


    score = networks[i].evaluate(x_test, y_test)

   


    train_score_z = fitness(networks[i])
    fields = [i+1, population_names[i], train_score_z, score*100]




    ###########################################  Change the csv file name ######################################

    with open('evaluate_QDTL.csv', 'a') as f:
      writer = csv.writer(f)
      writer.writerow(fields)




  return score

retain=0.4
random_select=0.1
mutate_chance=0.2


########################################## New modified evolve function ##################################################

def evolve(population_networks, generation_num, population_names, model_accuracy, population_networks_diversity, population_names_diversity, model_accuracy_diversity ):

  if(generation_num==1):

    network=population_networks[0]
    desired_length=9
    child=[]
    i=1

    while len(population_networks) < 10:


      mutated_child=mutation(network)
      for kk in range(0, 5):
        mutated_child.fit(x_train, y_train,
              batch_size=10,
              epochs=1,
              verbose=0)

        


      mutated_child_accuracy= fitness(mutated_child)
      model_accuracy.append(mutated_child_accuracy)
      print("The fitness of new mutated child is: ",mutated_child_accuracy)
      population_networks.append(mutated_child)
      child_name= "model%d_%d" %(generation_num, i)
      population_names.append(child_name)

      print("******* Generation 1: Quality The model accuracies before are: ", model_accuracy)

      i+=1

  if(generation_num==1):

    network_diversity=population_networks_diversity[0]
    desired_length_diversity=9
    child_diversity=[]
    j=1

    while len(population_networks_diversity) < 10:


      mutated_child_diversity=mutation(network_diversity)


      mutated_child_accuracy_diversity= fitness_diversity(mutated_child_diversity)
      model_accuracy_diversity.append(mutated_child_accuracy_diversity)
      print("The fitness of new mutated child in diversity: ",mutated_child_accuracy_diversity)


      population_networks_diversity.append(mutated_child_diversity)
      child_name_diversity= "model%d_%d" %(generation_num, j)
      population_names_diversity.append(child_name_diversity)

      j+=1


    print("The number Networks after Evolution for quality: ", len(population_networks))
    print("The number Networks after Evolution for Diversity: ", len(population_networks_diversity))

    print("******* Generation 1: Diversity The model accuracies before are: ", model_accuracy_diversity)

    return population_networks, population_names, model_accuracy, population_networks_diversity, population_names_diversity, model_accuracy_diversity
    
  else:
    population=20

    graded = [(model_accuracy1, network, network_name, model_accuracy2) for model_accuracy1, network, network_name, model_accuracy2 in zip(model_accuracy, population_networks, population_names, model_accuracy)]

    graded_networks = [x[1] for x in sorted(graded, key=lambda x: x[0])]

    graded_network_names=[x[2] for x in sorted(graded, key=lambda x: x[0])]

    grade_model_accuracy=[x[3] for x in sorted(graded, key=lambda x: x[0])]
    retain_length = int(len(graded_networks)*retain)

    parent_networks = graded_networks[::]

    parent_network_names= graded_network_names[::]

    parent_model_accuracy= grade_model_accuracy[::]



    parents_length = len(parent_networks)
    desired_length = population - parents_length
    children = []
    children_names=[]

#Diversity Network 
    graded_diversity = [(model_accuracy1_diversity, network_diversity, network_name_diversity, model_accuracy2_diversity) for model_accuracy1_diversity, network_diversity, network_name_diversity, model_accuracy2_diversity in zip(model_accuracy_diversity, population_networks_diversity, population_names_diversity, model_accuracy_diversity)]

    graded_networks_diversity = [x[1] for x in sorted(graded_diversity, key=lambda x: x[0])]

    graded_network_names_diversity=[x[2] for x in sorted(graded_diversity, key=lambda x: x[0])]

    grade_model_accuracy_diversity=[x[3] for x in sorted(graded_diversity, key=lambda x: x[0])]
    retain_length_diversity = int(len(graded_networks_diversity)*retain)

    parent_networks_diversity = graded_networks_diversity[::]

    parent_network_names_diversity= graded_network_names_diversity[::]

    parent_model_accuracy_diversity= grade_model_accuracy_diversity[::]



    parents_length_diversity = len(parent_networks_diversity)
    desired_length_diversity = population - parents_length_diversity
    children_diversity = []
    children_names_diversity=[]

    i=0
    while len(children) < desired_length:
      male = random.randint(0, parents_length-1)
      female = random.randint(0, parents_length_diversity-1)

      if male != female:
        print("------------------------------------------> The crossed position of male is: ", male, "and Female is: ", female)
        print("CHECK__6")
        male = parent_networks[male]
        female = parent_networks_diversity[female]
        print("Crossover and Mutation is called!!")
        babies = crossover_and_mutation(male, female)

        for baby in babies:
          if len(children) < desired_length:
            children.append(baby)
            children_diversity.append(baby)
            child_accuracy=fitness(baby)
            child_accuracy_diversity=fitness_diversity(baby)
            parent_model_accuracy.append(child_accuracy)
            parent_model_accuracy_diversity.append(child_accuracy_diversity)
            child_name='model%d_%d' %(generation_num, i)
            children_names.append(child_name)
            children_names_diversity.append(child_name)
            i+=1

    parent_networks.extend(children)
    parent_network_names.extend(children_names)

    parent_networks_diversity.extend(children_diversity)
    parent_network_names_diversity.extend(children_names_diversity)


    # Sort the models and then pick top 10 models to the population

    print("The number of models is :", len(parent_networks))


    graded_1 = [(model_accuracy1_1, network_1, network_name_1, model_accuracy2_1) for model_accuracy1_1, network_1, network_name_1, model_accuracy2_1 in zip(parent_model_accuracy, parent_networks, parent_network_names, parent_model_accuracy)]

    parent_networks = [x[1] for x in sorted(graded_1, key=lambda x: x[0])]

    parent_network_names = [x[2] for x in sorted(graded_1, key=lambda x: x[0])]

    parent_model_accuracy = [x[3] for x in sorted(graded_1, key=lambda x: x[0])]

    print(f" Generation {generation_num}: The model accuracies before are: ", parent_model_accuracy)

    parent_networks = parent_networks[:10]

    parent_network_names = parent_network_names[:10]

    parent_model_accuracy = parent_model_accuracy[:10]

    print(f" Generation {generation_num}: The model accuracies after are pruning: ", parent_model_accuracy)


    #Diversity


    graded_1_diversity = [(model_accuracy1_1_diversity, network_1_diversity, network_name_1_diversity, model_accuracy2_1_diversity) for model_accuracy1_1_diversity, network_1_diversity, network_name_1_diversity, model_accuracy2_1_diversity in zip(parent_model_accuracy_diversity, parent_networks_diversity, parent_network_names_diversity, parent_model_accuracy_diversity)]

    parent_networks_diversity = [x[1] for x in sorted(graded_1_diversity, key=lambda x: x[0], reverse=True)]

    parent_network_names_diversity = [x[2] for x in sorted(graded_1_diversity, key=lambda x: x[0], reverse=True)]

    parent_model_accuracy_diversity = [x[3] for x in sorted(graded_1_diversity, key=lambda x: x[0],reverse=True)]

    print(f"Generation {generation_num}: The Diversity model accuracies before are: ", parent_model_accuracy_diversity)

    parent_networks_diversity = parent_networks_diversity[:10]

    parent_network_names_diversity = parent_network_names_diversity[:10]

    parent_model_accuracy_diversity = parent_model_accuracy_diversity[:10]

    print(f"Generation {generation_num}: The Diversity model accuracies after are: ", parent_model_accuracy_diversity)

    return parent_networks, parent_network_names, parent_model_accuracy, parent_networks_diversity, parent_network_names_diversity, parent_model_accuracy_diversity


#Quality Diversity Combibned Mutation

# early_stopper = EarlyStopping(patience=5)

import os

def generate(generations, population):

    #Assign the Trained Model the population (Initial Population 1 pre-trained Model)

    print("The current working directory is: %s" %(os.getcwd()))

    os.chdir('Experiments_Neurips/Final_PTB_Experiments/PTB_Data')

    fields = ["Model Number", "Model Name", "Accuracy on Training Data", "Accuracy of testing data"]

    ############################################# Change the file name ###############################

    with open('log_QDTL_PTB.csv', 'a') as f:
      writer = csv.writer(f)
      writer.writerow(fields)

    json_file = open('Ed_L_fold5D_model_G2b.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model1_0 = model_from_json(loaded_model_json)

    model1_0.load_weights("Ed_L_fold5D_model_G2b_weights.h5")
    print(model1_0.load_weights)
    print("Loaded model from disk")
    
    RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001,decay=0.0,clipnorm=1.0, clipvalue=0.5) 


    model1_0.compile(optimizer = RMSprop, loss = 'mean_squared_error')

    networks = []
    networks.append(model1_0)
    population_names=[]
    population_names.append('model1_0')
    average_accuracy=0

    model_accuracy=[]
    accuracy_first_model = fitness(model1_0)
    model_accuracy.append(accuracy_first_model)



    networks_diversity = []
    networks_diversity.append(model1_0)
    population_names_diversity=[]
    population_names_diversity.append('model1_0')
    average_accuracy_diversity=0

    model_accuracy_diversity=[]
    accuracy_first_model_diversity = fitness_diversity(model1_0)
    model_accuracy_diversity.append(accuracy_first_model_diversity)

    for generation_num in range(1, generations+1):

      print("Generation %d started!!" %(generation_num))

      networks, population_names, model_accuracy, networks_diversity, population_names_diversity, model_accuracy_diversity = evolve(networks, generation_num, population_names, model_accuracy,networks_diversity, population_names_diversity, model_accuracy_diversity)

      # print("The fitness of the 1st Model is: ", fitness(networks[0]))

      networks, model_accuracy = remove_duplicated_models(networks, generation_num, model_accuracy)
      networks_diversity, model_accuracy_diversity=remove_duplicated_models_diversity(networks_diversity, generation_num, model_accuracy_diversity)
      
      average_accuracy = train_networks_and_accuracy(networks, population_names, model_accuracy, generation_num)

      fields = ["---------------", "-----------------", "------------------------------------", "------------------------------------"]

      #################################### Change the file name #######################################

      with open('evaluate_QDTL.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)



      print("Generation %d is over!!" %(generation_num))



      print("###############################################################################################################")

    print("All Evolutionary Generations are over!!")



    print("Only wanted model are saved reamaining are deleted!!")

generations=20
n_population=10



generate(generations, n_population)

#Executing top 10 models saved after running mutation and selecting the one with lowest MSE on train set
with open('model18_7.json') as f:
    model = model_from_json(f.read())
model.load_weights('model18_7.h5')
RMSprop = tf.keras.optimizers.RMSprop(learning_rate=0.001,decay=0.0) 
model.compile(optimizer = RMSprop, loss = 'mean_squared_error')
model.evaluate(x_train, y_train, verbose=2)