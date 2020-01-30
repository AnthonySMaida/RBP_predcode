
'''
Train PredNet on KITTI sequences. (Geiger et al. 2013, http://www.cvlibs.net/datasets/kitti/)
'''

#from numpy.random import seed
#seed(1)
#from tensorflow import set_random_seed
#set_random_seed(2)

import os
import pprint
import pickle
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)

from keras import backend as K  # Actual backend is specified in $HOME/.keras/keras.json, or environment var KERAS_BACKEND
from keras.models import Model
from keras.layers import Input, Dense, Flatten, Softmax
from keras.layers import TimeDistributed
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.constraints import UnitNorm

from prednet_RBP import PredNet_RBP
from data_utils_RBP import SequenceGenerator # defines train_generator and val_generator
from kitti_settings import DATA_DIR, WEIGHTS_DIR

#from kitti_settings import *

save_model = True  # if weights will be saved
weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')  # where weights will be saved
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')       # where model will be saved

#train_file = os.path.join(DATA_DIR, 'X_train.npy')
#train_sources = os.path.join(DATA_DIR, 'source_train.hkl')
#val_file = os.path.join(DATA_DIR, 'X_val.npy')
#val_sources = os.path.join(DATA_DIR, 'source_val.hkl')

#DATA_DIR='D:\datasets\kitti\data' # convert to Unix
# .npy stands for numpy array file
#DATA_DIR='/Users/maida/Desktop/2019_Matin_Rao_model_keras/data'
train_file    = os.path.join(DATA_DIR, 'X_train.npy')
train_sources = os.path.join(DATA_DIR, 'source_train.npy')
val_file      = os.path.join(DATA_DIR, 'X_val.npy')
val_sources   = os.path.join(DATA_DIR, 'source_val.npy')



#import keras
import tensorflow as tf
#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)
#with device("/cpu:0"):
config = tf.ConfigProto() # for configuring the GPU
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Training parameters
nb_epoch   = 50
batch_size = 4
samples_per_epoch = 500
N_seq_val  = 100  # number of sequences to use for validation

# Model parameters
n_channels, im_height, im_width = (3, 128, 160)
# if keras.backend.backend() == 'tensorflow', then image_data_format is 'channels_last'
input_shape = (n_channels, im_height, im_width) if K.image_data_format() == 'channels_first' else (im_height, im_width, n_channels)

# PARAMETERS FOR 2-LAYER NETWORK
#===============================
stack_sizes   = (n_channels, 3) # n_channels == 3 from line 68.
R_stack_sizes = (3, 12)
A_filt_sizes  = (3, 3)         # length == len(stack_sizes)
Ahat_filt_sizes = (3, 3)       # length == len(stack_sizes)
R_filt_sizes  = (3, 3)         # 3x3 filters for 2 layers
# weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = np.array([0.5, 0.5])  
#===============================

# PARAMETERS FOR 3-LAYER NETWORK
#===============================
stack_sizes   = (n_channels, 3, 12) # n_channels == 3 from line 68.
# R_stack_sizes = (3, 12, 24)
# A_filt_sizes  = (3, 3, 3)        # length == len(stack_sizes)
# Ahat_filt_sizes = (3, 3, 3)   # length == len(stack_sizes)
# R_filt_sizes  = (3, 3, 3)     # 3x3 filters for 2 layers
# # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
# layer_loss_weights = np.array([1, 0, 0])  
#===============================

# PARAMETERS FOR 4-LAYER NETWORK
# =============================================================================
# stack_sizes = (n_channels, n_channels, 48, 96)
# R_stack_sizes = (n_channels, 48, 96, 192)
# A_filt_sizes = (3, 3, 3, 3)     # Not used.
# Ahat_filt_sizes = (1, 1, 1 ,1)  # length == len(stack_sizes)
# R_filt_sizes = (3, 3, 3, 3)     # 3x3 filters for 2 layers
# # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
# layer_loss_weights = np.array([0.6, 0.2, 0.1, 0.1])  
# =============================================================================


layer_loss_weights = np.expand_dims(layer_loss_weights, 1) # change shape from (2,) to (2,1)
nt = 10  # number of timesteps used for sequences in training

# equally weight all timesteps except the first
# b/c the first will be the default prediction?
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))  
time_loss_weights[0] = 0

# BUILD the model
#================
#================

# The 'prednet' variable is bound to an instance of PredNet class
print("\nkitti_train_RBP.py: Building prednet_RBP w/ the params below:")
print("stack_sizes   == ", stack_sizes)
print("R_stack_sizes == ", R_stack_sizes)
print("A_filt_sizes  == ", A_filt_sizes)
print("R_filt_sizes  == ", R_filt_sizes)
# Below: build PredNet_RBP instance
# Apparently defines a layer
prednet_RBP = PredNet_RBP(stack_sizes,                     # (3, 3) Nb of output channels in A and Ahat for each layer
                          R_stack_sizes,                   # (3, 48) Nb of output channels in R for each layer
                          A_filt_sizes,                    # (3, 3)  Size of single A filter in Matin's 2-layer model
                          Ahat_filt_sizes,                 # (3, 3)  Size of Ahat filters.
                          R_filt_sizes,                    # (3, 3)  Size of R filters.
                          output_mode='error', 
                          return_sequences=True)

# START BUILDING THE FUNCTIONAL MODEL
#====================================
# Start with the Input layer
# Below: declares an input layer but does not provide input data.
# input-shape = (128, 160, 3) if using tensorflow backend.
# When concatentated w/ (nt,), yields (10, 128, 160, 3).`
#        Data is not given to the model until .fit_generator() is called.
inputs = Input(shape=(nt,) + input_shape) # returns tensor w/ shape=(?,10,128,160,3) dtype = float32
                                          # (batch_sz, nt, height, width, input channels)
                                          
print("\nkitti_train_RBP.py inputs: ", inputs)
print("                                   (?, nt, hght, wid, chans)")

# ADD THE PREDNET LAYER
#======================
print("\n------------------------------------------------------------------")
print("Starting to execute: prednet_RBP(inputs) from kitti_train_RBP.py")
# Below: from prednet_RBP.py calls: build(), get_initial_state(), step(), step(), compute_output_shape()
errors = prednet_RBP(inputs)  # errors will be (batch_size, nt, nb_layers)
print("Finished executing: prednet_RBP(inputs) from kitti_train_RBP.py")
print("------------------------------------------------------------------")
print("\nkitti_train_RBP.py errors: ", errors)

#====================================
# Add a TimeDistributed, DENSE layer.
#====================================
#first_dense_layer = Dense(1, 
#                          kernel_constraint=UnitNorm(), 
#                          # W_constraint=UnitNorm(), # Don't use: legacy
#                          # weights=[layer_loss_weights, np.zeros(1)], 
#                          trainable=False)
#errors_by_time0 = TimeDistributed(first_dense_layer, 
#                                  weights=[layer_loss_weights, np.zeros(1)], 
#                                  trainable=True)(errors)  
#==================================== The above is a failed experiment
#
errors_by_time = TimeDistributed(Dense(1, trainable=False, kernel_constraint=UnitNorm()), 
                                 weights=[layer_loss_weights, np.zeros(1)], 
                                 trainable=False)(errors)  
# Flatten and then add another DENSE layer.
errors_by_time = Flatten()(errors_by_time)  # will be (batch_size, nt)

print("\nkitti_train_RBP.py errors_by_time: ", errors_by_time)

final_errors = Dense(1, weights=[time_loss_weights, np.zeros(1)], trainable=False)(errors_by_time)  
# Above: weight errors by time
print("\nkitti_train_RBP.py final_errors: ", final_errors)


# CREATE the MODEL
model = Model(inputs=inputs, outputs=final_errors) # Using functional API.

# Print a SUMMARY of the model
plot_model(model, to_file='model_diagram.png', show_shapes=True)
print("\nModel summary below:")
model.summary()
model_wts_list   = model.get_weights()
model_layer_dict = dict([(layer.name, layer) for layer in model.layers])
print("weight matrix shapes:")
for wt_mat in model_wts_list:
    print("    :", wt_mat.shape)
print("model_wts_list[-5]: ", model_wts_list[-5])
print("model_wts_list[-4]: ", model_wts_list[-4])
print("model_wts_list[-3]: ", model_wts_list[-3])
print("model_wts_list[-2]: ", model_wts_list[-2])
print("model_wts_list[-1]: ", model_wts_list[-1])
print("model_layer_dict: ", model_layer_dict)
#print("time_distributed_1 weights BEFORE training:\n ", model_layer_dict['time_distributed_1'].get_weights())


#===============================================
# COMPILE the MODEL
# Tell the optimizer to use mean_absolute_error.
model.compile(loss='mean_absolute_error', optimizer='adam', metrics = ['mse', 'mae']) # configure model for training
print("\nkitty_train_RBP.py: model has been compiled.\n")
 
# SequenceGenerator() is defined in data_utils_RPB.py
#    train_file:      X_train.npy
# train_sources: source_train.npy
#    batch_size:                4
#            nt:               10
#                                   X_train.npy  source_train.npy, 10
train_generator = SequenceGenerator(train_file, train_sources, nt, batch_size=batch_size, shuffle=True)
#                                   X_val.npy  source_val.npy, 10
val_generator   = SequenceGenerator(val_file, val_sources, nt, batch_size=batch_size, N_seq=N_seq_val)
print("\nkitty_train.py: Sequence generators completed.")

lr_schedule = lambda epoch: 0.001 if epoch < 75 else 0.0001   # start with lr of 0.001 and then drop to 0.0001 after 75 epochs
callbacks = [LearningRateScheduler(lr_schedule)]              # Must be a list. Initially contains 1 element.

# SAVE MODEL
#===========
# Use a callback to save the model.
if save_model:
    if not os.path.exists(WEIGHTS_DIR): os.mkdir(WEIGHTS_DIR)
    # weights_file: 'prednet_kitti_weights.hdf5'
    callbacks.append(ModelCheckpoint(filepath=weights_file, 
                                     verbose=1,
                                     monitor='val_loss', 
                                     save_best_only=True))

print("\nkitty_train.py number of callbacks: ", len(callbacks))

# START TRAINING
#===============
# samples_per_epoch: 500
# nb_epoch:            5
# batch_size:          4
print("\nkitty_train.py: Starting to fit model.")
# trains the model on data generated batch-by-batch by a generator
history = model.fit_generator(train_generator,                  # Deined in line 162. inputs, targets
                              samples_per_epoch / batch_size,   # steps_per_epoch == 125
                              nb_epoch,                         #   5
                              callbacks=callbacks,
                              validation_data=val_generator,    # defined in line 164
                              validation_steps=N_seq_val / batch_size,
                              verbose=1)

model_wts_list_after   = model.get_weights()
print("model_wts_list_after[-5]: ", model_wts_list_after[-5])
print("model_wts_list_after[-4]: ", model_wts_list_after[-4])
print("model_wts_list_after[-3]: ", model_wts_list_after[-3])
print("model_wts_list_after[-2]: ", model_wts_list_after[-2])
print("model_wts_list_after[-1]: ", model_wts_list_after[-1])
print("time_distributed_1 weights AFTER training:\n ", model_layer_dict['time_distributed_1'].get_weights())
print("\nHistory:\n ")
pp = pprint.PrettyPrinter(indent = 1)
pp.pprint(history.history)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0.0, 0.1)
plt.title('Mean Absolute Error')  # See line 202: model.compile()
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()

plt.figure()
plt.plot(history.history['mse'])
plt.plot(history.history['val_mse'])
plt.ylim(0.0, 0.04)
plt.title('Mean Squared Error')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()

# save history data
history_out = open("trainHistory.pickle", "wb")
pickle.dump(history.history, history_out)
history_out.close()

if save_model:  # is True for this configuration
    json_string = model.to_json()  # convert the model to a JSON string
    # prednet_kitti_weights.json
    with open(json_file, "w") as f:
        f.write(json_string)

