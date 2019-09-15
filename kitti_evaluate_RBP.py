'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import h5py
import numpy as np
from six.moves import cPickle
import matplotlib
# 'Agg' backend only renders PNGs.
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten
from keras.utils import HDF5Matrix

from prednet_RBP import PredNet_RBP
from data_utils_RBP import SequenceGenerator
from kitti_settings import WEIGHTS_DIR, DATA_DIR, RESULTS_SAVE_DIR

from skimage import measure

import tensorflow as tf
#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)
#with device("/cpu:0"):
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


n_plot = 40
batch_size = 10
nt = 10         # number of time steps used for sequences. Also defined in kitti_train_RBP.py

# The model and weights are stored on separate files.
# wts are stored on an HDF5 file that is good for storing multidimensional arrays.
weights_file_nm = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file_nm    = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
test_file_nm    = os.path.join(DATA_DIR, 'X_test.npy')
#test_sources = os.path.join(DATA_DIR, 'source_test.hkl')
test_sources = os.path.join(DATA_DIR, 'source_test.npy')


# Load trained model
f = open(json_file_nm, 'r')
json_string = f.read()
f.close()

# =============================================================================
# f1 = h5py.File(weights_file_nm,'r')
# print("Keys: %s" % f1.keys())
# print("Model weights:", f1['model_weights'])
# print("Model weights length:", len(f1['model_weights']))
# print("Model weights type:", type(f1['model_weights']))
# f1_wts_dict = f1['model_weights'].__dict__
# print("f1['model_weights'].__dict__:", f1_wts_dict)
# #for i in range(len(f1['model_weights'])):
# #    print("    ", f1['model_weights'][i])
# 
# =============================================================================
# x_data = HDF5Matrix(weights_file_nm, 'model_weights') # doesn't work

# Reconstruct the trained model by loading .json file and .hdf5 file.
#====================================================================
print("json_model_string:", json_string)
trained_model = model_from_json(json_string, custom_objects = {'PredNet_RBP': PredNet_RBP}) # loads a model from a file holding a JSON string
print("\nTrained model summary below:")
trained_model.summary()
trained_model.load_weights(weights_file_nm)
# Note: things may go faster if the model is compiled

# Create testing model (to output predictions)
print("\nTrained model layers: ", trained_model.layers)
# Above printout indicates 2nd list element describes PredNet architecture.
layer_config = trained_model.layers[1].get_config() # what does this line do?
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
print("\nFinal layer_config: ", layer_config)

# Instantiate PredNet
#====================
# Below: unclear how the args in this line work. Must inherit from Recurrent superclass (no documentation)
test_prednet_RBP = PredNet_RBP(weights=trained_model.layers[1].get_weights(), **layer_config) # create new prednet instance
input_shape = list(trained_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet_RBP(inputs)
test_model = Model(inputs=inputs, outputs=predictions) # compare w/ kitti_train_RBP.py Model(inputs=inputs, outputs=predictions)

# Below: SequenceGenerator is defined in data_utils_RBP.py
#                            Args: X_test.npy, source_test.npy, 10
test_generator = SequenceGenerator(test_file_nm, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
X_test = test_generator.create_all() # Returns 10 frames of test data
X_hat = test_model.predict(X_test, batch_size) # Run the test data through the model to get predictions
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat,   (0, 1, 3, 4, 2))

# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
#=================================================================================================
# Below: X_test is 5-dimensional, e.g., (?, 10, 128, 160, 3)
#        2nd entry is number of frames (10).
#        "X_test[:, 1:]" removes 1st video frame.
print("\nX_test.shape:        ", X_test.shape)
print("X_test[:, 1:].shape: " , X_test[:, 1:].shape)  # leave out first frame
print("X_test[:, :-1].shape:" , X_test[:, :-1].shape) # leave out last frame
print("\nX_hat.shape:         ", X_hat.shape)
print("X_hat[:, 1:].shape:  " , X_hat[:, 1:].shape)   # leave out first frame

# Below: compute mean SSE
#        Using elementwise operations on 5D matrices to produce a scalar.
mse_model  = np.mean( (X_test[:, 1:] - X_hat[:, 1:])**2 )    # measures quality of model's one-step predictions
mae_model  = np.mean(np.abs(X_test[:, 1:] - X_hat[:, 1:]))
ssim_model = measure.compare_ssim(X_test[:,1:],X_hat[:,1:], win_size=3, multichannel=True)
mse_prev   = np.mean( (X_test[:, :-1] - X_test[:, 1:])**2 )  # measures quality of data's one-step predictions (baseline)
mae_prev   = np.mean(np.abs(X_test[:, :-1] - X_test[:, 1:]))
ssim_prev  = measure.compare_ssim(X_test[:, :-1], X_test[:, 1:], win_size=3, multichannel=True)
  
if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
f.write("Model MSE : %f\n" % mse_model)
f.write("Model MAE : %f\n" % mae_model)
f.write("Model SSIM: %f\n" % ssim_model)
f.write("Previous Frame MSE : %f\n" % mse_prev)
f.write("Previous Frame MAE : %f\n" % mae_prev)
f.write("Previous Frame SSIM: %f\n" % ssim_prev)
f.close()
print("\nMSE results saved on: prediction_scores.txt")

# Plot some predictions
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3] # 128/160
plt.figure(figsize = (nt, 2*aspect_ratio)) # How do we increase rendered image size? Too small to be useful.
gs = gridspec.GridSpec(2, nt)              # Dimension is 2 rows x 10 cols.
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot] # Select 40 random plots. value of n_plot is 40
print("Value of plot_idx: ", plot_idx)
for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X_test[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

#        print(X_hat[i,t])
        plt.subplot(gs[t + nt])
        X_hat[i,t] = np.minimum(X_hat[i,t], 1.0) # make sure max val in X_hat is <= 1.0. X_test was normalized in data_utiles.py
        plt.imshow(X_hat[i,t], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)
    print("Saving: plot_"+str(i)+".png")
    plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
    plt.clf()
