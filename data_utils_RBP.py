#import hickle as hkl
import numpy as np
from keras import backend as K
from keras.preprocessing.image import Iterator
import matplotlib.pyplot as plt

# Called from kitti_train.py and kitti_evaluate.py.
# Class SequenceGenerator is a subclass of Iterator.
# Data generator that creates sequences for input into PredNet.
class SequenceGenerator(Iterator): # Iterator: can be iterated over in for-loop
    def __init__(self, data_file, source_file, nt,
                 batch_size=8, shuffle=False, seed=None,
                 output_mode='error', sequence_start_mode='all', N_seq=None,
                 data_format=K.image_data_format()):
        print("\ndata_utils_RPB.py: Instantiating sequence generator:\n")
        
        # LOAD DATA FILE
        print("data_utils_RBP.py: Data file: \n", data_file)
        self.X = np.load(data_file)  # X will be like (n_images, nb_cols, nb_rows, nb_channels) 
        #self.X =hkl.transpose(self.X, (0, 3, 2, 1))
        
# ===============================================================
# Added statements to print out two consecutive frames. ASM
        print("data_utils.py: self.X.shape\n", self.X.shape) # e.g., (41396, 128, 160, 3)
        # print("1st row:\n", self.X[0,:,:,:]) # will print the raw array
        # Print 1st two consecutive frames
        # NOTE: the video sequence seems to be stored in reverse order!!! Is this a bug?
        #       1. When called from "kitti_train.py" the frames for X_train.py and X_val.py seem
        #          to be stored in reverse order.
        #       2. When called from "kitti_evaluate.py" the frames for X_test.py seem to be
        #          in correct order.
        #       3. Need to make sure that the source files properly match the data files.
        my_temp = np.array(self.X[0,:,:,:], dtype = int, copy = True) # convert from float to int
        plt.imshow(my_temp) # look at 1st image
        plt.show()
        my_temp = np.array(self.X[1,:,:,:], dtype = int, copy = True) # convert from float to int
        plt.imshow(my_temp) # look at 2nd image
        plt.show()
        
        # LOAD SOURCE FILE
        print("data_utils.py: Source file: \n", source_file)
        self.sources = np.load(source_file) # Labels in b'string' format
        # Above: source for each image so when creating sequences can assure that consecutive 
        #        frames are from same video
        print("data_utils.py: self.sources.shape\n", self.sources.shape) # e.g., (41396,)
        print(self.sources[0]) # should print a byte literal representation of a string
# End of print statements
# ===============================================================

        # SET OTHER PARAMS
        self.nt = nt                     # 10
        self.batch_size = batch_size     #  4 if called from "kitti_train.py"
        self.data_format = data_format   # K.image_data_format()
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'
        self.sequence_start_mode = sequence_start_mode # default is 'all'
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        self.output_mode = output_mode   # default is 'error'

        if self.data_format == 'channels_first': # tensorflow data format is 'channels_last'
            self.X = np.transpose(self.X, (0, 3, 1, 2))
        self.im_shape = self.X[0].shape  # (128, 160, 3)  I think.

        if self.sequence_start_mode == 'all':  # allow for any possible sequence, starting from any frame
            self.possible_starts = np.array([i for i in range(self.X.shape[0] - self.nt) if self.sources[i] == self.sources[i + self.nt - 1]])
            print("data_utils.py: possible_starts all: ", self.possible_starts)
        elif self.sequence_start_mode == 'unique':  #create sequences where each unique frame is in at most one sequence
            curr_location = 0
            possible_starts = []
            while curr_location < self.X.shape[0] - self.nt + 1:
                if self.sources[curr_location] == self.sources[curr_location + self.nt - 1]:
                    possible_starts.append(curr_location)
                    curr_location += self.nt
                else:
                    curr_location += 1
            self.possible_starts = possible_starts
            print("data_utils.py: possible_starts unique: ", self.possible_starts)

        if shuffle:
            self.possible_starts = np.random.permutation(self.possible_starts)
        if N_seq is not None and len(self.possible_starts) > N_seq:  # select a subset of sequences if want to
            self.possible_starts = self.possible_starts[:N_seq]
        self.N_sequences = len(self.possible_starts)
        super(SequenceGenerator, self).__init__(len(self.possible_starts), batch_size, shuffle, seed)
    # End of __init__()

    def __getitem__(self, null):
        return self.next()

    def next(self): # Returns a batch of x and y data
        with self.lock:
            current_index = (self.batch_index * self.batch_size) % self.n
            index_array, current_batch_size = next(self.index_generator), self.batch_size
        batch_x = np.zeros((current_batch_size, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(index_array):
            idx = self.possible_starts[idx]
            batch_x[i] = self.preprocess(self.X[idx:idx+self.nt])
        if self.output_mode == 'error':  # model outputs errors, so y should be zeros
            batch_y = np.zeros(current_batch_size, np.float32)
        elif self.output_mode == 'prediction':  # output actual pixels
            batch_y = batch_x
        return batch_x, batch_y  # inputs, targets

    def preprocess(self, X):
        return X.astype(np.float32) / 255 # maps to [0, 1]

    # Returns 10 frames
    def create_all(self):
        # Below: plus operator is concatentation. Initialize multidim array of float32 zeros w/ specified shape
        X_all = np.zeros((self.N_sequences, self.nt) + self.im_shape, np.float32)
        for i, idx in enumerate(self.possible_starts):
            X_all[i] = self.preprocess(self.X[idx:idx+self.nt]) # map [0,255] to [0,1] for 10 frames
        return X_all
