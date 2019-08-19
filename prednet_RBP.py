import numpy as np

from keras import backend as K
from keras import activations
from keras.layers import Recurrent
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from keras.engine import InputSpec
from keras_utils import legacy_prednet_support

# Based on Lotter et al: https://github.com/coxlab/prednet
# Modifications by Anthony S. Maida and Matin Hosseini

# Defines class PredNet, inherits from Recurrent layer class.
# It has an internal state and a step function.
# In keras.legacy.layers, class Recurrent is a subclass of Layer.
class PredNet_RBP(Recurrent):
    '''PredNet architecture - Lotter 2016.
        Stacked convolutional LSTM inspired by predictive coding principles.

    # Arguments
        STACK SIZES REMOVED FROM RPB MODEL: Replaced by Ahat_stack_sizes and A_stack_sizes.
        stack_sizes: number of channels in targets (A) and predictions (Ahat) in each layer of the architecture.
            Length is the number of layers in the architecture.
            First element is the number of channels in the input.
            Ex. (3, 16, 32) would correspond to a 3 layer architecture that takes in RGB images and has 16 and 32
                channels in the second and third layers, respectively.
        R_stack_sizes: number of channels in the representation (R) modules.
            Length must equal length of stack_sizes, but the number of channels per layer can be different.
        Ahat_stack_sizes: number of output channels in prediction modules (Ahat) for each layer.
        A_stack_sizes:    number of output channels in A modules for each layer.
        A_filt_sizes: filter sizes for the target (A) modules.
            Has length of 1 - len(stack_sizes).
            Ex. (3, 3) would mean that targets for layers 2 and 3 are computed by a 3x3 convolution of the errors (E)
                from the layer below (followed by max-pooling)
        Ahat_filt_sizes: filter sizes for the prediction (Ahat) modules.
            Has length equal to length of stack_sizes.
            Ex. (3, 3, 3) would mean that the predictions for each layer are computed by a 3x3 convolution of the
                representation (R) modules at each layer.
        R_filt_sizes: filter sizes for the representation (R) modules.
            Has length equal to length of stack_sizes.
            Corresponds to the filter sizes for all convolutions in the LSTM.
        pixel_max: the maximum pixel value.
            Used to clip the pixel-layer prediction.
        error_activation: activation function for the error (E) units.
        A_activation: activation function for the target (A) and prediction (A_hat) units.
        LSTM_activation: activation function for the cell and hidden states of the LSTM.
        LSTM_inner_activation: activation function for the gates in the LSTM.
        output_mode: either 'error', 'prediction', 'all' or layer specification (ex. R2, see below).
            Controls what is outputted by the PredNet.
            If 'error', the mean response of the error (E) units of each layer will be outputted.
                That is, the output shape will be (batch_size, nb_layers).
            If 'prediction', the frame prediction will be outputted.
            If 'all', the output will be the frame prediction concatenated with the mean layer errors.
                The frame prediction is flattened before concatenation.
                Nomenclature of 'all' is kept for backwards compatibility, 
                but should not be confused with returning all of the layers of the model
            For returning the features of a particular layer, output_mode should be of the form unit_type + layer_number.
                For instance, to return the features of the LSTM "representational" units in the lowest layer, output_mode should be specificied as 'R0'.
                The possible unit types are 'R', 'Ahat', 'A', and 'E' corresponding to the 'representation', 'prediction', 'target', and 'error' units respectively.
        extrap_start_time: time step for which model will start extrapolating.
            Starting at this time step, the prediction from the previous time step will be treated as the "actual"
        data_format: 'channels_first' or 'channels_last'.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.

    # References
        - [Deep predictive coding networks for video prediction and unsupervised learning](https://arxiv.org/abs/1605.08104)
        - [Long short-term memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
        - [Convolutional LSTM network: a machine learning approach for precipitation nowcasting](http://arxiv.org/abs/1506.04214)
        - [Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects](http://www.nature.com/neuro/journal/v2/n1/pdf/nn0199_79.pdf)
    '''
    # Note: Symbol 'legacy_prednet_support' defined locally in keras_utils.py
    # Below is a decorator
    # Decorators provide a simple syntax for calling higher-order functions
    #
    # Note: The file "kitti_evaluate_RPB.py" invokes the PredNet constructor using a
    #       keyword arg "weights" which is not defined for this __init__().
    #       I presume "weights" is defined in the superclass "Recurrent".
    #       Unfortunately I can't fine the legacy documentation for "Recurrent" to check this.
    @legacy_prednet_support
    def __init__(self, stack_sizes, R_stack_sizes, # Ahat_stack_sizes, A_stack_sizes,
                 A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                 pixel_max=1., error_activation='relu', A_activation='relu',
                 LSTM_activation='tanh', LSTM_inner_activation='hard_sigmoid',
                 output_mode='error', extrap_start_time=None,
                 data_format=K.image_data_format(), **kwargs):
        print("\n------------------------------------------------------------------")
        print("prednet_RBP_28June2019.py: '__init__()' called")
        print("\n------------------------------------------------------------------")
        self.stack_sizes = stack_sizes
#        self.Ahat_stack_sizes = Ahat_stack_sizes
#        self.A_stack_sizes = A_stack_sizes
        self.nb_layers = len(R_stack_sizes)
        assert len(R_stack_sizes) == self.nb_layers, 'len(R_stack_sizes) must equal len(stack_sizes)'
        self.R_stack_sizes = R_stack_sizes
        assert len(A_filt_sizes) == (self.nb_layers), 'len(A_filt_sizes) must equal len(stack_sizes)' # for RBP model
        self.A_filt_sizes = A_filt_sizes
        assert len(Ahat_filt_sizes) == self.nb_layers, 'len(Ahat_filt_sizes) must equal len(stack_sizes)'
        self.Ahat_filt_sizes = Ahat_filt_sizes
        assert len(R_filt_sizes) == (self.nb_layers), 'len(R_filt_sizes) must equal len(stack_sizes)'
        self.R_filt_sizes = R_filt_sizes

        self.pixel_max = pixel_max                                          # 1
        self.error_activation = activations.get(error_activation)           # relu
        self.A_activation = activations.get(A_activation)                   # relu
        self.LSTM_activation = activations.get(LSTM_activation)             # tanh
        self.LSTM_inner_activation = activations.get(LSTM_inner_activation) # hard_sigmoid

        # check if supplied output mode is legal & assign to self.output_mode
        default_output_modes = ['prediction', 'error', 'all']
        # if nb_layers == 2, then returns
        # [R0, E0, A0, Ahat0, R1, E1, A1, Ahat1]
        layer_output_modes = [layer + str(n) for n in range(self.nb_layers) for layer in ['R', 'E', 'A', 'Ahat']]
        # plus operation below concatentates lists
        assert output_mode in default_output_modes + layer_output_modes, 'Invalid output_mode: ' + str(output_mode)
        self.output_mode = output_mode
        if self.output_mode in layer_output_modes:   # e.g.: breaks "R1" into "R" and "1"
            self.output_layer_type = self.output_mode[:-1]
            self.output_layer_num = int(self.output_mode[-1])
        else:
            self.output_layer_type = None # these properties are not meaningful for default_output_modes
            self.output_layer_num = None
            
        # time step for which model will start extrapolating
        self.extrap_start_time = extrap_start_time

        #      element     in set
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {channels_last, channels_first}'
        self.data_format = data_format # if TensorFlow, then 'channels_last'
        self.channel_axis = -3 if data_format == 'channels_first' else -1
        self.row_axis = -2 if data_format == 'channels_first' else -3
        self.column_axis = -1 if data_format == 'channels_first' else -2
        super(PredNet_RBP, self).__init__(**kwargs) # boiler plate
        self.input_spec = [InputSpec(ndim=5)]   # why is this line after super?
                                                # ndim: expected rank of the input

    # COMPUTE_OUTPUT_SHAPE
    #=====================
    # needed for any custom layer
    # let's keras do automatic shape inference
    def compute_output_shape(self, input_shape):
        print("\n------------------------------------------------------------------")
        print("\033[91mprednet_RBP_28June2019.py: 'compute_output_shape()' called \033[00m")
        print("         input_shape:", input_shape)
        print("    self.output_mode:", self.output_mode)
        if self.output_mode == 'prediction':
            out_shape = input_shape[2:]
        elif self.output_mode == 'error':
            out_shape = (self.nb_layers,)
        elif self.output_mode == 'all':
            out_shape = (np.prod(input_shape[2:]) + self.nb_layers,)
        else:
            stack_str = 'R_stack_sizes' if self.output_layer_type == 'R' else 'stack_sizes'
            stack_mult = 2 if self.output_layer_type == 'E' else 1
            out_stack_size = stack_mult * getattr(self, stack_str)[self.output_layer_num]
            # IS THIS RIGHT!!!  Added "- 1" to the two lines below.
            out_nb_row = input_shape[self.row_axis] / 2**(self.output_layer_num - 1)       # RBP model: no downsampling in 1st layer
            out_nb_col = input_shape[self.column_axis] / 2**(self.output_layer_num - 1)    # RBP model: no downsampling in 1st layer
            if self.data_format == 'channels_first':
                out_shape = (out_stack_size, out_nb_row, out_nb_col)
            else:
                out_shape = (out_nb_row, out_nb_col, out_stack_size)
        print("            out_shape:", out_shape)
        print("self.return_sequences:", self.return_sequences)
        if self.return_sequences:
            return (input_shape[0], input_shape[1]) + out_shape # (batch_size, nt, nb_layers) == (None, 10, 2)
        else:
            return (input_shape[0],) + out_shape    # (batch_size, nb_layers) == (None, 2)

    # GET_INITIAL_STATE
    #==================
    def get_initial_state(self, x):
        print("\n------------------------------------------------------------------")
        print("\033[91mprednet_RBP_28June2019.py: 'get_initial_state()' called \033[00m")
        input_shape = self.input_spec[0].shape
        init_nb_row = input_shape[self.row_axis]
        init_nb_col = input_shape[self.column_axis]
        print("              x: ", x)
        print("        x.shape: ", x.shape)
        print("    input_shape: ", input_shape)
        print("    init_nb_row: ", init_nb_row)
        print("    init_nb_col: ", init_nb_col)

        base_initial_state = K.zeros_like(x)  # (samples, timesteps) + image_shape
        print("    Initial base_initial_state.shape: ", base_initial_state.shape)
        non_channel_axis = -1 if self.data_format == 'channels_first' else -2
        for _ in range(2):
            base_initial_state = K.sum(base_initial_state, axis=non_channel_axis)
        base_initial_state = K.sum(base_initial_state, axis=1)  # (samples, nb_channels)
        print("    Final base_initial_state: ", base_initial_state)
        print("    Final base_initial_state: ", base_initial_state.shape)

        initial_states = []
        states_to_pass = ['r', 'c', 'e']
        nlayers_to_pass = {u: self.nb_layers for u in states_to_pass}
        # Above: returns "{'r': 2, 'c': 2, 'e': 2}" if two layers
        if self.extrap_start_time is not None:
           states_to_pass.append('ahat')  # pass prediction in states so can use as actual for t+1 when extrapolating
           nlayers_to_pass['ahat'] = 1
        print("    Calculate stack and output sizes")
        for u in states_to_pass: # iterate over ['r', 'c', 'e']
            for l in range(nlayers_to_pass[u]): # the value will always be the nb of layers in the network
                print("\n                  layer:" ,l)
                ds_factor = 2 ** l
                nb_row = init_nb_row // ds_factor
                nb_col = init_nb_col // ds_factor
                if u in ['r', 'c']:
                    stack_size = self.R_stack_sizes[l]
                elif u == 'e':
                    stack_size = 2 * self.stack_sizes[l]
                elif u == 'ahat':
                    stack_size = self.stack_sizes[l]
                print("        state component:", u)
                print("             stack_size:", stack_size)
                output_size = stack_size * nb_row * nb_col  # flattened size
                print("            output_size:", output_size)

                reducer = K.zeros((input_shape[self.channel_axis], output_size)) # (nb_channels, output_size)
                initial_state = K.dot(base_initial_state, reducer) # (samples, output_size)
                if self.data_format == 'channels_first':
                    output_shp = (-1, stack_size, nb_row, nb_col)
                else:
                    output_shp = (-1, nb_row, nb_col, stack_size)
                initial_state = K.reshape(initial_state, output_shp)
                print("          initial_state: ", initial_state, " for l=", l)
                initial_states += [initial_state]

        if K._BACKEND == 'theano':
            from theano import tensor as T
            # There is a known issue in the Theano scan op when dealing with inputs whose shape is 1 along a dimension.
            # In our case, this is a problem when training on grayscale images, and the below line fixes it.
            initial_states = [T.unbroadcast(init_state, 0, 1) for init_state in initial_states]

        if self.extrap_start_time is not None:
            initial_states += [K.variable(0, int if K.backend() != 'tensorflow' else 'int32')]  # the last state will correspond to the current timestep
        print("\nRETURNING from get_initial_state()")
        for i in range(len(initial_states)):
            print("        ", initial_states[i])
        print("States length:", len(initial_states))
        return initial_states  # return type is list

    # BUILD
    #======
    # Class 'InputSpec' specifies the ndim, dtype, and shape of every input to a layer
    # Signature: 'build(self, input_shape)' is necessary.
    # Doesn't have a return statement.
    def build(self, input_shape):
        print("\n------------------------------------------------------------------")
        print("prednet_RBP_28June2019.py: 'build()' called")
        print("Input shape == ", input_shape)
        self.input_spec = [InputSpec(shape=input_shape)] # boiler plate for build, but not sure how input_shape is determined
        # Above: input_spec is [InputSpec(shape=(None, 10, 128, 160, 3), ndim=5)]
        
        # Below returns dictionary of empty lists: {'i': [], 'f': [], 'c': [], 'o': [], 'a': [], 'ahat': []}
        # The lists will hold Conv2D components used in the model sorted by layer within each list.
        # i, f, c, and o are all cLSTM components.
        
        # Below: leave out A module. Not needed in RBP model.
        #self.conv_layers = {c: [] for c in ['i', 'f', 'c', 'o', 'a', 'ahat']}
        # Below: revised code w/o A module
        self.conv_layers = {c: [] for c in ['i', 'f', 'c', 'o', 'ahat']}
        # Initial dictionary should look like:
        # {'i': [], 'f': [], 'c': [], 'o': [], ahat': []}
      
        
        #self.fullyconnected=K.Dense(10,activation=K.Sigmoid, use_bias=True)
        #self.flattened=K.flatten()
        
        # BUILD LOOP
        #===========
        # Build network, layer by layer, starting from layer 0.
        # For current model, the number of layers is 2, numbered: 0, 1.
        print("\nStarting BUILD loop:")
        print("Iterating over layers.")
        for l in range(self.nb_layers): # starting w/ layer 0
            print("")
            # start by building convolution objects within cLSTM for layer l.
            for c in ['i', 'f', 'c', 'o']: # iterate over gates within a layer. input, forget, update, output
                # Gate codes:
                #             i: input gate   (hardsig act)
                #             f: forget gate  (hardsig act)
                #             c: input update (tanh act)
                #             o: output gate  (hardsig act)
                # assign correct activation function for gate
                #     'tanh'                                'hard_sigmoid'
                act = self.LSTM_activation if c == 'c' else self.LSTM_inner_activation # inner_act means gate act
                # 1. conv_layers dict is defined at beginning of this function
                # 2. adds Conv2D objects to the appropriate gate in the dictionary
                # 3. order of elts in key value is determined by l
                # 4. Calling format:
                #    layers.Conv2d(nb of output channels, kernel size, *kwargs)
                conv2D_temp_obj = Conv2D(self.R_stack_sizes[l], # indexed by l: (3, 48). Nb of OUTPUT chanels. 3-channels for each gate on 1st layer
                                         self.R_filt_sizes[l],  # indexed by l: use 3x3 filters for each gate on 1st & 2nd layers
                                         padding='same', 
                                         activation=act, # LSTM_activation is 'tanh'. LSTM_inner_activation is 'hard_sigmoid'.
                                         data_format=self.data_format)
                print("layer: ", l, "   gate: ", c, "   nb of core/out channels: ", conv2D_temp_obj.filters, 
                      "   kernel size: ", conv2D_temp_obj.kernel_size)
                self.conv_layers[c].append(conv2D_temp_obj)
# =============================================================================
#                 self.conv_layers[c].append(Conv2D(self.R_stack_sizes[l], # indexed by l: (3, 48). Nb of OUTPUT chanels. 3-channels for each gate on 1st layer
#                                                   self.R_filt_sizes[l],  # indexed by l: use 3x3 filters for each gate on 1st & 2nd layers
#                                                   padding='same', 
#                                                   activation=act, # LSTM_activation is 'tanh'. LSTM_inner_activation is 'hard_sigmoid'.
#                                                   data_format=self.data_format))
# =============================================================================
            # endFor c

            # All all gates for cLSTM for each layer have been built.

            # Build AHAT and A for layer.
            #
            # If 2-layer model, builds 'a' and 'ahat' conv for 1st layer on first l iteration, and then 2nd layer on second iteration
            # 'relu' for lowest layer. self.A_activation happens to also be 'relu'
            act = 'relu' if l == 0 else self.A_activation
            ahat_temp_obj = Conv2D(self.stack_sizes[l],
                                   self.Ahat_filt_sizes[l], 
                                   padding='same', 
                                   activation=act, 
                                   data_format=self.data_format)
            print("ahat layer: ", l, "       nb of core/out channels: ", ahat_temp_obj.filters, 
                  "   kernel size: ", ahat_temp_obj.kernel_size)
            self.conv_layers['ahat'].append(ahat_temp_obj)
# =============================================================================
#             self.conv_layers['ahat'].append(Conv2D(self.stack_sizes[l],
#                                                    self.Ahat_filt_sizes[l], 
#                                                    padding='same', 
#                                                    activation=act, 
#                                                    data_format=self.data_format))
# 
# =============================================================================
            #Below: Leave out A module
            #self.conv_layers['a'].append(Conv2D(self.A_stack_sizes[l],         # Nb of output channels for A
            #                                    self.A_filt_sizes[l],          # Code correction: 6/4/2019
            #                                    padding='same', 
            #                                    activation=self.A_activation,  # happens to be 'relu'
            #                                    data_format=self.data_format))

        # endFor l
        # Dictionary should now look like:
        # {'i': [conv2d, conv2d], 'f': [conv2d, conv2d], 'c': [conv2d, conv2d], 'o': [conv2d, conv2d], 'a': [conv2d, conv2d], 'ahat': [conv2d, conv2d]}
 
    
        # Data_format is 'channels_last' which gives: (batch, nt?, rows, cols, channels).
        # Since the 2 layers below do not have trainable wts and the inputs can be any image dimension, 
        # they are not layer-specific and can be reused in a multi-layer architecture.
        self.upsample = UpSampling2D(data_format=self.data_format) 
        self.pool = MaxPooling2D(data_format=self.data_format)     # maxpooling
        # Above: Remember no pooling or upsampling for RBP 1st layer.
        # Finished creating the module components for the model.

        # Start creating TRAINABLE WTS for all model components.
        # Initialize trainable_weights.
        self.trainable_weights = []
        
        # Different backends have different data formats.
        # Compute number of rows and number of cols.
        # For 'channels_last', input_shape should be: (batch, nt?, rows, cols, channels)
        nb_row, nb_col = (input_shape[-2], input_shape[-1]) if self.data_format == 'channels_first' else (input_shape[-3], input_shape[-2])
        # Above: nb_row == 128, nb_col == 160
        
        # At this point, all of the computational objects for the model have been built.
        # It's now time to connect them by weights.
        
        # Create/initialize WEIGHTS
        # =========================
        # For each conv2D op in each layer, set up wts
        print("\nStarting WEIGHTS loop:")
        # Below: not sure why keys need to be sorted.
        #        Doubly nested loop creates a wt set for each trainable component in the model.
        # LOOP: 2 nestings.
        nb_channels = 0 # Added b/c was not initialized. Seems to be number of input channels.
        print("Iterating over layers.")
        for l in range(len(self.conv_layers[c])): # iterate over Conv2D objects under key
            print("\nLayer: ", l)
            for c in sorted(self.conv_layers.keys()): # returns ['a', 'ahat', 'c', 'f', 'i', 'o']. 
                                                      # Dictionary conv_layers is defined at beginning of build() method.
                print("\n  gate == ", c)
                # First step: set the value of 'in_shape' for current component.
                ds_factor = 2 ** l  # downsample factor, for l=0,1 will be 1, 2.
                
                if c == 'ahat':
                    nb_channels = self.R_stack_sizes[l]     # nb of input channels
                elif c == 'a': # not used in RBP model
                    if l == 0:
                        nb_channels = 2*self.R_stack_sizes[0]
                    else:
                        nb_channels = 2*self.R_stack_sizes[l-1]
                else: # 'c', 'f', 'i', 'o'
                    # add up all of channels in all of the inputs to the R_module for layer l.
                    nb_channels = self.R_stack_sizes[l] + 2*self.stack_sizes[l]  # new for RBP model (2 doubles output chanels of error module)
                    if l < self.nb_layers - 1:
                        nb_channels += 2*self.R_stack_sizes[l] # In RBP model, adjacent input from E is 2*nb_core channels in R
                        
                print("    nb_inp_channels : ", nb_channels)
                # Below: Now we have info to define in_shape, which will be input to self.conv_layers[c][l].build(in_shape).
                #        ds_factor is used to calculate dimensions for 2x2 pooling
                in_shape = (input_shape[0], nb_channels, nb_row // ds_factor, nb_col // ds_factor) # '//' is floor division
                if self.data_format == 'channels_last': in_shape = (in_shape[0], in_shape[2], in_shape[3], in_shape[1])
                print("    in_shape        : ", in_shape)
                
                # Second step: build the wt set for the current component.
                # what does name scope do? (context manager when defining a Python op)
                # Need to make sure wt dimensions match input in step() method.
                print("    kernel_size     : ", self.conv_layers[c][l].kernel_size)
                print("    nb_out_channels : ", self.conv_layers[c][l].filters) # tells nb of output channels
                with K.name_scope('layer_' + c + '_' + str(l)): # eg 'layer_a_0'
                    self.conv_layers[c][l].build(in_shape)      # What is side-effect?
                    # Above: Conv2D() instance understands its own build() method.
                    # The build() method is defined for class '_Conv', direct superclass of Conv2D.
                    # This adds the weights and bias (b/c Conv2D.use_bias is True)
                    # After this, self.conv_layers[c][l].trainable_weights has wts and understands call().
                print("    trainable wts length : ", len(self.conv_layers[c][l].trainable_weights))
                print("    trainable wts shape  : ", self.conv_layers[c][l].trainable_weights[0])
                print("    trainable bias shape : ", self.conv_layers[c][l].trainable_weights[1])
                self.trainable_weights += self.conv_layers[c][l].trainable_weights
                # Above: For some reason add the newly created trainable wts to a list.
                #        Not used in this file.
                # associated w/ the prednet instance.

        self.states = [None] * self.nb_layers*3
        # Above: creates [None, None, None, None, None, None]
        # Don't know how the above is used.

        # Doesn't appear to be executed in current version b/c test eval's to False
        if self.extrap_start_time is not None:
            self.t_extrap = K.variable(self.extrap_start_time, int if K.backend() != 'tensorflow' else 'int32')
            self.states += [None] * 2  # [previous frame prediction, timestep]
        print("RETURNING from build()")
    # end build()

    # STEP
    #=====
    # Apparently used by Recurrent class
    # Arguments:
    #            a     : actual (target) current input frame
    #            states: For each layer, there are three state entries.
    #                    1. Outputs of the representation (cLSTM) at t-1.
    #                    2. Cell states of cLSTM at t-1.
    #                    3. Error states at t-1.
    # Returns: output, states
    def step(self, a, states):
        print("\n------------------------------------------------------------------")
        print("prednet_RBP_28June2019.py: 'step()' called")
        print("\noutput_mode: ", self.output_mode)
        print("   target a: ", a)
        print("\nStates at time t minus 1 (tm1):")
#        print("\nstates: ", states)
        for i in range(len(states)):
            print("        ", states[i])
        print("States length: ", len(states)) # states is a tuple of length 6, i.e., 3*nb_layers.
        # Below: components used to make up the inputs
        # r_tm1: representation (cLSTM) states (stack of images) at prev time step
        # c_tm1: cell states at prev time step (different from c-gate)
        # e_tm1: error states at prev time step
        # tm1: t - 1
        r_tm1 = states[:self.nb_layers]                    # 1st l elements of states tuple. One state per layer. LSTM output state
        c_tm1 = states[self.nb_layers:2*self.nb_layers]    # Next l elements. LSTM cell state
        e_tm1 = states[2*self.nb_layers:3*self.nb_layers]  # Last l elements. Error

        # Test eval's to False: ignore
        if self.extrap_start_time is not None: 
            t = states[-1]
            a = K.switch(t >= self.t_extrap, states[-2], a)  # if past self.extrap_start_time, 
                                                             # the previous prediction will be treated as the actual
                                                             
        # initialize state variables for current time step. 'states' will be: r + c + e (list append)
        r = []; c = []; e = []

        # LOOP1.
        # LOOP1. DOWNWARD UPDATE SWEEP.
        # Update R (cLSTM) units starting from the top
        print("\nstarting Downward Sweep (LOOP1)\n")
        for l in reversed(range(self.nb_layers)): # reversed() starts from the top layer
            
            # Calculating inputs for R modules.
            # NEW code for RBP model.
            if l < self.nb_layers - 1: # not the top layer
                upsamp_e_tm1 = self.upsample.call(e_tm1[l+1])
                print("    Layer:", l, ". Shape of upsamp_e_tm1: ", upsamp_e_tm1)
                inputs  = [r_tm1[l], upsamp_e_tm1]  # recurrent, horizontal
            else: 
                inputs = [r_tm1[l]] # no upsampling for top layer. Only recurrent inputs.

            print("    Layer:", l, "   Shape of e_tm1[l]: ", e_tm1[l])
#                 
#=============== Taking out pooling from e_tm[l] to r[l]
#            A_up_raw = e_tm1[l]
#            
#            if l > 0:
#                A_up = self.pool.call(A_up_raw)
#            else:
#                A_up = A_up_raw
#            
#            print("    Layer: l=", l, ". Shape of A_up: ", A_up.shape)
#==================================== END
            inputs.append(e_tm1[l])
            # end New code
            
            # The activation updates are performed by the call() method.
            # Seems to append current inputs to inputs from prev time step.
            print("\n    Inputs for R to concat: ")
            for i in range(len(inputs)):
                print("        ", inputs[i])
            print("    Inputs length:", len(inputs))
            
            inputs = K.concatenate(inputs, axis=self.channel_axis) # creates a stack of images
            print("    Inputs after concat: ", inputs)
            # Above: current input concatentated w/ previous output
            
            i = self.conv_layers['i'][l].call(inputs) # activations for input gate are calculated
            f = self.conv_layers['f'][l].call(inputs) # forget
            o = self.conv_layers['o'][l].call(inputs) # output
            # Above: the gate activations have been updated
            
            # Below: compute the output of the constant error carosel (output of  + operation)
            _cell_state = f * c_tm1[l] + i * self.conv_layers['c'][l].call(inputs)
            
            # Below: modulate '_cell_state' by the output gate activation
            _r = o * self.LSTM_activation(_cell_state)
            print("    _r.shape:", _r.shape)
            
            # update c and r state lists
            # Inserting into front of list sorts entries according to layer
            c.insert(0, _cell_state)  # Insert stack of images into list 'c' at the beginning (different than c gate)
            r.insert(0, _r)
            print("")
        # end of top-down sweep
        print("LOOP1 is finished. Examine states created:")
        # END LOOP1
        
        print("    c states:")
        for i in range(len(c)):
            print("        ", c[i])
        print("    c states length:", len(c))
        
        print("    r outputs:")
        for i in range(len(r)):
            print("        ", r[i])
        print("    r states length:", len(r))

        # LOOP2: Update feedforward path starting from the bottom
        # UPDATE E's
        # New code: replace 'e_up' and 'e_down' w/ ppe and npe. See "Predictive Processing," Keller et al., Neuron, 2018.
        print("\nstarting Upward Sweep (LOOP2)")
        for l in range(self.nb_layers):   # start from bottom layer
            
            # New code for RBP.
            ahat = self.conv_layers['ahat'][l].call(r[l]) # 'ahat' is prediction
            if l > 0:
                # ahat = self.upsample.call(ahat)
                a = self.pool.call(r[l-1])
                
            if l == 0: frame_prediction = ahat
            
            ppe = self.error_activation(ahat - a) # Positive prediction error: using ReLU (rectified)
            npe = self.error_activation(a - ahat)
            print("\n    Layer:", l, "   Shape of ppe: ", ppe.shape, ". Shape of npe: ", npe.shape, ".")
            
#            e.insert(0, K.concatenate((ppe, npe), axis=self.channel_axis))
            e.append(K.concatenate((ppe, npe), axis=self.channel_axis)) # change to append b/c not visiting layers in reverse order
                                                                        # Keep list sorted by layer.
                
            print("    output_mode:", self.output_mode);
            print("    self.output_layer_num:", self.output_layer_num);
            # self.output_layer_num seems to be none when called from kitti_train.py
            if self.output_layer_num == l:
                if self.output_layer_type == 'A':
                    output = a
                elif self.output_layer_type == 'Ahat':
                    output = ahat
                elif self.output_layer_type == 'R':
                    output = r[l]
                elif self.output_layer_type == 'E':
                    output = e[l]
            print("    self.output_layer_type:", self.output_layer_type)
        # end of bottom-up sweep 
        
        print("\n    Printing computed prediction error shapes.")
        for l in range(self.nb_layers):
            print("        Layer:", l, "    Pred error shape:", e[l].shape)
                
        
        if self.output_layer_type is None:
            if self.output_mode == 'prediction':
                print("\n     output_layer_type:", self.output_layer_type)
                print("\n           output_mode:", self.output_mode)
                output = frame_prediction
                print("\nframe_prediction.shape:", frame_prediction.shape)
            else:
                for l in range(self.nb_layers):
                    layer_error = K.mean(K.batch_flatten(e[l]), axis=-1, keepdims=True)
                    #lastlayerError=outclass-
                    all_error = layer_error if l == 0 else K.concatenate((all_error, layer_error), axis=-1)
                if self.output_mode == 'error':
                    output = all_error
                else:
                    output = K.concatenate((K.batch_flatten(frame_prediction), all_error), axis=-1)

        states = r + c + e # list concatentation. Each element is a stack of images.
        if self.extrap_start_time is not None:
            states += [frame_prediction, t + 1]
        print("RETURNING from step() with values:")
        print("    output: ", output)
        print("    states: ")
        for i in range(len(states)):
            print("        ", states[i])
        print("    states length:", len(states))
        return output, states
    # end step()

    def get_config(self):
        config = {'stack_sizes': self.stack_sizes,
                  'R_stack_sizes': self.R_stack_sizes,
                  'A_filt_sizes': self.A_filt_sizes,
                  'Ahat_filt_sizes': self.Ahat_filt_sizes,
                  'R_filt_sizes': self.R_filt_sizes,
                  'pixel_max': self.pixel_max,
                  'error_activation': self.error_activation.__name__,
                  'A_activation': self.A_activation.__name__,
                  'LSTM_activation': self.LSTM_activation.__name__,
                  'LSTM_inner_activation': self.LSTM_inner_activation.__name__,
                  'data_format': self.data_format,
                  'extrap_start_time': self.extrap_start_time,
                  'output_mode': self.output_mode}
        base_config = super(PredNet_RBP, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
