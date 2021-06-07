# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:39:51 2019

@author: Floris Chabrun <floris.chabrun@chu-angers.fr>
"""

# This script allows training of a model for segmenting SPE curves
# Into six fractions (Albumin,A1,A2,B1,B2,Gamma) + M-spikes

# BELOW : create FLAGS for hyperparameters used during training

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=int, default=1)
parser.add_argument("--reload_model", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--separate_peak_fraction", type=int, default=0)
# parser.add_argument("--gan_training", type=int, default=0)
FLAGS = parser.parse_args()

debug_mode = FLAGS.debug
reload_model = (FLAGS.reload_model!=0)
# gan_training = (FLAGS.gan_training!=0)
base_lr = 1e-3
min_lr = 1e-5
BATCH_SIZE = FLAGS.batch_size
separate_peak_fraction = (FLAGS.separate_peak_fraction!=0)

print("FLAGS:")
for f in dir(FLAGS):
    if f[:1] != "_":
        print("    {}: {}".format(f,getattr(FLAGS,f)))
print("")

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
import pickle
import os
if debug_mode==1:
    import matplotlib.pyplot as plt
    import matplotlib.lines as ml

# DATA PATH

path_in = './input'
path_out = './output'

# Define model and log names (for saving output)
if separate_peak_fraction:
    model_name = 'segmentation_best_full_model_2020-batchsize-{}.h5'.format(BATCH_SIZE)
    log_name = 'segmentation_training-batchsize-{}.log'.format(BATCH_SIZE)
else:
    model_name = 'segmentation_best_full_model_nopf_2020-batchsize-{}.h5'.format(BATCH_SIZE)
    log_name = 'segmentation_training_nopf-batchsize-{}.log'.format(BATCH_SIZE)
    
# load raw data
raw = pd.read_csv(os.path.join(path_in,'data.csv'))

# define the size of the curve: here, 304 pixels
spe_width=304

# define partitions used : training and supervision
train_partition = (raw.part == 0)
super_partition = (raw.part == 1)

# get column names of desired data in the raw csv file
curve_columns = [c for c in raw.columns if c[0]=='x'] # data for y values in the curves
fractions_columns = [c for c in raw.columns if c[0]=='d' and len(c)==2] # data for fractioning (i.e. location of each segmentation point in the 304 pixels curve)
peaks_columns = [c for c in raw.columns if c[0]=='p' and len(c)==3] # localization data for M-spikes
if len(curve_columns) != spe_width:
    raise Exception('Expected {} points curves, got {}'.format(spe_width,len(curve_columns)))
if len(fractions_columns) != 7:
    raise Exception('Expected {} fractions, got {}'.format(7,len(fractions_columns)))
if len(peaks_columns) != 8:
    raise Exception('Expected {} fractions, got {}'.format(8,len(peaks_columns)))

# extract values of curves    
x_train=raw.loc[train_partition,curve_columns].to_numpy()
x_super=raw.loc[super_partition,curve_columns].to_numpy()

# normalize
x_train = x_train/(np.max(x_train, axis = 1)[:,None])
x_super = x_super/(np.max(x_super, axis = 1)[:,None])

# fractions
f_train=raw.loc[train_partition,fractions_columns].to_numpy().astype(int)
f_super=raw.loc[super_partition,fractions_columns].to_numpy().astype(int)

# and m-spikes
s_train=raw.loc[train_partition,peaks_columns].to_numpy().astype(int)
s_super=raw.loc[super_partition,peaks_columns].to_numpy().astype(int)

# below we convert the fractions
# which are expressed as numbers
#     e.g. 100 means that pixels before 100 are in fraction 1 and pixels after 100 are in fraction 2
# into maps which our model can handle
# e.g. either 0 or 1 value x the length of the curve (304)
# to be noted: we can either create 304x6 maps or 304x7, if we consider M-spikes as fractions themselves
# if dim = 304x6, we set all values to 0 for the location of the M-spike
if separate_peak_fraction:
    fract_shape = 7
else:
    fract_shape = 6

# prepare empty maps   
y_map_train = np.zeros(x_train.shape+(fract_shape,))
y_map_super = np.zeros(x_super.shape+(fract_shape,))

# our function to convert fractions to maps
def mapFractionsAndPeaks(f, p):
    new_map = np.zeros((1,spe_width,fract_shape,))
    bounds = np.concatenate(((0,),f[1:-1]+1,(spe_width,))) 
    for v,i in enumerate(zip(bounds[:-1],bounds[1:])):
        new_map[0,i[0]:i[1],v] = 1
    if np.sum(p) > 0:
        peaks = p+((p>0)*1)
        peaks = peaks[peaks>0]
        for pi, pf in zip(peaks[::2],peaks[1::2]):
            new_map[0,pi:pf,...] = 0
            if separate_peak_fraction:
                new_map[0,pi:pf,-1] = 1
    return new_map

# we fill the maps
for ix in tqdm(range(f_train.shape[0])):
    y_map_train[ix,...] = mapFractionsAndPeaks(f_train[ix,:],s_train[ix,:])
for ix in tqdm(range(f_super.shape[0])):
    y_map_super[ix,...] = mapFractionsAndPeaks(f_super[ix,:],s_super[ix,:])

# if we want to normalize curves ; which is actually not needed
# x_mean = np.mean(x_train, axis = 0)
# x_sd = np.std(x_train, axis = 0)
# x_sd[x_sd == 0] = 1
# # Now we can scale data
# x_train = (x_train-x_mean)/x_sd
# x_valid = (x_valid-x_mean)/x_sd

# just a debug function to plot the results of the conversion
if debug_mode==1:
    def plotTrainSegmentsMap(ix):
        plt.figure(figsize=(8,6))
        # on calcule la couleur
        class_map=np.argmax(y_map_train[ix,:,:],axis=1)
        class_map[np.sum(y_map_train[ix,:,:],axis=1)==0]=-1
        us_x=x_train[ix,:]
        ax = plt.gca()
        tmp_x_i=0
        while tmp_x_i<us_x.shape[0]:
            delta=1
            while tmp_x_i+delta<us_x.shape[0]:
                if class_map[tmp_x_i+delta]!=class_map[tmp_x_i]:
                    break
                delta+=1
            clr = ('black','green','purple','yellow','pink','blue','orange')[class_map[tmp_x_i]+1]
            plt.plot(np.arange(tmp_x_i,tmp_x_i+delta)+1, us_x[np.arange(tmp_x_i,tmp_x_i+delta)], '-', color = clr)
            tmp_x_i=tmp_x_i+delta
        for i in range(1,6):
            ax.add_line(ml.Line2D([f_train[ix,i]+1,f_train[ix,i]+1], [0,1], color = "black"))
    plotTrainSegmentsMap(0)
    plotTrainSegmentsMap(2)
    plotTrainSegmentsMap(4)
    plotTrainSegmentsMap(6)
    plotTrainSegmentsMap(8)

# print sizes
print('training set X shape: '+str(x_train.shape))
print('training set Y shape: '+str(y_map_train.shape))
print('supervision set X shape: '+str(x_super.shape))
print('supervision set Y shape: '+str(y_map_super.shape))

# define our custom loss function for computing jaccard distance loss
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=-1)
    sum_ = tf.keras.backend.sum(tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# furthermore, we also define our own curve accuracy method
def curve_accuracy(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.equal(tf.keras.backend.argmax(y_true, axis=-1), tf.keras.backend.argmax(y_pred, axis=-1)))

# debug
if debug_mode==1:
    tmp=y_map_train[5,...]
    np.sum(tmp) # 294 -> 8 points in peak
    np.mean(jaccard_distance_loss(y_map_train[5,...],y_map_train[5,...])) # expected 0
    np.mean(curve_accuracy(y_map_train[5,...],y_map_train[5,...])) # expected 1

# add loss and metric to keras built-in methods
tf.keras.losses.jaccard_distance_loss = jaccard_distance_loss
tf.keras.metrics.curve_accuracy=curve_accuracy

# %%

# create the u-net architecture
def conv1d_block(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    # first layer
    #x = Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same", data_format='channels_last')(input_tensor)
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size,1), kernel_initializer="he_normal", padding="same", data_format='channels_last')(input_tensor)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    # second layer
    #x = Conv1D(filters=n_filters, kernel_size=kernel_size, kernel_initializer="he_normal", padding="same", data_format='channels_last')(x)
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size,1), kernel_initializer="he_normal", padding="same", data_format='channels_last')(x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def get_unet(input_signal, n_filters=16, dropout=0.5, batchnorm=True, n_classes=fract_shape):
    # contracting path
    c1 = conv1d_block(input_signal, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = tf.keras.layers.MaxPooling2D((2,1)) (c1)
    p1 = tf.keras.layers.Dropout(dropout*0.5)(p1)

    c2 = conv1d_block(p1, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = tf.keras.layers.MaxPooling2D((2,1)) (c2)
    p2 = tf.keras.layers.Dropout(dropout)(p2)

    c3 = conv1d_block(p2, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = tf.keras.layers.MaxPooling2D((2,1)) (c3)
    p3 = tf.keras.layers.Dropout(dropout)(p3)

    c4 = conv1d_block(p3, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = tf.keras.layers.MaxPooling2D((2,1)) (c4)
    p4 = tf.keras.layers.Dropout(dropout)(p4)
    
    c5 = conv1d_block(p4, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    # expansive path
    u6 = tf.keras.layers.Conv2DTranspose(n_filters*8, (3,1), strides=(2,1), padding='same') (c5)
    u6 = tf.keras.layers.Concatenate() ([u6, c4])
    u6 = tf.keras.layers.Dropout(dropout)(u6)
    c6 = conv1d_block(u6, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = tf.keras.layers.Conv2DTranspose(n_filters*4, (3,1), strides=(2,1), padding='same') (c6)
    u7 = tf.keras.layers.Concatenate() ([u7, c3])
    u7 = tf.keras.layers.Dropout(dropout)(u7)
    c7 = conv1d_block(u7, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = tf.keras.layers.Conv2DTranspose(n_filters*2, (3,1), strides=(2,1), padding='same') (c7)
    u8 = tf.keras.layers.Concatenate() ([u8, c2])
    u8 = tf.keras.layers.Dropout(dropout)(u8)
    c8 = conv1d_block(u8, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = tf.keras.layers.Conv2DTranspose(n_filters*1, (3,1), strides=(2,1), padding='same') (c8)
    u9 = tf.keras.layers.Concatenate() ([u9, c1])
    u9 = tf.keras.layers.Dropout(dropout)(u9)
    c9 = conv1d_block(u9, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    
    outputs = tf.keras.layers.Conv2D(n_classes, (1,1), activation='sigmoid') (c9)
    model = tf.keras.models.Model(inputs=[input_signal], outputs=[outputs])
    return model

# callbacks for monitoring training
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=min_lr, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(path_out,model_name), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
]

# reload model = transfer learning
if reload_model:
    model=tf.keras.models.load_model(os.path.join(path_out,model_name))
    with open(os.path.join(path_out,log_name), 'rb') as file_pi:
        old_history = pickle.load(file_pi)
else:
    # create model from scratch
    input_signal = tf.keras.layers.Input((spe_width, 1, 1), name='input_spe')
    model = get_unet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=fract_shape)
    model.compile(loss=jaccard_distance_loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_accuracy])

print(model.summary())

# Finally, save means and standard deviations
# dataset_parameters = {'x_mean': x_mean.tolist(), 'x_sd':x_sd.tolist()}
# json.dump(dataset_parameters, codecs.open(path_out+'segmentation_cnn_parameters.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) 

# train model
N_EPOCHS = 1000
# debug : reduce datasets size for speeding up the training process
if debug_mode==1:
    BATCH_SIZE=8
    N_EPOCHS=10
    sz=BATCH_SIZE*10
    x_train=x_train[:sz,...]
    x_super=x_super[:sz,...]
    y_map_train=y_map_train[:sz,...]
    y_map_super=y_map_super[:sz,...]
print('Setting batch size to: '+str(BATCH_SIZE))
print('Setting maximal number of epochs to: '+str(N_EPOCHS))

# actual taining
results = model.fit(x_train.reshape( x_train.shape + (1,1) ),
                    y_map_train.reshape( (y_map_train.shape[0], spe_width, 1, fract_shape) ),
                    batch_size=BATCH_SIZE,
                    epochs=N_EPOCHS,
                    callbacks=callbacks,
                    verbose=2,
                    validation_data=(x_super.reshape((x_super.shape[0], spe_width, 1, 1)),
                                     y_map_super.reshape((y_map_super.shape[0], spe_width, 1, fract_shape))))

if reload_model:
    # we have to merge old training history with new training history
    for key in old_history.keys():
        old_history[key].extend(results.history[key])
    new_history = old_history
else:
    new_history=results.history

# save history
with open(os.path.join(path_out,log_name), 'wb') as file_pi:
    pickle.dump(new_history, file_pi)

# %%

# read history
if debug_mode==1:
    # search all log files
    if separate_peak_fraction:
        log_name_template = 'segmentation_training-batchsize-[0-9]+.log'
    else:
        log_name_template = 'segmentation_training_nopf-batchsize-[0-9]+.log'
        
    import re
    log_names = [f for f in os.listdir(path_out) if re.match(log_name_template, f)]
    
    logs = None
    for log_name in log_names:
        with open(os.path.join(path_out,log_name), 'rb') as file_pi:
            temp = pickle.load(file_pi)
        temp = pd.DataFrame(temp) # convert to df
        temp['model'] = log_name # add a column stating which log those stats concern
        logs = pd.concat([logs,temp], axis=0) # concat with rest of data
        
    start_from = 0
    only_val = True
    
    std_colors =     ['#FF4500', '#1E90FF', '#556B2F', '#8B008B', '#8B4513']
    std_val_colors = ['#FF8C00', '#87CEFA', '#9ACD32', '#BA55D3', '#DEB887']
    
    # plots history logs (i.e. evolution of losses & metrics during training)
    plt.figure(figsize=(8, 8))
    # plt.title("Learning curve")
    for mi,metric in enumerate(zip(['loss','curve_accuracy',],['Loss','Accuracy',],['min','max'])):
        plt.subplot(2, 1, mi+1)
        for m,mod in enumerate(logs['model'].unique()):
            if not only_val:
                plt.plot(logs.loc[logs.model==mod,metric[0]].iloc[start_from:],
                         color = std_colors[m],
                         label='{}-{}'.format(metric[0],mod))
            plt.plot(logs.loc[logs.model==mod,'val_'+metric[0]].iloc[start_from:],
                     color = std_val_colors[m],
                     label='val_{}-{}'.format(metric[0],mod))
            if metric[2]=='min':
                plt.plot( np.argmin(logs.loc[logs.model==mod,"val_"+metric[0]].iloc[start_from:]),
                         np.min(logs.loc[logs.model==mod,"val_"+metric[0]].iloc[start_from:]), marker="x", color="r")
                # plt.text( np.argmin(logs.loc[logs.model==mod,"val_"+metric[0]].iloc[start_from:]),
                #          np.min(logs.loc[logs.model==mod,"val_"+metric[0]].iloc[start_from:]),
                #          s="{:.3f}".format(np.min(logs.loc[logs.model==mod,"val_"+metric[0]].iloc[start_from:])), color="r")
            else:
                plt.plot( np.argmax(logs.loc[logs.model==mod,"val_"+metric[0]].iloc[start_from:]),
                         np.max(logs.loc[logs.model==mod,"val_"+metric[0]].iloc[start_from:]), marker="x", color="r")
                plt.text( np.argmax(logs.loc[logs.model==mod,"val_"+metric[0]].iloc[start_from:]),
                         np.max(logs.loc[logs.model==mod,"val_"+metric[0]].iloc[start_from:]),
                         s="{:.3f}".format(100*np.max(logs.loc[logs.model==mod,"val_"+metric[0]].iloc[start_from:])), color="r")
        plt.xlabel("Epochs")
        plt.ylabel(metric[1])
        plt.legend()
    plt.show()
    
    model_name = 'segmentation_best_full_model_nopf_2020-batchsize-32.h5'
    
    # reload model
    model=tf.keras.models.load_model(os.path.join(path_out,model_name),
                                     custom_objects = dict(jaccard_distance_loss=jaccard_distance_loss,
                                                           curve_accuracy=curve_accuracy))
    
    # Make predictions and see those predictions
    size=x_super.shape[0]
    x=x_super[:size,...]
    y=y_map_super[:size,...]
    y_=model.predict(x.reshape(x.shape+(1,)))
    f=f_super[:size,...]
    def plotPredictedMap(ix):
        plt.figure(figsize=(8,6))
        # on calcule la couleur
        class_map=np.argmax(y_[ix,:,0,:],axis=1)+1
        # thresholding to avoid false classification
        class_map[np.max(y_[ix,:,0,:],axis=1)<.1]=0
        us_x=x[ix,:]
        ax = plt.gca()
        clr=np.array(('black','green','purple','yellow','pink','blue','orange','red'))
        for i in range(8):
            points=np.where(class_map==i)[0]
            plt.plot(points+1, us_x[points], '-', color=clr[i])
        # on va en plus représenter une droite pour la détection de pics
        # plt.plot(np.arange(1,spe_width+1,1), y_[ix,:,0,6], '-', color='red')
        # ou la somme
        plt.plot(np.arange(1,spe_width+1,1), np.sum(y_[ix,:,0,:], axis=1), '-', color='red')
        for i in range(1,6):
            ax.add_line(ml.Line2D([f[ix,i]+1,f[ix,i]+1], [0,1], color = "black"))
        return class_map