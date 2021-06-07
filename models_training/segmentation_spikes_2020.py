# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 23:39:51 2019

@author: Floris Chabrun <floris.chabrun@chu-angers.fr>
"""

# here we train a model for locating m-spikes on SPE curves

print('Starting spikes segmentation training script...')

# load required libraries & constants + hyperparameters for training

import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tqdm import tqdm
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
FLAGS = parser.parse_args()

debug_mode = FLAGS.debug
base_lr = 1e-3
min_lr = 1e-5
BATCH_SIZE = FLAGS.batch_size

path_in = './input'
path_out = './output'

model_name = 'segmentation_spikes_best_full_model_2020-batchsize-{}.h5'.format(BATCH_SIZE)
log_name = 'segmentation_spikes_training_2020-batchsize-{}.log'.format(BATCH_SIZE)
    
# load raw data
raw = pd.read_csv(os.path.join(path_in,'data.csv'))

# define size of the curve
spe_width=304

# select partitions
train_partition = (raw.part == 0)
super_partition = (raw.part == 1)

# select columns for extracting data
curve_columns = [c for c in raw.columns if c[0]=='x']
fractions_columns = [c for c in raw.columns if c[0]=='d' and len(c)==2]
peaks_columns = [c for c in raw.columns if c[0]=='p' and len(c)==3]
if len(curve_columns) != spe_width:
    raise Exception('Expected {} points curves, got {}'.format(spe_width,len(curve_columns)))
if len(fractions_columns) != 7:
    raise Exception('Expected {} fractions, got {}'.format(7,len(fractions_columns)))
if len(peaks_columns) != 8:
    raise Exception('Expected {} fractions, got {}'.format(8,len(peaks_columns)))

# extract spe curves data
x_train=raw.loc[train_partition,curve_columns].to_numpy()
x_super=raw.loc[super_partition,curve_columns].to_numpy()

# normalize
x_train = x_train/(np.max(x_train, axis = 1)[:,None])
x_super = x_super/(np.max(x_super, axis = 1)[:,None])

# extract m-spikes
s_train=raw.loc[train_partition,peaks_columns].to_numpy().astype(int)
s_super=raw.loc[super_partition,peaks_columns].to_numpy().astype(int)

# filter partitions
filter_train = s_train[:,0] != 0
filter_super = s_super[:,0] != 0

x_train = x_train[filter_train,:]
s_train = s_train[filter_train,:]
x_super = x_super[filter_super,:]
s_super = s_super[filter_super,:]

# check sizes
print('training set X shape: '+str(x_train.shape))
print('temporary training set Y shape: '+str(s_train.shape))
print('supervision set X shape: '+str(x_super.shape))
print('temporary supervision set Y shape: '+str(s_super.shape))

# convert to maps (cf. segmentation script)
y_map_train=np.zeros_like(x_train)
y_map_super=np.zeros_like(x_super)

points=np.arange(1,spe_width+1,1)
for ix in tqdm(range(s_train.shape[0])):
    y_map_train[ix,:]=1*(points>=s_train[ix,0])*(points<=s_train[ix,1])+1*(points>=s_train[ix,2])*(points<=s_train[ix,3])+1*(points>=s_train[ix,4])*(points<=s_train[ix,5])+1*(points>=s_train[ix,6])*(points<=s_train[ix,7])
for ix in tqdm(range(s_super.shape[0])):
    y_map_super[ix,:]=1*(points>=s_super[ix,0])*(points<=s_super[ix,1])+1*(points>=s_super[ix,2])*(points<=s_super[ix,3])+1*(points>=s_super[ix,4])*(points<=s_super[ix,5])+1*(points>=s_super[ix,6])*(points<=s_super[ix,7])

# debug : plot sample
if debug_mode == 1:
    import matplotlib.pyplot as plt
    def plotTrainSpikesMap(ix):
        plt.figure(figsize=(8,6))
        # on calcule la couleur
        class_map=y_map_train[ix,:]
        us_x=x_train[ix,:]
        ax = plt.gca()
        points=np.arange(1,385,1)
        s_clrs=('blue','green','purple','magenta')
        for i in range(4):
            sp_map=1*(points>=s_train[ix,2*i])*(points<=s_train[ix,2*i+1])
            ax.fill_between(np.where(sp_map==1)[0]+2, 0, us_x[np.where(sp_map==1)[0]+1], facecolor=s_clrs[i], interpolate=True)
        tmp_x_i=0
        while tmp_x_i<(us_x.shape[0]-2):
            tmp_x_f=tmp_x_i+2
            while tmp_x_f<us_x.shape[0]:
                if class_map[tmp_x_f]!=class_map[tmp_x_i]:
                    break
                tmp_x_f+=1
            if tmp_x_f>(us_x.shape[0]-2):
                tmp_x_f=(us_x.shape[0]-2)
            clr='black'
            if class_map[tmp_x_i]==1:
                clr='red'
            plt.plot(np.arange(tmp_x_i,tmp_x_f+2)+1, us_x[np.arange(tmp_x_i,tmp_x_f+2)], '-', color = clr)
            tmp_x_i=tmp_x_f
    plotTrainSpikesMap(0)
    plotTrainSpikesMap(2)

# check shapes
print('training set X shape: '+str(x_train.shape))
print('training set Y shape: '+str(y_map_train.shape))
print('supervision set X shape: '+str(x_super.shape))
print('supervision set Y shape: '+str(y_map_super.shape))

# construct unet
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

def get_unet(input_signal, n_filters=16, dropout=0.5, batchnorm=True, n_classes=1):
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

input_signal = tf.keras.layers.Input((spe_width, 1, 1), name='input_spe')
model = get_unet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1)

# define custom loss
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = tf.keras.backend.sum(tf.keras.backend.abs(y_true * y_pred), axis=-1)
    sum_ = tf.keras.backend.sum(tf.keras.backend.abs(y_true) + tf.keras.backend.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

# define custom accuracy
def curve_accuracy(y_true, y_pred):
    return tf.keras.backend.mean(tf.keras.backend.equal(y_true>.5, y_pred>.5))

def iou(y_true, y_pred, smooth=100):
    y_true_pos=tf.keras.backend.cast(y_true>.5, tf.keras.backend.floatx())
    y_pred_pos=tf.keras.backend.cast(y_pred>.5, tf.keras.backend.floatx())
    I=y_true_pos*y_pred_pos
    U=tf.keras.backend.minimum(y_true_pos+y_pred_pos, 1)
    IoU=tf.keras.backend.sum(I,axis=1)/tf.keras.backend.sum(U,axis=1)
    return IoU

# create callbacks for monitoring training
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=min_lr, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(path_out,model_name), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
]

# compile model
model.compile(loss=jaccard_distance_loss, optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=[curve_accuracy,iou])

print(model.summary())

# train
N_EPOCHS = 1000
if debug_mode==1:
    BATCH_SIZE=8
    N_EPOCHS=10
    sz=16
    x_train=x_train[:sz,...]
    x_super=x_super[:sz,...]
    y_map_train=y_map_train[:sz,...]
    y_map_super=y_map_super[:sz,...]
print('Setting batch size to: '+str(BATCH_SIZE))
print('Setting maximal number of epochs to: '+str(N_EPOCHS))

# actual training
results = model.fit(x_train.reshape( x_train.shape + (1,1) ),
                    y_map_train.reshape( y_map_train.shape + (1,1) ),
                    batch_size=BATCH_SIZE,
                    epochs=N_EPOCHS,
                    callbacks=callbacks,
                    verbose=2,
                    validation_data=(x_super.reshape((x_super.shape[0], spe_width, 1, 1)),
                                     y_map_super.reshape( y_map_super.shape + (1,1) )))

new_history=results.history

# Save history
with open(os.path.join(path_out,log_name), 'wb') as file_pi:
    pickle.dump(new_history, file_pi)
    
# print history, logs & plot samples
if debug_mode==1:
    
    log_name_template = 'segmentation_spikes_training_2020-batchsize-[0-9]+.log'
    
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
    
    from matplotlib import pyplot as plt

    # convert models to colors
    plt.figure(figsize=(8, 8))
    # plt.title("Learning curve")
    for mi,metric in enumerate(zip(['loss','curve_accuracy','iou',],['Loss','Accuracy','IoU'],['min','max','max'])):
        plt.subplot(3, 1, mi+1)
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