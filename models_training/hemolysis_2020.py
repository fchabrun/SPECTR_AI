# -*- coding: utf-8 -*-
"""
Author : Floris Chabrun <floris.chabrun@chu-angers.fr>
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import os
from tqdm import tqdm
import pickle
import argparse
import json
import codecs

parser = argparse.ArgumentParser()
parser.add_argument("--debug", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--input_fractions", type=int, default=0)
parser.add_argument("--weighted", type=int, default=0)
FLAGS = parser.parse_args()

debug_mode = FLAGS.debug
# reload_model = (FLAGS.reload_model!=0)
# gan_training = (FLAGS.gan_training!=0)
base_lr = 1e-3
min_lr = 1e-5
BATCH_SIZE = FLAGS.batch_size
input_fractions = (FLAGS.input_fractions!=0)
weighted = (FLAGS.weighted!=0)

path_in = './input'
path_out = './output'

model_name = 'hemolysis_best_full_model_2020-batchsize-{}-f-{}-w-{}.h5'.format(BATCH_SIZE,input_fractions*1,weighted*1)
log_name = 'hemolysis_training_2020-batchsize-{}-f-{}-w-{}.log'.format(BATCH_SIZE,input_fractions*1,weighted*1)

# load raw data
raw = pd.read_csv(os.path.join(path_in,'data.csv'))

# spe size
spe_width=304

# select training and supervision sets
train_partition = (raw.part == 0)
super_partition = (raw.part == 1)

curve_columns = [c for c in raw.columns if c[0]=='x']
fractions_columns = [c for c in raw.columns if c[0]=='d' and len(c)==2]
peaks_columns = [c for c in raw.columns if c[0]=='p' and len(c)==3]
if len(curve_columns) != spe_width:
    raise Exception('Expected {} points curves, got {}'.format(spe_width,len(curve_columns)))
if len(fractions_columns) != 7:
    raise Exception('Expected {} fractions, got {}'.format(7,len(fractions_columns)))
if len(peaks_columns) != 8:
    raise Exception('Expected {} fractions, got {}'.format(8,len(peaks_columns)))
    
# extract input data
x_train=raw.loc[train_partition,curve_columns].to_numpy()
x_super=raw.loc[super_partition,curve_columns].to_numpy()

# normalize
x_train = x_train/(np.max(x_train, axis = 1)[:,None])
x_super = x_super/(np.max(x_super, axis = 1)[:,None])

# We'll select beta-gamma bridges as one class, then monoclonal-like abnormalities : spike, hemolysis, restriction
# Other SPEs will be considered "normal"
y_hem=raw.loc[:,'annotation_hemolysis'].to_numpy()
y_train = y_hem[train_partition]
y_super = y_hem[super_partition]

TRAIN_NEG_OVER_POS_RATIO=3
SUPER_NEG_OVER_POS_RATIO=1

y_train_indices=np.where(y_train==0)[0]
np.random.seed(123) # set seed for reproducible results
np.random.shuffle(y_train_indices)
# now select the subset of negative samples
neg_train_selection = y_train_indices[:(TRAIN_NEG_OVER_POS_RATIO*np.sum(y_train==1))]
# we have the subset of positive samples
pos_train_selection = np.where(y_train==1)[0]
# we can merge both
train_selection=np.hstack((neg_train_selection, pos_train_selection))
# then shuffle pos and neg
np.random.seed(123) # set seed for reproducible results
np.random.shuffle(train_selection)
# now we can select our subset
x_train=x_train[train_selection,:]
y_train=y_train[train_selection]

# same for valid
y_super_indices=np.where(y_super==0)[0]
np.random.seed(123) # set seed for reproducible results
np.random.shuffle(y_super_indices)
# now select the subset of negative samples
neg_super_selection = y_super_indices[:(SUPER_NEG_OVER_POS_RATIO*np.sum(y_super==1))]
# we have the subset of positive samples
pos_super_selection = np.where(y_super==1)[0]
# we can merge both
super_selection=np.hstack((neg_super_selection, pos_super_selection))
# then shuffle pos and neg
np.random.seed(123) # set seed for reproducible results
np.random.shuffle(super_selection)
# now we can select our subset
x_super=x_super[super_selection,:]
y_super=y_super[super_selection]


# check again the balance
print('neg in train: '+str(np.sum(y_train == 0))) # 3666
print('pos in train: '+str(np.sum(y_train == 1))) # 1222

print('neg in super: '+str(np.sum(y_super == 0))) # 3666
print('pos in super: '+str(np.sum(y_super == 1))) # 1222

print('x training set shape: '+str(x_train.shape))
print('y training set shape: '+str(y_train.shape))
print('x validation set shape: '+str(x_super.shape))
print('y supervision set shape: '+str(y_super.shape))

# We have to load the rest of the input data :
# Age, and quantitative values
# First, we'll compute quantitative values
# fractions
dataset_parameters = dict()
if input_fractions:
    tp_train = raw.loc[train_partition,'tp'].to_numpy()
    tp_super = raw.loc[super_partition,'tp'].to_numpy()
    f_train=raw.loc[train_partition,fractions_columns].to_numpy().astype(int)
    f_super=raw.loc[super_partition,fractions_columns].to_numpy().astype(int)
        
    # Quantify fractions from curve
    def quantifyFractions(x, f):
        area = (x[1:] + x[:-1])/2
        f1 = (f[:-1]-1).astype(int)
        f2 = (f[1:]-1).astype(int)
        fractions_area = np.zeros(f.shape[0]-1)
        for i in range(f1.shape[0]):
            fractions_area[i] = np.sum(area[f1[i]:f2[i]])
        return(fractions_area/np.sum(fractions_area))
        
    def quantifyFractionsAbs(x, f, tp):
        return(quantifyFractions(x,f)*tp)
        
    # Compute quantities
    frac_abs_train = np.zeros((x_train.shape[0], 6))
    frac_abs_super = np.zeros((x_super.shape[0], 6))
    for ix in tqdm(range(x_train.shape[0])):
        x = x_train[ix,:]
        f = f_train[ix,:]
        tp = tp_train[ix]
        frac_abs_train[ix,:] = quantifyFractionsAbs(x, f, tp)
    for ix in tqdm(range(x_super.shape[0])):
        x = x_super[ix,:]
        f = f_super[ix,:]
        tp = tp_super[ix]
        frac_abs_super[ix,:] = quantifyFractionsAbs(x, f, tp)
        
    frac_abs_mean = np.mean(frac_abs_train, axis = 0)
    frac_abs_sd = np.std(frac_abs_train, axis = 0)
    
    frac_abs_train = (frac_abs_train-frac_abs_mean)/frac_abs_sd
    frac_abs_super = (frac_abs_super-frac_abs_mean)/frac_abs_sd

    # On sauvegarde les standards
    dataset_parameters['frac_abs_mean'] = frac_abs_mean.tolist()
    dataset_parameters['frac_abs_sd'] = frac_abs_sd.tolist()

    json.dump(dataset_parameters, codecs.open(os.path.join(path_out,'deep_hemolysis_parameters.json'), 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

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

def get_coremodel(inputs, n_filters=16, dropout=0.5, batchnorm=True, n_classes=4):
    # contracting path
    x = inputs[0]
    
    x = conv1d_block(x, n_filters=n_filters*1, kernel_size=3, batchnorm=batchnorm)
    x = tf.keras.layers.MaxPooling2D((2,1)) (x)
    x = tf.keras.layers.Dropout(dropout*0.5)(x)

    x = conv1d_block(x, n_filters=n_filters*2, kernel_size=3, batchnorm=batchnorm)
    x = tf.keras.layers.MaxPooling2D((2,1)) (x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = conv1d_block(x, n_filters=n_filters*4, kernel_size=3, batchnorm=batchnorm)
    x = tf.keras.layers.MaxPooling2D((2,1)) (x)
    x = tf.keras.layers.Dropout(dropout)(x)

    x = conv1d_block(x, n_filters=n_filters*8, kernel_size=3, batchnorm=batchnorm)
    x = tf.keras.layers.MaxPooling2D((2,1)) (x)
    x = tf.keras.layers.Dropout(dropout)(x)
    
    x = conv1d_block(x, n_filters=n_filters*16, kernel_size=3, batchnorm=batchnorm)
    
    x = tf.keras.layers.Flatten() (x)
    
    if len(inputs)>1:
        tmp_inputs = [x,]
        tmp_inputs.extend(inputs[1:])
        x = tf.keras.layers.Concatenate() (tmp_inputs)
        x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer="he_normal") (x)
    
    if n_classes>1:
        x = tf.keras.layers.Dense(n_classes, activation = 'softmax')(x)
    else:
        x = tf.keras.layers.Dense(n_classes, activation = 'sigmoid')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=[x])
    return model

inputs = [tf.keras.layers.Input((spe_width, 1, 1), name='input_spe'),]
if input_fractions:
    inputs.append(tf.keras.layers.Input(shape = (6,), name = 'input_fractions'))

model = get_coremodel(inputs=inputs, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1)

model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=base_lr), metrics=['accuracy'])

print(model.summary())

# define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=min_lr, verbose=1),
    tf.keras.callbacks.ModelCheckpoint(os.path.join(path_out,model_name), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)
]

# train model
N_EPOCHS = 1000
print('Setting batch size to: '+str(BATCH_SIZE))
print('Setting epochs to: '+str(N_EPOCHS))

if weighted:
    class_weight = {0: 1/(np.sum(y_train == 0)/y_train.shape[0]),
                    1: 1/(np.sum(y_train == 1)/y_train.shape[0])}
    print('Class weights: '+str(class_weight))
else:
    class_weight = None
    print('No class weights')

tmp_inputs_train = [x_train.reshape((x_train.shape[0], spe_width, 1, 1)),]
tmp_inputs_super = [x_super.reshape((x_super.shape[0], spe_width, 1, 1)),]
if input_fractions:
    tmp_inputs_train.append(frac_abs_train)
    tmp_inputs_super.append(frac_abs_super)

results = model.fit(x = tmp_inputs_train,
                    y = y_train,
                    batch_size = BATCH_SIZE,
                    epochs = N_EPOCHS,
                    verbose = 2,
                    callbacks=callbacks,
                    validation_data = (tmp_inputs_super,
                                       y_super),
                    class_weight = class_weight)

new_history=results.history

# save history
with open(os.path.join(path_out,log_name), 'wb') as file_pi:
    pickle.dump(new_history, file_pi)
    
if debug_mode==1:
    log_name_template = 'hemolysis_training_2020-batchsize-[0-9]+-f-[01]-w-[01].log'
        
    from matplotlib import pyplot as plt
    import re
    log_names = [f for f in os.listdir(path_out) if re.match(log_name_template, f)]
    
    len(log_names)
    
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
    
    # convert models to colors
    plt.figure(figsize=(8, 8))
    # plt.title("Learning curve")
    for mi,metric in enumerate(zip(['loss','accuracy',],['Loss','Accuracy',],['min','max'])):
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