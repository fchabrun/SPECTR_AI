#!/usr/bin/python
# -*- coding: utf-8 -*-

print('Running python script')

print('Loading required libraries')

import argparse
import os
import h5py
import tensorflow as tf
import numpy as np
import json

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

def F_MODEL(spe_width = 304):
    def get_funet(input_signal, n_filters=16, dropout=0.5, batchnorm=True, n_classes=6):
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
    
    input_signal = tf.keras.layers.Input((spe_width, 1, 1))
    model = get_funet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=6)
    
    return model

def S_MODEL(spe_width = 304):
    def get_sunet(input_signal, n_filters=16, dropout=0.5, batchnorm=True, n_classes=1):
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
    model = get_sunet(input_signal, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1)
    
    return model

def H_MODEL(spe_width = 304):
    def get_hcoremodel(inputs, n_filters=16, dropout=0.5, batchnorm=True, n_classes=4):
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
    
    inputs = [tf.keras.layers.Input((spe_width, 1, 1)),]
    model = get_hcoremodel(inputs=inputs, n_filters=16, dropout=0.05, batchnorm=True, n_classes=1)
    
    return model

def C_MODEL(spe_width = 304):
    def get_ccoremodel(inputs, n_filters=16, dropout=0.5, batchnorm=True, n_classes=4):
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
        
        x = tf.keras.layers.Dense(n_classes, activation = 'softmax')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=[x])
        return model
    
    inputs = [tf.keras.layers.Input((spe_width, 1, 1)),]
    model = get_ccoremodel(inputs=inputs, n_filters=16, dropout=0.05, batchnorm=True, n_classes=4)
    
    return model

def LOADMODELS(model_path, save_format, spe_width = 304):
    loaded_models = []
    print('Loading models ({})'.format(model_path))
    for key in ['f','c','s','h']:
        print('Loading model: {}'.format(key))
        # reconstruct then load weights
        if key=="f":
            tmp_m = F_MODEL(spe_width)
        elif key=="c":
            tmp_m = C_MODEL(spe_width)
        elif key=="s":
            tmp_m = S_MODEL(spe_width)
        elif key=="h":
            tmp_m = H_MODEL(spe_width)
        if save_format=="hd5":
            tmp_m.load_weights(os.path.join(model_path,'{}_weights.h5'.format(key)))
        elif save_format=="tf":
            tmp_m.load_weights(os.path.join(model_path, 'tf','{}_weights.tf'.format(key)))
        print('Model successfully loaded')
        loaded_models.append(tmp_m)
    print('All models successfully loaded')
    return loaded_models

def SPECTR(data, output, model_path, loaded_models = None, save_format = "tf", spe_width=304):
    # load data
    print('Loading temporary data file ({})'.format(data))
    with open(data, 'r') as f:
        json_data = json.load(f)
    print('Data successfully loaded')
    
    results = [dict(id=s['id']) for s in json_data['samples']]
        
    # load models
    models = [dict(key='f',model=None,export_shape=(spe_width,6)),
              dict(key='c',model=None,export_shape=(4,)),
              dict(key='s',model=None,export_shape=(spe_width,)),
              dict(key='h',model=None,export_shape=(1,))]
    
    if loaded_models is None:
        loaded_models = LOADMODELS(model_path, save_format, spe_width)
    
    for i,m in enumerate(models):
        m['model'] = loaded_models[i]
    
    print('Converting curves to AI-readable data')
    data = []
    for s,sample in enumerate(json_data['samples']):
        temp_curve = np.array(sample['data'])
        if temp_curve.shape[0] < spe_width:
            pad = spe_width-temp_curve.shape[0]
            pre_pad = pad//2
            post_pad = pad-pre_pad
            temp_curve = np.concatenate([np.zeros(pre_pad),temp_curve,np.zeros(post_pad)])
        if temp_curve.shape[0] > spe_width:
            raise Exception('Length of curve ({}) is larger than expected ({})'.format(temp_curve.shape[0],spe_width))
        temp_curve = temp_curve / np.max(temp_curve) # normalize
        results[s]['x'] = temp_curve.tolist() # add to results
        temp_curve = temp_curve.reshape((1,spe_width,1)) # reshape
        data.append(temp_curve)
    n = len(data)
    data = np.concatenate(data).reshape((n,spe_width,1,1))
    print('Data successfully converted')
    
    # launch prediction
    print('Running predictions')
    
    for m in models:
        print('Running predictions for model {}'.format(m['key']))
        temp_results = m['model'].predict(data)
        print('Converting output to R-readable data')
        temp_results = temp_results.reshape((n,)+m['export_shape'])
        for i in range(len(temp_results)):
            if m['key']=='f':
                for j in range(6):
                    results[i]['f{}'.format(j+1)] = temp_results[i][:,j].tolist()
            else:
                results[i][m['key']] = temp_results[i].tolist()
                
    # export
    print('Exporting temporary data file ({})'.format(output))
    with open(output, 'w') as f:
        json.dump(results, f)
    print('Data successfully exported')


if __name__ == "__main__":
    print('Loading arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--model_path", type=str)
    
    FLAGS = parser.parse_args()
    
    print("FLAGS:")
    for f in dir(FLAGS):
        if f[:1] != "_":
            print("    {}: {}".format(f,getattr(FLAGS,f)))
    print("")
    
    print('Running AI')
    SPECTR(data = FLAGS.data, output = FLAGS.output, loaded_models = None, model_path = FLAGS.model_path)
    print('Done')
