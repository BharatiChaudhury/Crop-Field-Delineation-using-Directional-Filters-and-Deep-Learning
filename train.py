import prepare_data
import eff_multi_task_unet
import numpy as np
import tensorflow as tf
import os
from utils import Prediction
label1, label2, label3, std, inp_data = prepare_data.load_data()

label2 = label2.astype('float64')

input_shape = (256, 256, 32)

x_train = np.zeros((133,256,256,32))
for l in range(inp_data.shape[0]):
    for k in range(inp_data.shape[1]):
        for i in range(inp_data.shape[2]):
            for j in range(inp_data.shape[3]):
                x_train[l,i,j,k] = inp_data[l,k,i,j]
#Append standard deviation as a feature
x_train = np.append(x_train,std,axis=3)

y_train_1 = np.zeros((133,256,256))
for i in range(label1.shape[0]):
    for l in range(label1.shape[1]):
        for j in range(label1.shape[2]):
            for k in range(label1.shape[3]):
                y_train_1[i,j,k] = label1[i,l,j,k]

print(y_train_1[0,:,:].shape)

y_train_2 = label2
y_train_3 = label3

print(y_train_2.shape,y_train_1.shape, y_train_3.shape, x_train.shape)
x_train_inp = x_train

#model = eff_multi_task_unet.build_effienet_unet(input_shape)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_train_inp, y_train_1, test_size=0.20, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.20, random_state=0)

x_train_field, x_test_field, y_train_field, y_test_field = train_test_split(x_train_inp, y_train_2, test_size=0.20, random_state=42)
x_train_field, x_val_field, y_train_field, y_val_field = train_test_split(x_train_field, y_train_field, test_size=0.20, random_state=0)

x_train_field, x_test_field, y_train_extent, y_test_extent = train_test_split(x_train_inp, y_train_3, test_size=0.20, random_state=42)
x_train_field, x_val_field, y_train_extent, y_val_extent = train_test_split(x_train_field, y_train_extent, test_size=0.20, random_state=0)

print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)
print(x_val.shape, y_val.shape)

print(y_train_field.shape)
print(y_test_field.shape)
print(y_val_field.shape)

print(y_train_extent.shape)
print(y_test_extent.shape)
print(y_val_extent.shape)
#print(model.summary())

import tensorflow.keras.backend as K

def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    axes = tuple(range(1, len(y_pred.shape)-1))
    numerator = 2. * K.sum(y_pred * y_true, axes)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)
    return 1 - K.mean((numerator + epsilon) / (denominator + epsilon))
metric = tf.keras.metrics.MeanIoU(num_classes=1)
def dice_coef(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
    denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
    return (numerator + 1) / (denominator + 1)

from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.callbacks import EarlyStopping

def scheduler(epoch, lr):
    if epoch <8:
        return float(lr)
    else:
        return float(lr * tf.math.exp(-0.1))
callback = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose = 1)
"""
checkpoint_path = "training_3/eff_multiunet_cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


checkpoint = tf.keras.callbacks.ModelCheckpoint(
filepath = checkpoint_path,
save_weights_only=True, verbose = 1)
"""
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.03, mode='auto', patience=20,restore_best_weights=True)

input_shape = (256, 256, 33)
model = eff_multi_task_unet.build_effienet_unet(input_shape,Prediction.i)
Prediction.i+=1

def scheduler(epoch, lr):
    if epoch < 30:
        return 0.0001
    elif epoch > 30 and epoch<150:
        return 0.0001
    else:
        return 0.00001
    
callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

#tf.random.set_seed(1234)
loss_list = {'segment_extent': 'huber_loss','segment_field': soft_dice_loss, 'segment_boundary': soft_dice_loss}
test_metrics = {'segment_extent': 'mae','segment_field': dice_coef, 'segment_boundary': dice_coef}
loss_weights = {'segment_extent': 0.5,'segment_field': 0.5, 'segment_boundary': 0.5}

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=loss_list,
              metrics=test_metrics,
              loss_weights = loss_weights
              )

#model.compile(optimizer=adam, loss=soft_dice_loss, metrics=[dice_coef])

#model_history=model.fit(x_train, [y_train, y_train_field, y_train_extent], epochs=50, batch_size = 6, validation_data=(x_val, [y_val,y_val_field,y_val_extent]),callbacks=[checkpoint,early_stop,callback])

model_history=model.fit(x_train, [y_train,y_train_field,y_train_extent], epochs=60, batch_size = 6, validation_data=(x_val, [y_val,y_val_field,y_val_extent]))


data_file_ = ['clipped_small_parcels.tif', 'clipped_small_parcels_3.tif','clipped_small_parcels_4.tif']
p = Prediction(model=model)
#p.mIoU(x_test,y_test)
p.loss_plots(model_history)

for file in data_file_: 
    p = Prediction(data_file=file,model=model)
    p.pred_data()
#Prediction.i+=3