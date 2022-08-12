from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input,Flatten, Dropout
from tensorflow.keras.models import Model
from keras import regularizers
from tensorflow.keras.applications import EfficientNetB3
import tensorflow as tf

print("TF Version: ", tf.__version__)

def conv_block(inputs, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def decoder_block(inputs, skip, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip])
    x = conv_block(x, num_filters)
    return x

def build_effienet_unet(input_shape,i):
    """ Input """
    inputs = Input(input_shape)
    print(inputs)
    """ Pre-trained Encoder """
    encoder = EfficientNetB3(include_top=False, weights=None, input_tensor=inputs)
    for layer in encoder.layers:
        layer.trainable = True

    s1 = encoder.get_layer("input_"+str(i)).output                      ## 256
    s2 = encoder.get_layer("block2a_expand_activation").output    ## 128
    s3 = encoder.get_layer("block3a_expand_activation").output    ## 64
    s4 = encoder.get_layer("block4a_expand_activation").output    ## 32

    """ Bottleneck """
    b1 = encoder.get_layer("block6a_expand_activation").output    ## 16

    """ Decoder """
    #b1 = Dropout(0.5,seed=1234)(b1)
    d1 = decoder_block(b1, s4, 256)                               ## 32
    d2 = decoder_block(d1, s3, 128)                               ## 64
    d3 = decoder_block(d2, s2, 64)                               ## 128
    d4 = decoder_block(d3, s1, 32)                                ## 256

    """ Output """
    y1 = Conv2D(1,(1,1), padding="same", activation="sigmoid",name="segment_boundary")(d4)
    y2 = Conv2D(1,(1,1), padding="same", activation="sigmoid",name="segment_field")(d4)
    y3 = Conv2D(1,(1,1), padding="same", activation="sigmoid",name="segment_extent")(d4)

    #outputs = Flatten()(outputs)
    #print(outputs)
    
    model = Model(inputs=inputs, outputs=[y1,y2,y3], name="segment")
    print(inputs)
    return model

