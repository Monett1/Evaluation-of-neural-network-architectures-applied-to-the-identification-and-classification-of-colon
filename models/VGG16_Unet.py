from keras.layers import Input, Concatenate, Conv2D, Conv2DTranspose, BatchNormalization, ZeroPadding2D, MaxPooling2D, Conv2D,PReLU, SpatialDropout2D, UpSampling2D, ReLU
from tensorflow.keras.optimizers import Adam
from keras.activations import softmax
from keras.models import Model
from tensorflow.keras.layers import Input, Add, Permute, Activation
from tensorflow.keras.initializers import RandomNormal

from utils.metrics import dice_0, dice_1, dice_2, dice_3, dice_4, dice_5, dice_6, iou, categorical_dice, categorical_dice_loss

def conv_block(inputs,num_filters):
  x = Conv2D(num_filters,3,padding='same')(inputs)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(num_filters,3,padding='same')(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  return x
def define_decoder(inputs,skip_layer,num_filters):
  init = RandomNormal(stddev=0.02)
  x = Conv2DTranspose(num_filters,(2,2),strides=(2,2),padding='same',kernel_initializer=init)(inputs)
  g = Concatenate()([x,skip_layer])
  g = conv_block(g,num_filters)
  return g

from tensorflow.keras.applications.vgg16 import VGG16

vgg16 = VGG16(include_top=False,weights=None)

def get_vgg16_unet(input_shape):
  inputs = Input(shape=input_shape)
  vgg16 = VGG16(include_top=False,weights=None,input_tensor=inputs)  # We will extract encoder layers based on their output shape from vgg16 model
  s1 = vgg16.get_layer('block1_conv2').output
  s2 = vgg16.get_layer('block2_conv2').output
  s3 = vgg16.get_layer('block3_conv3').output
  s4 = vgg16.get_layer('block4_conv3').output    # bottleneck/bridege layer from vgg16
  b1 = vgg16.get_layer('block5_conv3').output #32

  # Decoder Block
  d1 = define_decoder(b1,s4,512)
  d2 = define_decoder(d1,s3,256)
  d3 = define_decoder(d2,s2,128)
  d4 = define_decoder(d3,s1,64)  #output layer
  outputs = Conv2D(7, 1, activation='softmax')(d4)
  model = Model(inputs=[inputs], outputs=[outputs])
  loss = categorical_dice_loss

  model.compile(optimizer=Adam(lr=1e-3), loss=loss, metrics=[dice_0, dice_1, dice_2, dice_3, dice_4, dice_5, dice_6, iou, categorical_dice])

  return model
