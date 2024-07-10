from keras.layers import Input, Concatenate, SeparableConv2D, Conv2D, Conv2DTranspose, BatchNormalization, ZeroPadding2D, MaxPooling2D, Conv2D,PReLU, SpatialDropout2D, UpSampling2D, ReLU
from tensorflow.keras.optimizers import Adam
from keras.activations import softmax
from keras.models import Model
from tensorflow.keras.layers import Input, Add, Permute, Activation

from utils.metrics import dice_0, dice_1, dice_2, dice_3, dice_4, dice_5, dice_6, iou, categorical_dice, categorical_dice_loss


def aspp_separableConvolution(x,filters):
  c=SeparableConv2D(filters,(3,3),strides=1,padding="same",dilation_rate=2)(x)
  c=Activation("relu")(c)
  c=SeparableConv2D(filters,(1,1),strides=1,padding="same")(c)
  c=Activation("relu")(c)
  c=SeparableConv2D(filters,(1,1),strides=1,padding="same")(c)
  c=Activation("relu")(c)

  a=SeparableConv2D(filters,(3,3),strides=1,padding="same",dilation_rate=4)(x)
  a=Activation("relu")(a)
  a=SeparableConv2D(filters,(1,1),strides=1,padding="same")(a)
  a=Activation("relu")(a)
  a=SeparableConv2D(filters,(1,1),strides=1,padding="same")(a)
  a=Activation("relu")(a)

  b=SeparableConv2D(filters,(3,3),strides=1,padding="same",dilation_rate=6)(x)
  b=Activation("relu")(b)
  b=SeparableConv2D(filters,(1,1),strides=1,padding="same")(b)
  b=Activation("relu")(b)
  b=SeparableConv2D(filters,(1,1),strides=1,padding="same")(b)
  b=Activation("relu")(b)

  d=SeparableConv2D(filters,(3,3),strides=1,padding="same",dilation_rate=8)(x)
  d=Activation("relu")(d)
  d=SeparableConv2D(filters,(1,1),strides=1,padding="same")(d)
  d=Activation("relu")(d)
  d=SeparableConv2D(filters,(1,1),strides=1,padding="same")(d)
  d=Activation("relu")(d)

  out=Concatenate()([c,a,b,d])

  out=SeparableConv2D(256,(1,1),strides=1,padding="same")(out)
  out=Activation("relu")(out)

  return out

def get_SCPPnet():

  inputs= Input((256,256,3))

  #Encoder
  x1=Conv2D(32,(3,3),strides=1,padding="same")(inputs)
  x1 = BatchNormalization()(x1)
  x1=Activation("relu")(x1)
  x1=Conv2D(32,(3,3),strides=1,padding="same")(x1)
  x1=Activation("relu")(x1)

  p1=MaxPooling2D(pool_size=(2,2))(x1)

  x2=Conv2D(64,(3,3),strides=1,padding="same")(p1)
  x2 = BatchNormalization()(x2)
  x2=Activation("relu")(x2)
  x2=Conv2D(64,(3,3),strides=1,padding="same")(x2)
  x2=Activation("relu")(x2)

  p2=MaxPooling2D(pool_size=(2,2))(x2)

  x3=Conv2D(128,(3,3),strides=1,padding="same")(p2)
  x3 = BatchNormalization()(x3)
  x3=Activation("relu")(x3)
  x3=Conv2D(128,(3,3),strides=1,padding="same")(x3)
  x3=Activation("relu")(x3)
  x3=Conv2D(128,(3,3),strides=1,padding="same")(x3)
  x3=Activation("relu")(x3)

  p3=MaxPooling2D(pool_size=(2,2))(x3)

  x4=Conv2D(256,(3,3),strides=1,padding="same")(p3)
  x4 = BatchNormalization()(x4)
  x4=Activation("relu")(x4)
  x4=Conv2D(256,(3,3),strides=1,padding="same")(x4)
  x4=Activation("relu")(x4)
  x4=Conv2D(256,(3,3),strides=1,padding="same")(x4)
  x4 = BatchNormalization()(x4)
  x4=Activation("relu")(x4)
  p4=MaxPooling2D(pool_size=(2,2))(x4)

  #aspp
  p4=aspp_separableConvolution(p4,512)


  u1=UpSampling2D(size=(16,16))(p4) #16

  u1=Conv2D(2,(3,3),strides=1,padding="same")(u1)
  u1= BatchNormalization()(u1)
  u1=Activation("relu")(u1)
  u1=Conv2D(2,(1,1),strides=1,padding="same")(u1)
  u1=Activation("relu")(u1)



  u2=UpSampling2D(size=(8,8))(x4) #8

  u2=Conv2D(2,(3,3),strides=1,padding="same")(u2)
  u2 = BatchNormalization()(u2)
  u2=Activation("relu")(u2)
  u2=Conv2D(2,(1,1),strides=1,padding="same")(u2)
  u2=Activation("relu")(u2)


  u3=UpSampling2D(size=(4,4))(x3)

  u3=Conv2D(2,(3,3),strides=1,padding="same")(u3)
  u3 = BatchNormalization()(u3)
  u3=Activation("relu")(u3)
  u3=Conv2D(2,(1,1),strides=1,padding="same")(u3)
  u3=Activation("relu")(u3)



  output = Add()([u1,u2,u3])


  output=Conv2D(7,(1,1),activation='softmax')(output)

  model = Model(inputs,output)

  loss = categorical_dice_loss

  model.compile(optimizer=Adam(lr=1e-3), loss=loss, metrics=[dice_0, dice_1, dice_2, dice_3, dice_4, dice_5, dice_6, iou, categorical_dice])

  return model