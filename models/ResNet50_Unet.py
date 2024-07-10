from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam


from utils.metrics import dice_0, dice_1, dice_2, dice_3, dice_4, dice_5, dice_6, iou, categorical_dice, categorical_dice_loss


def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def decoder_block(input, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def get_resnet50_unet(input_shape=(256,256,3)):
    """ Input """
    inputs = Input(input_shape)

    """ Pre-trained ResNet50 Model """
    resnet50 = ResNet50(include_top=False, weights=None, input_tensor=inputs)

    """ Encoder """
    s1 = resnet50.get_layer("input_1").output           ## (512 x 512)
    s2 = resnet50.get_layer("conv1_relu").output        ## (256 x 256)
    s3 = resnet50.get_layer("conv2_block3_out").output  ## (128 x 128)
    s4 = resnet50.get_layer("conv3_block4_out").output  ## (64 x 64)

    """ Bridge """
    b1 = resnet50.get_layer("conv4_block6_out").output  ## (32 x 32)
  

    """ Decoder """
    d1 = decoder_block(b1, s4, 1024)                     ## (16 x 16)
    d2 = decoder_block(d1, s3, 512)                     ## (32 x 32)
    d3 = decoder_block(d2, s2, 256)                     ## (64 x 64)
    d4 = decoder_block(d3, s1, 64)                      ## (128 x 128)

    """ Output """
    outputs = Conv2D(7, 1, padding="same", activation="softmax")(d4)

    model = Model(inputs, outputs)

    loss= categorical_dice_loss

    model.compile(optimizer=Adam(lr=1e-3), loss=loss, metrics=[dice_0, dice_1, dice_2, dice_3, dice_4, dice_5, dice_6, iou, categorical_dice])

    return model







    # """ Encoder """

    # s1 = resnet50.get_layer("conv1_relu").output           ## (256,256,64)
    # s2 = resnet50.get_layer("conv2_block3_out").output        ## (128, 128, 256)
    # s3 = resnet50.get_layer("conv3_block4_out").output  ## (32, 32, 512)
    # s4 = resnet50.get_layer("conv4_block6_out").output  ## (16, 16, 1024)

    # """ Bridge """
    # b1 = resnet50.get_layer("conv5_block3_out").output  ## (8 x 8, 2048)