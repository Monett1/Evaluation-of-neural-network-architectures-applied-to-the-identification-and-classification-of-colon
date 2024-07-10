from keras.layers import Input, concatenate, Conv2D, Conv2DTranspose, BatchNormalization, ZeroPadding2D, MaxPooling2D, Conv2D,PReLU, SpatialDropout2D, UpSampling2D, ReLU
from tensorflow.keras.optimizers import Adam
from keras.activations import softmax
from keras.models import Model
from tensorflow.keras.layers import Input, Add, Permute, Activation

from utils.metrics import dice_0, dice_1, dice_2, dice_3, dice_4, dice_5, dice_6, iou, categorical_dice, categorical_dice_loss


def initial_block(inp):
    inp1 = inp
    conv = Conv2D(filters=13, kernel_size=3, strides=2,
                  padding='same', kernel_initializer='he_normal')(inp)
    pool = MaxPooling2D(2)(inp1)
    concat = concatenate([conv, pool])
    return concat


def encoder_bottleneck(inp, filters, name, dilation_rate=2, downsample=False, dilated=False, asymmetric=False, drop_rate=0.1):
    reduce = filters // 4
    down = inp
    kernel_stride = 1

    # Downsample
    if downsample:
        kernel_stride = 2
        pad_activations = filters - inp.shape.as_list()[-1]
        down = MaxPooling2D(2)(down)
        down = Permute(dims=(1, 3, 2))(down)
        down = ZeroPadding2D(padding=((0, 0), (0, pad_activations)))(down)
        down = Permute(dims=(1, 3, 2))(down)

    # 1*1 Reduce
    x = Conv2D(filters=reduce, kernel_size=kernel_stride, strides=kernel_stride, padding='same',
               use_bias=False, kernel_initializer='he_normal', name=f'{name}_reduce')(inp)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # Conv
    if not dilated and not asymmetric:
        x = Conv2D(filters=reduce, kernel_size=3, padding='same',
                   kernel_initializer='he_normal', name=f'{name}_conv_reg')(x)
    elif dilated:
        x = Conv2D(filters=reduce, kernel_size=3, padding='same', dilation_rate=dilation_rate,
                   kernel_initializer='he_normal', name=f'{name}_reduce_dilated')(x)
    elif asymmetric:
        x = Conv2D(filters=reduce, kernel_size=(1, 5), padding='same', use_bias=False,
                   kernel_initializer='he_normal', name=f'{name}_asymmetric')(x)
        x = Conv2D(filters=reduce, kernel_size=(5, 1), padding='same',
                   kernel_initializer='he_normal', name=name)(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # 1*1 Expand
    x = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False,
               kernel_initializer='he_normal', name=f'{name}_expand')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = SpatialDropout2D(rate=drop_rate)(x)

    concat = Add()([x, down])
    concat = PReLU(shared_axes=[1, 2])(concat)
    return concat


def decoder_bottleneck(inp, filters, name, upsample=False):
    reduce = filters // 4
    up = inp

    # Upsample
    if upsample:
        up = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same',
                    use_bias=False, kernel_initializer='he_normal', name=f'{name}_upsample')(up)
        up = UpSampling2D(size=2)(up)

    # 1*1 Reduce
    x = Conv2D(filters=reduce, kernel_size=1, strides=1, padding='same',
               use_bias=False, kernel_initializer='he_normal', name=f'{name}_reduce')(inp)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # Conv
    if not upsample:
        x = Conv2D(filters=reduce, kernel_size=3, strides=1, padding='same',
                   kernel_initializer='he_normal', name=f'{name}_conv_reg')(x)
    else:
        x = Conv2DTranspose(filters=reduce, kernel_size=3, strides=2, padding='same',
                            kernel_initializer='he_normal', name=f'{name}_transpose')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = PReLU(shared_axes=[1, 2])(x)

    # 1*1 Expand
    x = Conv2D(filters=filters, kernel_size=1, strides=1, padding='same',
               use_bias=False, kernel_initializer='he_normal', name=f'{name}_expand')(x)
    x = BatchNormalization(momentum=0.1)(x)

    concat = Add()([x, up])
    concat = ReLU()(concat)

    return concat

def get_ENet(H=256, W=256, nclasses=7):
    '''
    '''

    print('Loading ENet')
    inp = Input(shape=(H, W, 3))
    enc = initial_block(inp)

    # Bottleneck 1.0
    enc = encoder_bottleneck(enc, 64, name='enc1',
                             downsample=True, drop_rate=0.001)

    enc = encoder_bottleneck(enc, 64, name='enc1.1', drop_rate=0.001)
    enc = encoder_bottleneck(enc, 64, name='enc1.2', drop_rate=0.001)
    enc = encoder_bottleneck(enc, 64, name='enc1.3', drop_rate=0.001)
    enc = encoder_bottleneck(enc, 64, name='enc1.4', drop_rate=0.001)

    enc = encoder_bottleneck(enc, 64, name='enc1.5', drop_rate=0.001)
    enc = encoder_bottleneck(enc, 64, name='enc1.6', drop_rate=0.001)
    enc = encoder_bottleneck(enc, 64, name='enc1.7', drop_rate=0.001)

    # Bottleneck 2.0
    enc = encoder_bottleneck(enc, 128, name='enc2.0', downsample=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.1')
    enc = encoder_bottleneck(enc, 128, name='enc2.2',
                             dilation_rate=2, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.3', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.4',
                             dilation_rate=4, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.5', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.6',
                             dilation_rate=6, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.7', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.8',
                             dilation_rate=8, dilated=True)

    enc = encoder_bottleneck(enc, 128, name='enc2.9')
    enc = encoder_bottleneck(enc, 128, name='enc2.10',
                             dilation_rate=10, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.11', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.12',
                             dilation_rate=12, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.13', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.14',
                             dilation_rate=14, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.15', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc2.16',
                             dilation_rate=16, dilated=True)

    # Bottleneck 3.0
    enc = encoder_bottleneck(enc, 128, name='enc3.0')
    enc = encoder_bottleneck(enc, 128, name='enc3.1',
                             dilation_rate=2, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.2', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.3',
                             dilation_rate=4, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.4')
    enc = encoder_bottleneck(enc, 128, name='enc3.5',
                             dilation_rate=6, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.6')
    enc = encoder_bottleneck(enc, 128, name='enc3.7',
                             dilation_rate=8, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.8', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.9',
                             dilation_rate=10, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.10', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.11',
                             dilation_rate=12, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.12', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.13',
                             dilation_rate=14, dilated=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.14', asymmetric=True)
    enc = encoder_bottleneck(enc, 128, name='enc3.15',
                             dilation_rate=16, dilated=True)




    # Bottleneck 4.0
    dec = decoder_bottleneck(enc, 64, name='dec4.0', upsample=True)
    dec = decoder_bottleneck(dec, 64, name='dec4.1')
    dec = decoder_bottleneck(dec, 64, name='dec4.2')
    dec = decoder_bottleneck(dec, 64, name='dec4.3')
    dec = decoder_bottleneck(dec, 64, name='dec4.4')

    # Bottleneck 5.0
    dec = decoder_bottleneck(dec, 16, name='dec5.0', upsample=True)
    dec = decoder_bottleneck(dec, 16, name='dec5.1')
    dec = decoder_bottleneck(dec, 16, name='dec5.2')
    dec = decoder_bottleneck(dec, 16, name='dec5.3')

    dec = Conv2DTranspose(filters=nclasses, kernel_size=2, strides=2,
                          padding='same', kernel_initializer='he_normal', name='fullconv')(dec)
    dec = Activation('softmax')(dec)

    model = Model(inputs=inp, outputs=dec, name='Enet')

    loss = categorical_dice_loss

    model.compile(optimizer=Adam(lr=1e-3), loss=loss, metrics=[dice_0, dice_1, dice_2, dice_3, dice_4, dice_5, dice_6, iou, categorical_dice])

    return model