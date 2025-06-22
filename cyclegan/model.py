from layer import *

# ResNet Generator
class ResNetGenerator(tf.keras.Model):
    def __init__(self, out_channels=3, nker=64, norm='inorm'):
        super(ResNetGenerator, self).__init__()

        self.enc1 = CNR2d(1 * nker, kernel_size=7, stride=1, norm=norm, relu=0.0, padding=3)
        self.enc2 = CNR2d(2 * nker, kernel_size=3, stride=2, norm=norm, relu=0.0)
        self.enc3 = CNR2d(4 * nker, kernel_size=3, stride=2, norm=norm, relu=0.0)

        self.res1 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res2 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res3 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res4 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res5 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res6 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res7 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res8 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)
        self.res9 = ResBlock(4 * nker, kernel_size=3, stride=1, norm=norm, relu=0.0, padding=1)

        self.dec1 = DECNR2d(2 * nker, kernel_size=3, stride=2, norm=norm, relu=0.0)
        self.dec2 = DECNR2d(1 * nker, kernel_size=3, stride=2, norm=norm, relu=0.0)
        self.dec3 = CNR2d(out_channels, kernel_size=7, stride=1, norm=None, relu=None, bias=False, padding=3)

    def call(self, x, training=False):
        x = self.enc1(x, training=training)
        x = self.enc2(x, training=training)
        x = self.enc3(x, training=training)

        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        x = self.res3(x, training=training)
        x = self.res4(x, training=training)
        x = self.res5(x, training=training)
        x = self.res6(x, training=training)
        x = self.res7(x, training=training)
        x = self.res8(x, training=training)
        x = self.res9(x, training=training)

        x = self.dec1(x, training=training)
        x = self.dec2(x, training=training)
        x = self.dec3(x, training=training)

        return tf.nn.tanh(x)


# Patch GAN
class Discriminator(tf.keras.Model):
    def __init__(self, out_channels=1, nker=64, norm='inorm', lsgan=True):
        super(Discriminator, self).__init__()

        self.lsgan = lsgan

        self.dsc1 = CNR2d(1 * nker, kernel_size=4, stride=2, norm=None, relu=0.2)
        self.dsc2 = CNR2d(2 * nker, kernel_size=4, stride=2, norm=norm, relu=0.2)
        self.dsc3 = CNR2d(4 * nker, kernel_size=4, stride=2, norm=norm, relu=0.2)
        self.dsc4 = CNR2d(8 * nker, kernel_size=4, stride=1, norm=norm, relu=0.2, padding=1, padding_type='ZERO')
        self.dsc5 = CNR2d(out_channels, kernel_size=4, stride=1, norm=None, relu=None, bias=False, padding=1, padding_type='ZERO')

    def call(self, x, training=False):
        x = self.dsc1(x)
        x = self.dsc2(x)
        x = self.dsc3(x)
        x = self.dsc4(x)
        x = self.dsc5(x)
        if not self.lsgan:
            x = tf.nn.sigmoid(x)

        return x

