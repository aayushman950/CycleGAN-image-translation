import time
from dataset_loader import *
from model import *
import flask
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

checkpoint_path = os.path.dirname(__file__) + "/checkpoints/monet"

# Load pre-trained model
generator_a2b = ResNetGenerator()
generator_b2a = ResNetGenerator()

ckpt = tf.train.Checkpoint(generator_a2b=generator_a2b,
                           generator_b2a=generator_b2a)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)


def preprocess(image):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.expand_dims(image, 0)
    return image


def generate_image(image_path, save_path, mode=2):
    last_time = time.time()

    if mode == 1:
        model = generator_a2b
    elif mode == 2:
        model = generator_b2a

    image = dataset_loader.load_image(image_path)
    image = preprocess(image)
    prediction = model(image)

    tf.keras.preprocessing.image.save_img(save_path, prediction[0].numpy())
    print(time.time() - last_time)


if ckpt_manager.latest_checkpoint:
    checkpoint = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint Epoch {} restored'.format(checkpoint))
