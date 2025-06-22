import os
import time
import tensorflow as tf
from dataset_loader import *
from model import ResNetGenerator, Discriminator

print("Using GPU:", tf.config.list_physical_devices('GPU'))

# Change this path if your dataset is in another location
DATA_DIR = "../apple2orange"
CHECKPOINT_DIR = "./checkpoints/apple2orange"

IMG_HEIGHT, IMG_WIDTH = 128, 128
BATCH_SIZE = 1
EPOCHS = 100  # Change to 100+ for real training

def preprocess_image(image):
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    return image

def make_dataset(path):
    dataset = image_folder_to_dataset(path)
    dataset = dataset.map(preprocess_image).batch(BATCH_SIZE)
    return dataset

trainA = make_dataset(os.path.join(DATA_DIR, "trainA"))
trainB = make_dataset(os.path.join(DATA_DIR, "trainB"))

generator_g = ResNetGenerator()  # A → B
generator_f = ResNetGenerator()  # B → A
discriminator_x = Discriminator()  # Discriminator for domain A
discriminator_y = Discriminator()  # Discriminator for domain B

loss_obj = tf.keras.losses.MeanSquaredError()

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    return (real_loss + generated_loss) * 0.5

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

LAMBDA = 10

def cycle_loss(real_image, cycled_image):
    return LAMBDA * tf.reduce_mean(tf.abs(real_image - cycled_image))

def identity_loss(real_image, same_image):
    return LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))

g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint = tf.train.Checkpoint(generator_g=generator_g,
                                 generator_f=generator_f,
                                 discriminator_x=discriminator_x,
                                 discriminator_y=discriminator_y,
                                 g_optimizer=g_optimizer,
                                 f_optimizer=f_optimizer,
                                 x_optimizer=x_optimizer,
                                 y_optimizer=y_optimizer)

manager = tf.train.CheckpointManager(checkpoint, CHECKPOINT_DIR, max_to_keep=5)

@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        fake_y = generator_g(real_x, training=True)
        cycled_x = generator_f(fake_y, training=True)

        fake_x = generator_f(real_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)

        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        total_cycle_loss = cycle_loss(real_x, cycled_x) + cycle_loss(real_y, cycled_y)
        id_loss = identity_loss(real_y, same_y) + identity_loss(real_x, same_x)

        total_gen_g = gen_g_loss + total_cycle_loss + id_loss
        total_gen_f = gen_f_loss + total_cycle_loss + id_loss

        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    g_gradients = tape.gradient(total_gen_g, generator_g.trainable_variables)
    f_gradients = tape.gradient(total_gen_f, generator_f.trainable_variables)
    x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    g_optimizer.apply_gradients(zip(g_gradients, generator_g.trainable_variables))
    f_optimizer.apply_gradients(zip(f_gradients, generator_f.trainable_variables))
    x_optimizer.apply_gradients(zip(x_gradients, discriminator_x.trainable_variables))
    y_optimizer.apply_gradients(zip(y_gradients, discriminator_y.trainable_variables))

    return {
        'gen_g_loss': gen_g_loss,
        'gen_f_loss': gen_f_loss,
        'disc_x_loss': disc_x_loss,
        'disc_y_loss': disc_y_loss,
        'cycle_loss': total_cycle_loss,
        'identity_loss': id_loss
    }

# 🚫 DO NOT clear session or garbage collect inside the training loop
# That slows down training and prevents GPU from working efficiently

SAVE_FREQ = 10  # Save model every 10 epochs

for epoch in range(EPOCHS):
    start = time.time()
    print(f"\n----- Epoch {epoch + 1} -----")

    for step, (real_x, real_y) in enumerate(tf.data.Dataset.zip((trainA, trainB))):
        losses = train_step(real_x, real_y)
        if step % 100 == 0:
            print(f"Step {step} | " + " | ".join([f"{k}: {v.numpy():.4f}" for k, v in losses.items()]))

    # ✅ Save model every N epochs
    if (epoch + 1) % SAVE_FREQ == 0 or (epoch + 1) == EPOCHS:
        manager.save()
        print(f"Checkpoint saved at epoch {epoch + 1}")

    print(f"Time taken for epoch {epoch + 1} is {time.time() - start:.2f} sec")

