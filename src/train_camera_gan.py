import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "camera_trajectory_sequences.npz"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

GRAPH_DIR = os.path.join(
    BASE_DIR,
    "results",
    "graphs"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(GRAPH_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

PAST_FRAMES = 20
FUTURE_FRAMES = 10
FEATURES = 5
TARGETS = 2

LATENT_DIM = 16
BATCH_SIZE = 32
EPOCHS = 30

LEARNING_RATE = 0.0002


# ============================================================
# LOAD CAMERA DATA
# ============================================================

print("\nLoading camera trajectory data...")

data = np.load(DATA_FILE)

X_train = data["X_train"]
Y_train = data["Y_train"]

X_val = data["X_val"]
Y_val = data["Y_val"]

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)

print("X_val:", X_val.shape)
print("Y_val:", Y_val.shape)


# ============================================================
# TARGET SCALING
# ============================================================

# Y values are currently in world coordinates.
# GAN output will be trained in normalized target space.

target_min = np.min(Y_train, axis=(0, 1))
target_max = np.max(Y_train, axis=(0, 1))

target_range = target_max - target_min
target_range[target_range == 0] = 1.0


def scale_targets(y):

    return (
        y - target_min
    ) / target_range


Y_train_scaled = scale_targets(Y_train)
Y_val_scaled = scale_targets(Y_val)

print("\nTarget scaling complete.")

print("Target min:", target_min)
print("Target max:", target_max)


# ============================================================
# GENERATOR
# ============================================================

def build_generator():

    trajectory_input = layers.Input(
        shape=(PAST_FRAMES, FEATURES),
        name="past_trajectory"
    )

    noise_input = layers.Input(
        shape=(LATENT_DIM,),
        name="noise"
    )

    x = layers.LSTM(
        128,
        return_sequences=False
    )(trajectory_input)

    x = layers.Concatenate()(
        [x, noise_input]
    )

    x = layers.RepeatVector(
        FUTURE_FRAMES
    )(x)

    x = layers.LSTM(
        128,
        return_sequences=True
    )(x)

    output = layers.TimeDistributed(
        layers.Dense(TARGETS, activation="sigmoid")
    )(x)

    return tf.keras.Model(
        [trajectory_input, noise_input],
        output,
        name="camera_trajectory_generator"
    )


# ============================================================
# DISCRIMINATOR
# ============================================================

def build_discriminator():

    future_input = layers.Input(
        shape=(FUTURE_FRAMES, TARGETS),
        name="future_trajectory"
    )

    x = layers.LSTM(
        128,
        return_sequences=False
    )(future_input)

    x = layers.Dense(
        64,
        activation="relu"
    )(x)

    x = layers.Dropout(0.3)(x)

    output = layers.Dense(
        1,
        activation="sigmoid"
    )(x)

    return tf.keras.Model(
        future_input,
        output,
        name="camera_trajectory_discriminator"
    )


generator = build_generator()
discriminator = build_discriminator()

print("\nGenerator:")
generator.summary()

print("\nDiscriminator:")
discriminator.summary()


# ============================================================
# OPTIMIZERS
# ============================================================

generator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.5
)

discriminator_optimizer = tf.keras.optimizers.Adam(
    learning_rate=LEARNING_RATE,
    beta_1=0.5
)


# ============================================================
# LOSS
# ============================================================

binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output, fake_output):

    real_loss = binary_cross_entropy(
        tf.ones_like(real_output),
        real_output
    )

    fake_loss = binary_cross_entropy(
        tf.zeros_like(fake_output),
        fake_output
    )

    return real_loss + fake_loss


def generator_loss(fake_output):

    return binary_cross_entropy(
        tf.ones_like(fake_output),
        fake_output
    )


# ============================================================
# TRAINING STEP
# ============================================================

@tf.function
def train_step(past, real_future):

    batch_size = tf.shape(past)[0]

    noise = tf.random.normal(
        [batch_size, LATENT_DIM]
    )

    with tf.GradientTape() as gen_tape, \
         tf.GradientTape() as disc_tape:

        fake_future = generator(
            [past, noise],
            training=True
        )

        real_output = discriminator(
            real_future,
            training=True
        )

        fake_output = discriminator(
            fake_future,
            training=True
        )

        gen_loss = generator_loss(
            fake_output
        )

        disc_loss = discriminator_loss(
            real_output,
            fake_output
        )

    generator_gradients = gen_tape.gradient(
        gen_loss,
        generator.trainable_variables
    )

    discriminator_gradients = disc_tape.gradient(
        disc_loss,
        discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(
            generator_gradients,
            generator.trainable_variables
        )
    )

    discriminator_optimizer.apply_gradients(
        zip(
            discriminator_gradients,
            discriminator.trainable_variables
        )
    )

    return gen_loss, disc_loss


# ============================================================
# DATASET
# ============================================================

train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, Y_train_scaled)
)

train_dataset = train_dataset.shuffle(
    len(X_train)
)

train_dataset = train_dataset.batch(
    BATCH_SIZE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# TRAINING
# ============================================================

generator_losses = []
discriminator_losses = []

print("\n======================================")
print("CAMERA GAN TRAINING STARTED")
print("======================================")

for epoch in range(EPOCHS):

    epoch_gen_loss = []
    epoch_disc_loss = []

    for past, real_future in train_dataset:

        gen_loss, disc_loss = train_step(
            past,
            real_future
        )

        epoch_gen_loss.append(
            float(gen_loss.numpy())
        )

        epoch_disc_loss.append(
            float(disc_loss.numpy())
        )

    avg_gen_loss = np.mean(
        epoch_gen_loss
    )

    avg_disc_loss = np.mean(
        epoch_disc_loss
    )

    generator_losses.append(
        avg_gen_loss
    )

    discriminator_losses.append(
        avg_disc_loss
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Generator Loss: {avg_gen_loss:.4f} | "
        f"Discriminator Loss: {avg_disc_loss:.4f}"
    )


# ============================================================
# SAVE MODELS
# ============================================================

generator_file = os.path.join(
    MODEL_DIR,
    "camera_trajectory_generator.keras"
)

discriminator_file = os.path.join(
    MODEL_DIR,
    "camera_trajectory_discriminator.keras"
)

generator.save(generator_file)
discriminator.save(discriminator_file)


# ============================================================
# SAVE TARGET SCALING
# ============================================================

import joblib

target_scaler_file = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "camera_gan_target_scaler.pkl"
)

joblib.dump(
    {
        "min": target_min,
        "max": target_max
    },
    target_scaler_file
)


# ============================================================
# TRAINING GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    generator_losses,
    label="Generator Loss"
)

plt.plot(
    discriminator_losses,
    label="Discriminator Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Camera GAN Training Loss")

plt.legend()
plt.grid(True)
plt.tight_layout()

graph_file = os.path.join(
    GRAPH_DIR,
    "camera_gan_training_loss.png"
)

plt.savefig(
    graph_file,
    dpi=200
)

plt.close()


# ============================================================
# TEST GENERATION
# ============================================================

sample_index = 0

past_sample = X_train[
    sample_index:sample_index + 1
]

noise = tf.random.normal(
    [1, LATENT_DIM]
)

past_sample_tensor = tf.convert_to_tensor(
    past_sample,
    dtype=tf.float32
)

generated = generator(
    [past_sample_tensor, noise],
    training=False
).numpy()


# Inverse target scaling

generated_world = (
    generated * target_range
) + target_min


actual_world = Y_train[
    sample_index
]


# ============================================================
# GENERATED TRAJECTORY GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    actual_world[:, 0],
    actual_world[:, 1],
    marker="o",
    label="Actual"
)

plt.plot(
    generated_world[0, :, 0],
    generated_world[0, :, 1],
    marker="x",
    label="GAN Generated"
)

plt.xlabel("World X")
plt.ylabel("World Y")

plt.title(
    "Camera GAN Generated Trajectory"
)

plt.legend()
plt.grid(True)

plt.tight_layout()

generated_graph = os.path.join(
    GRAPH_DIR,
    "camera_gan_generated_trajectory.png"
)

plt.savefig(
    generated_graph,
    dpi=200
)

plt.close()


# ============================================================
# FINAL
# ============================================================

print("\n======================================")
print("CAMERA GAN TRAINING COMPLETE")
print("======================================")

print("\nGenerator saved:")
print(generator_file)

print("\nDiscriminator saved:")
print(discriminator_file)

print("\nTarget scaler saved:")
print(target_scaler_file)

print("\nTraining graph:")
print(graph_file)

print("\nGenerated trajectory graph:")
print(generated_graph)