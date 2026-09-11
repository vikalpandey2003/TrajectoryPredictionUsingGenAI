import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    RepeatVector,
    TimeDistributed,
    Concatenate
)
from tensorflow.keras.optimizers import Adam


# =====================================================
# CONFIGURATION
# =====================================================

DATA_PATH = "../data/processed/trajectory_sequences.npz"

PAST_FRAMES = 20
FUTURE_FRAMES = 10
FEATURES = 5

EPOCHS = 30
BATCH_SIZE = 64

LATENT_DIM = 16


print("========================================")
print("   LSTM-GAN TRAJECTORY GENERATION")
print("========================================")


# =====================================================
# LOAD DATA
# =====================================================

data = np.load(DATA_PATH)

X_train = data["X_train"]
Y_train = data["Y_train"]

X_val = data["X_val"]
Y_val = data["Y_val"]

X_test = data["X_test"]
Y_test = data["Y_test"]


print("\nDataset Loaded!")

print("X_train:", X_train.shape)
print("Y_train:", Y_train.shape)

print("X_val  :", X_val.shape)
print("Y_val  :", Y_val.shape)

print("X_test :", X_test.shape)
print("Y_test :", Y_test.shape)


# =====================================================
# GENERATOR
# =====================================================

def build_generator():

    past_input = Input(
        shape=(PAST_FRAMES, FEATURES),
        name="past_trajectory"
    )

    noise_input = Input(
        shape=(LATENT_DIM,),
        name="noise"
    )

    # Encode historical trajectory
    encoded = LSTM(
        128,
        activation="tanh"
    )(past_input)

    # Combine trajectory representation with random noise
    combined = Concatenate()(
        [encoded, noise_input]
    )

    # Convert to future sequence
    repeated = RepeatVector(FUTURE_FRAMES)(
        combined
    )

    decoded = LSTM(
        128,
        activation="tanh",
        return_sequences=True
    )(repeated)

    output = TimeDistributed(
        Dense(2)
    )(decoded)

    model = Model(
        [past_input, noise_input],
        output,
        name="Trajectory_Generator"
    )

    return model


# =====================================================
# DISCRIMINATOR
# =====================================================

def build_discriminator():

    trajectory_input = Input(
        shape=(FUTURE_FRAMES, 2),
        name="future_trajectory"
    )

    x = LSTM(
        128,
        activation="tanh"
    )(trajectory_input)

    x = Dense(
        64,
        activation="relu"
    )(x)

    output = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        trajectory_input,
        output,
        name="Trajectory_Discriminator"
    )

    return model


# =====================================================
# CREATE MODELS
# =====================================================

generator = build_generator()

discriminator = build_discriminator()


print("\n========================================")
print(" GENERATOR SUMMARY")
print("========================================")

generator.summary()


print("\n========================================")
print(" DISCRIMINATOR SUMMARY")
print("========================================")

discriminator.summary()


# =====================================================
# OPTIMIZERS
# =====================================================

generator_optimizer = Adam(
    learning_rate=0.0002,
    beta_1=0.5
)

discriminator_optimizer = Adam(
    learning_rate=0.0002,
    beta_1=0.5
)


# =====================================================
# LOSS FUNCTION
# =====================================================

binary_cross_entropy = \
    __import__("tensorflow").keras.losses.BinaryCrossentropy()


# =====================================================
# TRAINING
# =====================================================

generator_losses = []
discriminator_losses = []


num_samples = len(X_train)

print("\n========================================")
print(" STARTING GAN TRAINING")
print("========================================")


for epoch in range(EPOCHS):

    # Shuffle training samples
    indices = np.random.permutation(num_samples)

    X_train_shuffled = X_train[indices]
    Y_train_shuffled = Y_train[indices]


    epoch_g_loss = []
    epoch_d_loss = []


    for start in range(
        0,
        num_samples,
        BATCH_SIZE
    ):

        end = min(
            start + BATCH_SIZE,
            num_samples
        )

        past_batch = X_train_shuffled[start:end]
        real_future = Y_train_shuffled[start:end]

        current_batch_size = len(past_batch)


        # -------------------------------------------------
        # Generate random noise
        # -------------------------------------------------

        noise = np.random.normal(
            0,
            1,
            size=(current_batch_size, LATENT_DIM)
        ).astype(np.float32)


        # -------------------------------------------------
        # Generate fake trajectories
        # -------------------------------------------------

        fake_future = generator(
            [past_batch, noise],
            training=True
        )


        # -------------------------------------------------
        # Train discriminator
        # -------------------------------------------------

        real_labels = np.ones(
            (current_batch_size, 1)
        )

        fake_labels = np.zeros(
            (current_batch_size, 1)
        )


        with __import__("tensorflow").GradientTape() as tape:

            real_prediction = discriminator(
                real_future,
                training=True
            )

            fake_prediction = discriminator(
                fake_future,
                training=True
            )


            real_loss = binary_cross_entropy(
                real_labels,
                real_prediction
            )

            fake_loss = binary_cross_entropy(
                fake_labels,
                fake_prediction
            )


            d_loss = (
                real_loss + fake_loss
            ) / 2


        discriminator_gradients = tape.gradient(
            d_loss,
            discriminator.trainable_variables
        )


        discriminator_optimizer.apply_gradients(
            zip(
                discriminator_gradients,
                discriminator.trainable_variables
            )
        )


        # -------------------------------------------------
        # Train generator
        # -------------------------------------------------

        noise = np.random.normal(
            0,
            1,
            size=(current_batch_size, LATENT_DIM)
        ).astype(np.float32)


        with __import__("tensorflow").GradientTape() as tape:

            fake_future = generator(
                [past_batch, noise],
                training=True
            )

            fake_prediction = discriminator(
                fake_future,
                training=True
            )


            g_loss = binary_cross_entropy(
                real_labels,
                fake_prediction
            )


        generator_gradients = tape.gradient(
            g_loss,
            generator.trainable_variables
        )


        generator_optimizer.apply_gradients(
            zip(
                generator_gradients,
                generator.trainable_variables
            )
        )


        epoch_g_loss.append(
            float(g_loss)
        )

        epoch_d_loss.append(
            float(d_loss)
        )


    avg_g_loss = np.mean(epoch_g_loss)
    avg_d_loss = np.mean(epoch_d_loss)


    generator_losses.append(avg_g_loss)
    discriminator_losses.append(avg_d_loss)


    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"| Generator Loss: {avg_g_loss:.6f} "
        f"| Discriminator Loss: {avg_d_loss:.6f}"
    )


# =====================================================
# SAVE MODELS
# =====================================================

os.makedirs(
    "../models",
    exist_ok=True
)

generator.save(
    "../models/trajectory_generator.keras"
)

discriminator.save(
    "../models/trajectory_discriminator.keras"
)


# =====================================================
# SAVE LOSS GRAPH
# =====================================================

os.makedirs(
    "../results/graphs",
    exist_ok=True
)


plt.figure(figsize=(10, 6))

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

plt.title(
    "LSTM-GAN Training Loss"
)

plt.legend()
plt.grid(True)

plt.savefig(
    "../results/graphs/gan_training_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# =====================================================
# GENERATE SAMPLE TRAJECTORY
# =====================================================

sample_input = X_test[:1]

sample_noise = np.random.normal(
    0,
    1,
    size=(1, LATENT_DIM)
).astype(np.float32)


generated = generator.predict(
    [sample_input, sample_noise],
    verbose=0
)


generated = generated[0]

actual = Y_test[0]


# =====================================================
# VISUALIZATION
# =====================================================

plt.figure(figsize=(10, 7))


plt.plot(
    sample_input[0, :, 0],
    sample_input[0, :, 1],
    marker="o",
    label="Historical"
)


plt.plot(
    actual[:, 0],
    actual[:, 1],
    marker="o",
    label="Actual Future"
)


plt.plot(
    generated[:, 0],
    generated[:, 1],
    marker="x",
    linestyle="--",
    label="GAN Generated"
)


plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.title(
    "GAN Generated Vehicle Trajectory"
)

plt.legend()
plt.grid(True)


plt.savefig(
    "../results/graphs/gan_generated_trajectory.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


print("\n========================================")
print(" GAN TRAINING COMPLETED")
print("========================================")

print("\nModels saved:")

print(
    "../models/trajectory_generator.keras"
)

print(
    "../models/trajectory_discriminator.keras"
)

print("\nGraphs saved:")

print(
    "../results/graphs/gan_training_loss.png"
)

print(
    "../results/graphs/gan_generated_trajectory.png"
)