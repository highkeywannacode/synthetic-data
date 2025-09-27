import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Load the dataset
file_name = "pima-indians-diabetes.csv"
try:
    data = pd.read_csv(file_name)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found. Please ensure it is in the same directory.")
    exit()

# The Pima Indians dataset does not have headers. We will assign them manually
data.columns = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age',
    'Outcome'
]
print("\nColumn names assigned.")

# Inspect the data
print("\nData Info:")
data.info()
print("\nData Head:")
print(data.head())

# Preprocessing: Scale numerical columns to a [-1, 1] range
# This is a critical step for GANs using tanh activation in the final layer.
numerical_cols = data.columns[:-1]
min_values = data[numerical_cols].min()
max_values = data[numerical_cols].max()

data_scaled = data.copy()
for col in numerical_cols:
    data_scaled[col] = 2 * (data_scaled[col] - min_values[col]) / (max_values[col] - min_values[col]) - 1

# Convert the DataFrame to a NumPy array for the GAN
data_array = data_scaled.values.astype(np.float32)

# Define GAN parameters
latent_dim = 100
output_dim = data_array.shape[1]
batch_size = 64
epochs = 5000

# Define the Generator model
def make_generator_model(latent_dim, output_dim):
    model = keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(output_dim, activation='tanh')
    ])
    return model

# Define the Discriminator model
def make_discriminator_model(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Create the models and optimizers
generator = make_generator_model(latent_dim, output_dim)
discriminator = make_discriminator_model(output_dim)

cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
generator_optimizer = keras.optimizers.Adam(1e-4)
discriminator_optimizer = keras.optimizers.Adam(1e-4)

# Training step function
@tf.function
def train_step(real_data):
    noise = tf.random.normal([batch_size, latent_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_data = generator(noise, training=True)

        real_output = discriminator(real_data, training=True)
        fake_output = discriminator(generated_data, training=True)

        # Discriminator loss: make sure it can distinguish real from fake
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_disc_loss = real_loss + fake_loss

        # Generator loss: make sure it can fool the discriminator
        gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

    # Apply gradients to update weights
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

# Full training loop
print("\nStarting GAN training...")
for epoch in range(epochs):
    np.random.shuffle(data_array)
    for i in range(len(data_array) // batch_size):
        real_data_batch = data_array[i*batch_size:(i+1)*batch_size]
        train_step(real_data_batch)

    if epoch % 500 == 0:
        print(f"Epoch {epoch}/{epochs}")

print("Training complete.")

# Generate a final set of synthetic data
print("\nGenerating synthetic data...")
num_to_generate = 500
noise = tf.random.normal([num_to_generate, latent_dim])
generated_data = generator(noise, training=False).numpy()

# Post-processing: Reverse the scaling
generated_df = pd.DataFrame(generated_data, columns=data_scaled.columns)
for col in numerical_cols:
    generated_df[col] = (generated_df[col] + 1) / 2 * (max_values[col] - min_values[col]) + min_values[col]

# Convert the 'Outcome' column back to binary (0 or 1)
# The output is between -1 and 1, which we convert to a 0-1 range and then round.
generated_df['Outcome'] = np.round((generated_df['Outcome'] + 1) / 2)

# Save the synthetic data to a CSV file
output_file = 'synthetic_pima_diabetes_data.csv'
generated_df.to_csv(output_file, index=False)
print(f"\nSynthetic data saved to '{output_file}'.")

# Display the head of the generated data to verify the results
print("\nSynthetic Data Head:")
print(generated_df.head())