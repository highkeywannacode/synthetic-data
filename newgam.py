import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse

# Define the GAN models (same as before)
def make_generator_model(latent_dim, output_dim):
    model = keras.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(256, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(output_dim, activation='tanh')
    ])
    return model

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

def generate_synthetic_data(num_samples, epochs, seed_file, has_header):
    """
    Trains a GAN and generates synthetic data.

    Args:
        num_samples (int): The number of synthetic data rows to generate.
        epochs (int): The number of training epochs for the GAN.
        seed_file (str): The filename of the seed dataset.
        has_header (bool): True if the CSV file has a header row.
    """
    try:
        if has_header:
            data = pd.read_csv(seed_file)
        else:
            data = pd.read_csv(seed_file, header=None)
            data.columns = [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
                'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
            ]
        print(f"Dataset loaded successfully from '{seed_file}'.")
    except FileNotFoundError:
        print(f"Error: The file '{seed_file}' was not found.")
        return

    # Check for non-numeric columns and convert them if possible.
    for col in data.columns:
        if data[col].dtype == 'object':
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except ValueError:
                pass

    output_dim = data.shape[1]
    numerical_cols = data.columns[:-1]
    
    # Preprocessing
    min_values = data[numerical_cols].min()
    max_values = data[numerical_cols].max()

    data_scaled = data.copy()
    for col in numerical_cols:
        data_scaled[col] = 2 * (data_scaled[col] - min_values[col]) / (max_values[col] - min_values[col]) - 1

    target_col = data.columns[-1]
    data_scaled[target_col] = 2 * (data_scaled[target_col] - data[target_col].min()) / (data[target_col].max() - data[target_col].min()) - 1

    data_array = data_scaled.values.astype(np.float32)

    # ... The rest of the GAN logic is the same ...
    latent_dim = 100
    batch_size = 64
    
    generator = make_generator_model(latent_dim, output_dim)
    discriminator = make_discriminator_model(output_dim)
    cross_entropy = keras.losses.BinaryCrossentropy(from_logits=False)
    generator_optimizer = keras.optimizers.Adam(1e-4)
    discriminator_optimizer = keras.optimizers.Adam(1e-4)

    @tf.function
    def train_step(real_data):
        noise = tf.random.normal([batch_size, latent_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = generator(noise, training=True)
            real_output = discriminator(real_data, training=True)
            fake_output = discriminator(generated_data, training=True)
            total_disc_loss = cross_entropy(tf.ones_like(real_output), real_output) + \
                              cross_entropy(tf.zeros_like(fake_output), fake_output)
            gen_loss = cross_entropy(tf.ones_like(fake_output), fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(total_disc_loss, discriminator.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    print(f"Starting GAN training for {epochs} epochs...")
    for epoch in range(epochs):
        np.random.shuffle(data_array)
        for i in range(len(data_array) // batch_size):
            real_data_batch = data_array[i * batch_size:(i + 1) * batch_size]
            train_step(real_data_batch)
        if epoch % 500 == 0:
            print(f"Epoch {epoch}/{epochs}")
    print("Training complete.")

    print(f"\nGenerating {num_samples} synthetic data points...")
    noise = tf.random.normal([num_samples, latent_dim])
    generated_data = generator(noise, training=False).numpy()

    generated_df = pd.DataFrame(generated_data, columns=data_scaled.columns)
    for col in numerical_cols:
        generated_df[col] = (generated_df[col] + 1) / 2 * (max_values[col] - min_values[col]) + min_values[col]

    generated_df[target_col] = np.round((generated_df[target_col] + 1) / 2)

    output_file = 'synthetic_output.csv'
    generated_df.to_csv(output_file, index=False)
    print(f"\nSynthetic data saved to '{output_file}'.")
    print("\nSynthetic Data Head:")
    print(generated_df.head())

# --- New CLI section with the 'input' prompt ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic medical data using a GAN.')
    parser.add_argument('--epochs', type=int, default=5000,
                        help='Number of training epochs for the GAN.')
    parser.add_argument('--seed_file', type=str, required=True,
                        help='Path to the seed dataset CSV file.')
    parser.add_argument('--has_header', type=bool, default=True,
                        help='True if the CSV file has a header row.')

    args = parser.parse_args()

    # Get the number of samples from the user interactively
    while True:
        try:
            num_samples = int(input("Enter the number of synthetic data rows to generate: "))
            break
        except ValueError:
            print("Invalid input. Please enter a whole number.")

    generate_synthetic_data(num_samples, args.epochs, args.seed_file, args.has_header)