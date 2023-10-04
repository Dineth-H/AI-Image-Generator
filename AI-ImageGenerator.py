import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog

# Create the GAN 
# Generator model
generator = Sequential([
    Dense(128, input_shape=(100,), activation='relu'),
    Dense(784, activation='sigmoid'),
    Reshape((28, 28, 1))
])

# Discriminator model
discriminator = Sequential([
    Flatten(input_shape=(28, 28, 1)),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the discriminator
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# GAN model
discriminator.trainable = False  # Freeze discriminator during GAN training
gan = Sequential([generator, discriminator])
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Load and preprocess your dataset, e.g., MNIST
# Here, we'll use a simple example with random noise
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0  # Normalize pixel values to [0, 1]

# Training loop
batch_size = 64
epochs = 30000

for epoch in range(epochs):
    noise = np.random.normal(0, 1, (batch_size, 100))
    generated_images = generator.predict(noise)
    real_images = x_train[np.random.randint(0, x_train.shape[0], batch_size)]

    # Create labels for real and generated images
    real_labels = np.ones((batch_size, 1))
    generated_labels = np.zeros((batch_size, 1))

    # Train the discriminator
    d_loss_real = discriminator.train_on_batch(real_images, real_labels)
    d_loss_generated = discriminator.train_on_batch(generated_images, generated_labels)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_generated)

    # Train the generator (via the GAN model)
    noise = np.random.normal(0, 1, (batch_size, 100))
    valid_labels = np.ones((batch_size, 1))
    g_loss = gan.train_on_batch(noise, valid_labels)

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# Create a function to generate and display images
def generate_and_display_image():
    noise = np.random.normal(0, 1, (1, 100))
    generated_image = generator.predict(noise)[0]
    generated_image = (generated_image * 255).astype(np.uint8)  # Scale to [0, 255]
    generated_image = Image.fromarray(generated_image.reshape(28, 28))
    generated_image = ImageTk.PhotoImage(generated_image)

    # Display the generated image
    generated_label.config(image=generated_image)
    generated_label.image = generated_image

# Create a function to save the generated image
def save_generated_image():
    noise = np.random.normal(0, 1, (1, 100))
    generated_image = generator.predict(noise)[0]
    generated_image = (generated_image * 255).astype(np.uint8)  # Scale to [0, 255]
    generated_image = Image.fromarray(generated_image.reshape(28, 28))

    # Ask the user for a save location
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
    if save_path:
        generated_image.save(save_path)

# Create the main GUI window
root = Tk()
root.title("Image Generator")

# Create input label and entry for image prompts (not implemented for GAN)
prompt_label = Label(root, text="Enter Image Prompt:")
prompt_label.pack()
prompt_entry = Entry(root)
prompt_entry.pack()

# Create a label to display the generated image
generated_label = Label(root)
generated_label.pack()

# Create "Generate" and "Save" buttons
generate_button = Button(root, text="Generate", command=generate_and_display_image)
generate_button.pack()
save_button = Button(root, text="Save", command=save_generated_image)
save_button.pack()

# Create the credit label
credit_label = Label(root, text="Created by Dineth Hesara using Python, Keras & TensorFlow")
credit_label.pack()

# Start the GUI main loop
root.mainloop()
