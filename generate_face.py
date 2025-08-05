import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

generator = tf.keras.models.load_model("gan_generator.keras")

z = tf.random.normal((1, 100))
generated_image = generator(z)[0].numpy()
generated_image = (generated_image + 1) / 2.0  # Rescale to [0, 1]

plt.imshow(generated_image)
plt.axis('off')
plt.show()
