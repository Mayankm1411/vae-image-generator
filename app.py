import tensorflow as tf
import numpy as np
import gradio as gr

model = tf.keras.models.load_model("gan_generator.keras")

def generate_face():
    z = tf.random.normal((1, 100))
    img = model(z)[0].numpy()
    img = (img + 1) / 2.0
    return img

gr.Interface(fn=generate_face, inputs=[], outputs="image", title="GAN Face Generator").launch()
