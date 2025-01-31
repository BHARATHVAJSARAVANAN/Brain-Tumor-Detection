import os
import cv2
import warnings
import itertools
import pandas as pd
import numpy  as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras import layers
from tensorflow.keras import regularizers

from flask import Flask, request, render_template

#  Load data
path_no_tumor = 'No Tumor'
path_pituitary_tumor = 'Pituitary Tumor'
path_meningioma_tumor = 'Meningioma Tumor'
path_glioma_tumor = 'Glioma Tumor'
tumor_check = {'No Tumor': 0, 'Pituitary Tumor': 1, 'Meningioma Tumor': 2, 'Glioma Tumor': 3}

dataset = []
lab = []
for cls in tumor_check:
    if cls == 'No Tumor':
        path = path_no_tumor
    elif cls == 'Pituitary Tumor':
        path = path_pituitary_tumor
    elif cls == 'Meningioma Tumor':
        path = path_meningioma_tumor
    elif cls == 'Glioma Tumor':
        path = path_glioma_tumor
    for j in os.listdir(path):
        image = cv2.imread(path+'/'+j, 0)
        image = cv2.resize(image, (200, 200))
        dataset.append(image)
        lab.append(tumor_check[cls])

dataset = np.array(dataset)
lab = np.array(lab)

pd.Series(lab).value_counts()

# Prepare data
dataset_update = dataset.reshape(len(dataset), -1)
x_train, x_test, y_train, y_test = train_test_split(dataset_update, lab, random_state=10, test_size=0.3)

x_train = x_train / 255
x_test = x_test / 255

pca = PCA(.98)
pca_train = pca.fit_transform(x_train)
pca_test = pca.transform(x_test)

logistic = LogisticRegression(C=0.1)
logistic.fit(pca_train, y_train)

sv = SVC()
sv.fit(pca_train, y_train)

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 240
patch_size = 20 
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  
transformer_layers = 8
mlp_head_units = [2048, 1024] 

data_augmentation = tf.keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)

data_augmentation.layers[0].adapt(x_train)

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    
def create_vit_classifier():
    inputs = layers.Input(shape=(240, 240, 3))
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(2)(features)
    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "checkpoint"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(x_test, y_test),
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)

    return history

# Load the trained model
checkpoint_filepath = "checkpoints/brain_model.h5"

app = Flask(__name__)


@app.route("/")
def hello_world():
  return render_template('home.html')

# Execute function
@app.route("/execute_python_function", methods=["POST"])


def execute_python_function():
    file = request.files['image']
    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))
        img = img.reshape(1, -1) / 255
        p = sv.predict(pca.transform(img))
        dec = {0: 'No Tumor', 1: 'Pituitary Tumor', 2: 'Meningioma Tumor', 3: 'Glioma Tumor'}
        tumor_type = dec[p[0]]
        return tumor_type


if __name__ == '__main__':
  app.run(debug=True)
