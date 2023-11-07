#Emisiones_finetunning_2 clasificado en dos categorias

#Librería para cargar los archivos
import os
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg

os.listdir()

#El manual de TensorFlow dice que esta funciòn está obsoleta
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# Revisar todos los parámetros: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator?authuser=2


#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

from keras.applications.mobilenet_v2 import preprocess_input #Para usar la función de preprocesamiento de la siguiente función

# https://medium.com/analytics-vidhya/understanding-image-augmentation-using-keras-tensorflow-a6341669d9ca
#Crear el dataset generador
datagen = ImageDataGenerator(
    featurewise_center=True,
    samplewise_center=True,
    featurewise_std_normalization=True,
    samplewise_std_normalization=True,
    zca_whitening=True,  #https://stats.stackexchange.com/questions/117427/what-is-the-difference-between-zca-whitening-and-pca-whitening https://towardsdatascience.com/only-numpy-back-propagating-through-zca-whitening-in-numpy-manual-back-propagation-21eaa4a255fb
    zca_epsilon=1e-01,  #default 1e-06
    brightness_range=[0,2],
    channel_shift_range=100,
    fill_mode='wrap',  #https://fairyonice.github.io/Learn-about-ImageDataGenerator.html
    #cval=0.0,
    horizontal_flip=True,
    vertical_flip=True,
    rescale=1./255,
    rotation_range = 30,
    width_shift_range = 0.25,
    height_shift_range = 0.25,
    shear_range = 15,
    zoom_range = [0.5, 1.5],
    validation_split=0.2, # 20% para pruebas es lo normal en casi todos los modelos, pero se puede alterar
    preprocessing_function=preprocess_input,
    data_format=None,
    interpolation_order=1,
    dtype=None
)

#Es mejor usar esta otra función (sí la estamos usando en Emisiones_2_old
#https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory

#Generadores para sets de entrenamiento y pruebas
# https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
#Aquí en flow_from_directory se sabe cuántas clases existen. Quizàs para no duplicar los datos sea mejor hacer otra configuración de las carpetas
data_gen_entrenamiento = datagen.flow_from_directory('/2022_04/Popo_Imag/dataset', 
                                                     target_size=(224,224), #para recortar las imágenes a 224x224
                                                     batch_size=32, 
                                                     shuffle=True, 
                                                     subset='training')
data_gen_pruebas = datagen.flow_from_directory('/2022_04/Popo_Imag/dataset', 
                                               target_size=(224,224),
                                               batch_size=32, 
                                               shuffle=True, 
                                               subset='validation')

import tensorflow as tf
import tensorflow_hub as hub

#url = "https://tfhub.dev/google/edgetpu/vision/mobilenet-edgetpu-v2/tiny/1"
url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobilenetv2 = hub.KerasLayer(url, input_shape=(224,224,3)) #Los dos primeros números son los pixeles de la imagen y el tercer número es la matriz de color RGB

#https://tfhub.dev/


#Congelar el modelo descargado False, entrenar el modelo True
mobilenetv2.trainable = False

#global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
#feature_batch_average = global_average_layer(feature_batch)
#print(feature_batch_average.shape)
"""
inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = mobilnetv2(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.summary()

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adamax(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])
"""


modelo = tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.Dense(2, activation='softmax') #Número Dense tiene que ser el mismo que el número de categorías
])

#https://keras.io/api/models/sequential/

base_learning_rate = 0.0001
#Compilar
modelo.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=base_learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy']
    #loss_weights=None,
    #weighted_metrics=None,
    #run_eagerly=None,
    #steps_per_execution=None,
    #jit_compile=None,
    #pss_evaluation_shards=0,
    #**kwargs
)

#https://keras.io/api/models/model_training_apis/

#Entrenar el modelo
EPOCAS = 100  # Este número puede aumentar para hacerlo más preciso. Dependerá del tiempo de servidor y de la calidad de la predicción

historial = modelo.fit(
    data_gen_entrenamiento, 
    y=None,
    epochs=EPOCAS, 
    batch_size=32, #número 32 es  el número de lotes cada que se actualiza el gradiente, default 32
    #verbose="auto",
    #callbacks=None,
    #validation_split=0.0,
    validation_data=data_gen_pruebas,
    #shuffle=True,
    #class_weight=None,
    #sample_weight=None,
    initial_epoch=0,
    #steps_per_epoch=None,
    #validation_steps=None,
    #validation_batch_size=None,
    #validation_freq=1,
    #max_queue_size=10,
    #workers=1,
    #use_multiprocessing=False
)

#https://keras.io/api/models/model_training_apis/

#Congelar el modelo descargado False, entrenar el modelo True
mobilenetv2.trainable = True

modelo = tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.Dense(2, activation='softmax') #Número Dense tiene que ser el mismo que el número de categorías
])

#Compilar
modelo.compile(
    optimizer=tf.keras.optimizers.Adagrad(learning_rate=base_learning_rate/100),
    loss='binary_crossentropy',
    metrics=['accuracy']
    #loss_weights=None,
    #weighted_metrics=None,
    #run_eagerly=None,
    #steps_per_execution=None,
    #jit_compile=None,
    #pss_evaluation_shards=0,
    #**kwargs
)

#Entrenar el modelo
EPOCAS = 2000  # Este número puede aumentar para hacerlo más preciso. Dependerá del tiempo de servidor y de la calidad de la predicción

historial = modelo.fit(
    data_gen_entrenamiento, 
    y=None,
    epochs=EPOCAS, 
    batch_size=32, #número 32 es  el número de lotes cada que se actualiza el gradiente, default 32
    #verbose="auto",
    #callbacks=None,
    #validation_split=0.0,
    validation_data=data_gen_pruebas,
    #shuffle=True,
    #class_weight=None,
    #sample_weight=None,
    initial_epoch=101,
    #steps_per_epoch=None,
    #validation_steps=None,
    #validation_batch_size=None,
    #validation_freq=1,
    #max_queue_size=10,
    #workers=1,
    #use_multiprocessing=False
)

"""
fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)
"""
