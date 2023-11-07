#Emisiones_3_new
#Autor Eric Téllez
#Equipo 04 Alianza-Huawei-DGTIC

#Emisiones para 2 categorías Emisiones-SinEmisiones
#usando la función image_dataset_from_directory
#y sus funciones consecuentes.

#Este código ya contiene el segmento de código de aprendizaje lento
#y también contiene la parte de código para guardar los datos de salida
#y para guardar el modelo entrenado y exportarlo para usarlo
#En este código

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import os
os.listdir()

#################################################################
#Mejor fllujo de trabajo serìa revisar estas:
#https://www.tensorflow.org/tutorials/load_data/images
#https://www.tensorflow.org/tutorials/images/data_augmentation
#https://www.tensorflow.org/guide/keras/preprocessing_layers

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from keras.applications.mobilenet_v2 import preprocess_input #Para usar la función de preprocesamiento de la siguiente función

#https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory
train_data=tf.keras.utils.image_dataset_from_directory(
    "/2022_04/Popo_Imag/dataset1",
    labels='inferred',
    label_mode='categorical', #binary cuando sean solo dos y categorical cuando sean tres
    class_names=("Sin_emisiones", "Gases", "Ceniza"),
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="training",
    interpolation='area', #bilinear, nearest, bicubic, area, lanczos3, lanczos5, gaussian, mitchellcubic. Area es ele mejor y bilinear el peor, las demás funcionan casi como area  https://en.wikipedia.org/wiki/Comparison_gallery_of_image_scaling_algorithms
    follow_links=False,
    crop_to_aspect_ratio=False,
)

val_data=tf.keras.utils.image_dataset_from_directory(
    "/2022_04/Popo_Imag/dataset1",
    labels='inferred',
    label_mode='categorical', #binary cuando sean solo dos y categorical cuando sean tres
    class_names=("Sin_emisiones", "Gases","Ceniza"),
    color_mode='rgb',
    batch_size=32,
    image_size=(224, 224),
    shuffle=True,
    seed=123,
    validation_split=0.2,
    subset="validation",
    interpolation='area', #bilinear, nearest, bicubic, area, lanczos3, lanczos5, gaussian, mitchellcubic  https://en.wikipedia.org/wiki/Comparison_gallery_of_image_scaling_algorithms
    follow_links=False,
    crop_to_aspect_ratio=False,
)

AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_data = val_data.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_data = train_data.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_data))
first_image = image_batch[0]

#url = "https://tfhub.dev/google/edgetpu/vision/mobilenet-edgetpu-v2/tiny/1"
url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
mobilenetv2 = hub.KerasLayer(url, input_shape=(224,224,3)) #Los dos primeros números son los pixeles de la imagen y el tercer número es la matriz de color RGB

#https://tfhub.dev/

#Congelar el modelo descargado False, entrenar el modelo True
mobilenetv2.trainable = False


#https://keras.io/api/models/sequential/
modelo = tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.Dense(3, activation='sigmoid') #Número Dense es 1 cuando son dos categorías y es igual al número de categorías cuando son más. Esto es para el algoritmo nuevo.
])

"""
#Para hacer el modelo desde cero
num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])

"""
base_learning_rate = 0.001
#Compilar
modelo.compile(
    optimizer='adamax',  #Los mejores optimizadores son adam, adamax, adagrad y ftrl
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
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
EPOCAS = 1000  # Este número puede aumentar para hacerlo más preciso. Dependerá del tiempo de servidor y de la calidad de la predicción

historial = modelo.fit(
    train_data,
    #y=None,
    epochs=EPOCAS,
    batch_size=32, #número 32 es  el número de lotes cada que se actualiza el gradiente, default 32
    #verbose="auto",
    #callbacks=None,
    validation_split=0.2,
    validation_data=val_data,
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
    #use_multiprocessing=True
)

#https://keras.io/api/models/model_training_apis/

##########################
# Ajuste fino
#Congelar el modelo descargado False, entrenar el modelo True
mobilenetv2.trainable = True

modelo = tf.keras.Sequential([
    mobilenetv2,
    tf.keras.layers.Dense(3, activation='sigmoid') #Número Dense tiene que ser el mismo que el número de categorías
])

#Compilar
modelo.compile(
    optimizer=tf.keras.optimizers.Adamax(learning_rate=base_learning_rate/100),    
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
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
EPOCAS = 1000  # Este número puede aumentar para hacerlo más preciso. Dependerá del tiempo de servidor y de la calidad de la predicción

historial = modelo.fit(
    train_data, 
    epochs=EPOCAS, 
    batch_size=32, #número 32 es  el número de lotes cada que se actualiza el gradiente, default 32
    #verbose="auto",
    #callbacks=None,
    validation_split=0.2,
    validation_data=val_data,
    #shuffle=True,
    #class_weight=None,
    #sample_weight=None,
    initial_epoch=1000,
    #steps_per_epoch=None,
    #validation_steps=None,
    #validation_batch_size=None,
    #validation_freq=1,
    #max_queue_size=10,
    #workers=1,
    #use_multiprocessing=False
)



"""
from keras.callbacks import CSVLogger

csv_logger = CSVLogger('log.csv', append=True, separator=';')
model.fit(X_train, Y_train, callbacks=[csv_logger])


history_callback = model.fit(params...)
loss_history = history_callback.history["loss"]

#Categorizar una imagen de internet
from PIL import Image
import requests
from io import BytesIO
import cv2

def categorizar(url):
  respuesta = requests.get(url)
  img = Image.open(BytesIO(respuesta.content))
  img = np.array(img).astype(float)/255

  img = cv2.resize(img, (224,224))
  prediccion = modelo.predict(img.reshape(-1, 224, 224, 3))
  return np.argmax(prediccion[0], axis=-1)


#0 = sin emisión, 1 = ceniza, 2 = gases  #¿Cómo obtuvo el orden de las etiquetas?

#Aquí tengo que poner imágenes de cenapred para probar si el modelo funciona
url = 'https://www.cenapred.unam.mx/popo/2020/jun/p0601201.jpeg' #Este es solo un ejemplo y hay que pensarlo bien
prediccion = categorizar (url)
print(prediccion)

#Aquí va un condicional para que se muestre la etiqueta en lugar del número
#if prediccion == 0:
#  print("Sin emisión")
#else prediccion == 1:
#  print("Emisión")

#Guardar el modelo en formato SavedModel
#modelo.save('carpeta_salida/modelo_emisiones_1/1')
#Hacerlo un zip para bajarlo y usarlo en otro lado
#!zip -r modelo_cocina.zip /content/carpeta_salida/modelo_emisiones_1/
"""
