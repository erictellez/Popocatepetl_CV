# Popocatepetl_CV
Software to process Popocatepetl video throught computational vision.
This software was made with the sponsorship of Alianza-Huawei-DGTIC.
All the images and video of Popocatepetl volcano is taken from official pages and youtube channel of CENAPRED and Webcamsdemexico.  

All the software is written in TensorFlow.

There are several codes. 

-- Emisiones_1.ipynb
   This notebook is the first we wrote to make some tests in the general database. 

-- Emisiones_2_old.py
   This code is to classify images in 2 different types. One in the volcano not active. The other is when the volcano is active, emissions are visible.
  This code is made with the function ImageDataGenerator that is slow and is not recommended anymore by TensorFlow.

  -- Emisiones_3_old.py
This code is to classify images in 3 different types. One in the volcano not active, the other is when gas emissions are visible and the third is when the ash emissions are visible.
  This code is made with the function ImageDataGenerator that is slow and is not recommended anymore by TensorFlow. This code is not working really well, the validation percentage is low, about 60%.
   
-- Emisiones_2_new.py
  This code is also to classify images in 2 differente types.
  This code has a different function that is faster and produce higher percentage in the validation outcome than the _old file. This code is really accurate and is the one we are going to try to make into production in real time.

  -- Emisiones_3_new.py
  This code is to classify images in 3 differente types: no_emissions, gas, ashes.
  This code has a different function that is faster and produce higher percentage in the validation outcome than the _old file. Because this classification is very difficult, even for a person, the validation is about 70%. We are trying to impreove this code with several techniques, since only the ashes from the volcano are dangerous for people, cattle and planes.

  
All these codes produce some trained model that is able to classify the images and the result is a string with the text "Emisión", or "No emisión". This is the simplest model we get.

There are some other files for the object recognition.
-- 


An finally, there are the files of inference.

-- 
