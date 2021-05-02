#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  2 23:37:07 2021

@author: batuhan
"""
#Tenserflow ve Gerekli Kütüphanelerin eklenmesi
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Fashion MNIST veri kümesinin içe aktarımı 
"""
Bu kılavuz, 10 kategoride 70.000 gri tonlamalı görüntü içeren Fashion MNIST veri kümesini kullanır. 
Görüntüler, - düşük çözünürlükte (28 x 28 piksel) ayrı giyim eşyalarını göstermektedir:-
https://github.com/zalandoresearch/fashion-mnist
Burada, ağı eğitmek için 60.000 görüntü ve ağın görüntüleri sınıflandırmayı ne kadar doğru 
öğrendiğini değerlendirmek için 10.000 görüntü kullanılıyor. 
Fashion MNIST'e doğrudan TensorFlow'dan erişebilirsiniz. 
Fashion MNIST verilerini doğrudan TensorFlow'dan içe aktarılır ve yüklenir
"""
fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

"""
Veri kümesini yüklemek dört NumPy dizisi döndürür:

train_images ve train_labels dizileri eğitim setidir — modelin öğrenmek için kullandığı veriler.
Model, test seti , test_images ve test_labels dizileri ile test edilir.
Görüntüler, piksel değerleri 0 ile 255 arasında değişen 28x28 NumPy dizileridir. 
Etiketler , 0 ile 9 arasında değişen bir tamsayı dizisidir. 
Bunlar, görüntünün temsil ettiği giysi sınıfına karşılık gelir.
Her görüntü tek bir etiketle eşleştirilir. 
Sınıf adları veri kümesine dahil edilmediğinden, daha sonra görüntüleri çizerken kullanmak için 
burada saklanır:
"""

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#Modeli eğitmeden önce veri setinin formatını icelemek isterseniz
#Aşağıda, eğitim setinde her bir görüntünün 28 x 28 piksel olarak temsil edildiği 
#60.000 görüntü olduğu gösterilmektedir. 
train_images.shape
#(60000, 28, 28)


#Aynı şekilde eğitim setinde de 60.000 etiket vardır
len(train_labels)
#60000

#Her etiket, 0 ile 9 arasında bir tam sayıdır
train_labels
#array([9, 0, 0, ..., 3, 0, 5], dtype=uint8)

#Test setinde 10.000 görüntü var. Yine, her görüntü 28 x 28 piksel olarak temsil edilir.
test_images.shape
#(10000, 28, 28)

#Ve test seti 10.000 resim etiketi içerir
len(test_labels)
#10000

#Ağı eğitmeden önce veriler önceden işlenmelidir. Eğitim setindeki ilk resmi incelerseniz,
#piksel değerlerinin 0 ile 255 aralığında olduğunu göreceksiniz
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#RESİM1 ÇIKTISI KOD SONUNDA VERİLMİŞTİR


"""Bu değerleri sinir ağı modeline beslemeden önce 0 ila 1 aralığında ölçeklendirin. 
Bunu yapmak için değerleri 255'e bölün. Eğitim seti ve test setinin aynı şekilde 
ön işlemden geçirilmesi önemlidir.
"""
train_images = train_images / 255.0
test_images = test_images / 255.0
#Oluşturulan piksel tablosu kod sonunda verilmiştir.

#Verilerin doğru biçimde olduğunu ve ağı kurmaya ve eğitmeye hazır olduğunuzu doğrulamak için 
#eğitim setinden ilk 25 görüntüyü gösterelim ve her görüntünün altında sınıf adını gösterelim.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

#RESİM2 ÇIKTISI KOD SONUNDA VERİLMİŞTİR


#Bu kısımda model oluşturulmaktadır


#Modelin katmanlarının ayarlanması
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

#Modelin Derlenmesi
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Modelin eğitilmesi ve veri setiyle beslenmesi
model.fit(train_images, train_labels, epochs=10)

#Modelin test veri kümesinde nasıl performans gösterdiğinin karşılaştırılması
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

#Modelin tahminlerde bulunması
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

"""
Tahmin, 10 sayılık bir dizidir. Modelin görüntünün 10 farklı giyim eşyasının her birine 
karşılık geldiğine olan "güvenini" temsil ediyorlar. 
Hangi etiketin en yüksek güven değerine sahip olduğunu görebilirsiniz:

"""
predictions[0]

"""
array([5.1698703e-07, 5.0422708e-11, 1.0513627e-06, 4.2676376e-08,
       4.1753174e-07, 8.8213873e-04, 1.4294442e-06, 8.9591898e-02,
       3.7699414e-07, 9.0952224e-01], dtype=float32)
"""

np.argmax(predictions[0])
test_labels[0]

#10 farklı tahmininin tamamına bakmak için bunun grafiğini çizilir.

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
  
#Eğitilen model ile bazı görüntüler hakkında tahminlerde bulunmak için kullanabilirsiniz.

"""0. görüntüye, tahminlere ve tahmin dizisine bakalım. Doğru tahmin etiketleri mavidir ve 
yanlış tahmin etiketleri kırmızıdır. Sayı, tahmin edilen etiket için yüzdeyi (100 üzerinden) verir.
"""
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
#RESİM3 ÇIKTISI KOD SONUNDA VERİLMİŞTİR


i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()
#RESİM4 ÇIKTISI KOD SONUNDA VERİLMİŞTİR


#Tahminleriyle birkaç görüntüyü çizelim. Kendinden çok emin olsa bile
#modelin yanlış olabileceğini unutmayın.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
#RESİM5 ÇIKTISI KOD SONUNDA VERİLMİŞTİR


#Son olarak, tek bir görüntü hakkında bir tahmin yapmak için eğitimli modeli kullanın

img = test_images[1]

print(img.shape)

#tf.keras modelleri, bir kerede örneklerin bir toplu işi veya koleksiyonu 
#üzerinde tahminler yapmak için optimize edilmiştir.
#Buna göre, tek bir resim kullanıyor olsanız bile, onu bir listeye eklemeniz gerekir.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])



## MIT License
#
# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
































