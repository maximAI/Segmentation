# Segmentation
Задача: На собственной базе изображений решить задачу сегментации. Обучить модель сегментировать изображение на заданные классы.

<a name="4"></a>
## [Оглавление:](#4)
1. [Подготовка данных](#1)
2. [Линейная модель](#2)
3. [U-net](#3)

Импортируем нужные библиотеки.
```
from tensorflow.keras.models import Model                       # Загружаем абстрактный класс базовой модели сети от кераса
# Подключим необходимые слои
from tensorflow.keras.layers import Input, Conv2DTranspose, concatenate, Conv2D, \
                                    Activation, MaxPooling2D, BatchNormalization
from tensorflow.keras import backend as K                       # Подключим бэкэнд Керас
from tensorflow.keras.optimizers import Adam                    # Подключим оптимизатор
from tensorflow.keras import utils                              # Подключим utils
from tensorflow.keras.utils import plot_model                   # Подключим plot_model для отрисовки модели
from google.colab import files                                  # Загрузка данных
import matplotlib.pyplot as plt                                 # Подключим библиотеку для визуализации данных
import numpy as np                                              # Подключим numpy - библиотеку для работы с массивами данных
from tensorflow.keras.preprocessing import image                # Подключим image для работы с изображениями
from sklearn.model_selection import train_test_split            # Импортируем train_test_split для деления выборки
import time                                                     # Импортируем модуль time
import random                                                   # Импортируем библиотеку random
import os                                                       # Импортируем модуль os для загрузки данных
from PIL import Image                                           # Подключим Image для работы с изображениями
```
Объявим необходимые функции.
```
def color2index(color):
    '''
    Функция преобразования пикселя сегментированного изображения в индекс (7 классов)
    '''
    index=-1

    if   (229>=color[0]>199)  and(49>=color[1]>=0)    and(99>=color[2]>=49)   : index=0  # человек
    elif (99>=color[0]>=49)   and(99>=color[1]>=49)   and(99>=color[2]>=49)   : index=1  # дом
    elif (49>=color[0]>=0)    and(49>=color[1]>=0)    and(99>=color[2]>=49)   : index=2  # автомобили
    elif (255>=color[0]>=199) and(129>=color[1]>=79)  and(49>=color[2]>=0)    : index=2  # грузовики
    elif (49>=color[0]>=0)    and(99>=color[1]>=51)   and(129>=color[2]>=79)  : index=2  # автобус
    elif (149>=color[0]>=99)  and(49>=color[1]>=0)    and(49>=color[2]>=0)    : index=2  # велосипед
    elif (149>=color[0]>=99)  and(199>=color[1]>=99)  and(49>=color[2]>=0)    : index=3  # деревья
    elif (99>=color[0]>=49)   and(149>=color[1]>=99)  and(199>=color[2]>=151) : index=4  # небо
    elif (199>=color[0]>=139) and(199>=color[1]>=139) and(199>=color[2]>=139) : index=5  # столбы
    elif (129>=color[0]>=99)  and(99>=color[1]>=49)   and(49>=color[2]>=0)    : index=5  # контейнер
    elif (149>=color[0]>=99)  and(79>=color[1]>=49)   and(149>=color[2]>=99)  : index=7  # пол
    elif (129>=color[0]>=99)  and(129>=color[1]>=99)  and(199>=color[2]>=120) : index=5  # стены на улице
    elif (255>=color[0]>=199) and(199>=color[1]>=149) and(49>=color[2]>=0)    : index=6  # светофор
    elif (49>=color[0]>=0)    and(49>=color[1]>=0)    and(49>=color[2]>=0)    : index=6  # рекл.вывеска 
    elif (255>=color[0]>=199) and(49>=color[1]>=0)    and(255>=color[2]>=199) : index=7  # тротуар
    elif (199>=color[0]>=139) and(255>=color[1]>=199) and(199>=color[2]>=139) : index=8  # газон, трава
    elif (99>=color[0]>=49)   and(149>=color[1]>=99)  and(199>=color[2]>=139) : index=9  # река, озеро
    elif (199>=color[0]>=139) and(129>=color[1]>=49)  and(129>=color[2]>=49)  : index=5  # мост
    elif (249>=color[0]>=199) and(249>=color[1]>=199) and(49>=color[2]>=0)    : index=6  # дорожные знаки
    
    else: index=10
    
    return index 
```
```
def index2color(index2):
    '''
    Функция преобразования индекса в цвет пикселя
    '''
    index = np.argmax(index2) # Получаем индекс максимального элемента
    color=[]
    if   index == 0: color =  [255, 0, 0]      # человек (красный)
    elif index == 1: color =  [255, 150, 0]    # дом (коричневый)
    elif index == 2: color =  [0, 0, 250]      # автомобили, грузовики, автобус, велосипед (синий)
    elif index == 3: color =  [60, 110, 55]    # деревья (темно-зеленый)
    elif index == 4: color =  [200, 240, 240]  # небо (светло-голубой)
    elif index == 5: color =  [10, 15, 60]     # столбы, контейнер, стены на улице, мост (темно-синий)
    elif index == 6: color =  [50, 50, 10]     # контейнер, рекл.вывеска, дорожные знаки (темно-желтый)
    elif index == 7: color =  [160, 160, 160]  # тротуар (серый)
    elif index == 8: color =  [250, 150, 150]  # газон, трава (розовый)
    elif index == 9: color =  [255, 255, 0]    # река, озеро (желтый)
    elif index == 10: color = [0, 0, 0]        # остальное (черный)

    return color # Возвращаем цвет пикслея
```
```
def rgbToohe(y, num_classes):
    '''
    Функция перевода индекса пикслея в to_categorical
    '''
    y2 = y.copy()                               # Создаем копию входного массива
    y = y.reshape(y.shape[0] * y.shape[1], 3)   # Решейпим в двумерный массив
    yt = []                                     # Создаем пустой лист
    for i in range(len(y)):                     # Проходим по всем трем канала изображения
        yt.append(utils.to_categorical(color2index(y[i]), num_classes=num_classes)) # Переводим пиксели в индексы и преобразуем в OHE
    yt = np.array(yt)                           # Преобразуем в numpy
    yt = yt.reshape(y2.shape[0], y2.shape[1], num_classes) # Решейпим к исходныму размеру
    return yt                                   # Возвращаем сформированный массив
```
```
def yt_prep(data, num_classes):
    '''
    Функция формирования yTrain
    '''
    yTrain = []                         # Создаем пустой список под карты сегметации
    for seg in data:                    # Пробегаем по всем файлам набора с сегминтированными изображениями
        y = image.img_to_array(seg)     # Переводим изображение в numpy-массив размерностью: высота - ширина - количество каналов
        y = rgbToohe(y, num_classes)    # Получаем OHE-представление сформированного массива
        yTrain.append(y)                # Добавляем очередной элемент в yTrain
        if len(yTrain) % 100 == 0:      # Каждые 100 шагов
        print(len(yTrain))              # Выводим количество обработанных изображений
    return np.array(yTrain)             # Возвращаем сформированный yTrain
```
```
def processImage(model, count = 1, n_classes = 11):
    '''
    Функция визуализации сегментированных изображений
    '''
    indexes = np.random.randint(0, len(xVal), count)    # Получаем count случайных индексов
    fig, axs = plt.subplots(3, count, figsize=(25, 5))  # Создаем полотно из n графиков
    for i,idx in enumerate(indexes):                    # Проходим по всем сгенерированным индексам
        predict = np.array(model.predict(xVal[idx].reshape(1, img_width, img_height, 3))) # Предиктим картику
        pr = predict[0]                                 # Берем нулевой элемент из перидкта
        pr1 = []                                        # Пустой лист под сегментированную картинку из predicta
        pr2 = []                                        # Пустой лист под сегменитрованную картинку из yVal
        pr = pr.reshape(-1, n_classes)                  # Решейпим предикт
        yr = yVal[idx].reshape(-1, n_classes)           # Решейпим yVal
        for k in range(len(pr)):                        # Проходим по всем уровням (количесвто классов)
        pr1.append(index2color(pr[k]))                  # Переводим индекс в писксель
        pr2.append(index2color(yr[k]))                  # Переводим индекс в писксель
        pr1 = np.array(pr1)                             # Преобразуем в numpy
        pr1 = pr1.reshape(img_width, img_height,3)      # Решейпим к размеру изображения
        pr2 = np.array(pr2)                             # Преобразуем в numpy
        pr2 = pr2.reshape(img_width, img_height,3)      # Решейпим к размеру изображения
        img = Image.fromarray(pr1.astype('uint8'))      # Получаем картику из предикта
        axs[0,i].imshow(img.convert('RGBA'))            # Отображаем на графике в первой линии
        axs[1,i].imshow(Image.fromarray(pr2.astype('uint8')))       # Отображаем на графике во второй линии сегментированное изображение из yVal
        axs[2,i].imshow(Image.fromarray(xVal[idx].astype('uint8'))) # Отображаем на графике в третьей линии оригинальное изображение        
    plt.show()  
```
```
def dice_coef(y_true, y_pred):
    '''
    Функция метрики, обрабатывающая пересечение двух областей
    '''
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.) # Возвращаем площадь пересечения деленную на площадь объединения двух областей
```
[:arrow_up:Оглавление](#4)
<a name="1"></a>
## Подготовка данных.
Глобальные параметры.
```
img_width = 144     # Ширина уменьшенной картинки 
img_height = 256    # Высота уменьшенной картинки 
num_classes = 11    # Задаем количество классов на изображении
directory = '/content/drive/MyDrive/GTA/'   # Указываем путь к обучающей выборке с оригинальными изображения
train_directory = 'Тренировочная'           # Название папки с файлами обучающей выборки
val_directory = 'Проверочная'               # Название папки с файлами проверочной выборки
```
```
train_images = []       # Создаем пустой список для хранений оригинльных изображений обучающей выборки
val_images = []         # Создаем пустой список для хранений оригинльных изображений проверочной выборки

cur_time = time.time()  # Засекаем текущее время
for filename in sorted(os.listdir(directory + train_directory + '/Исходные')):  # Проходим по всем файлам в каталоге по указанному пути     
    train_images.append(image.load_img(os.path.join(directory + train_directory + '/Исходные', filename),
                                       target_size=(img_width, img_height)))    # Читаем очередную картинку и добавляем ее в список изображения с указанным target_size                                                      
print ('Обучающая выборка загржуена. Время загрузки: ', round(time.time() - cur_time, 2), 'c', sep='') # Отображаем время загрузки картинок обучающей выборки
print ('Количество изображений: ', len(train_images))                           # Отображаем количество элементов в обучающей выборке

cur_time = time.time()  # Засекаем текущее время
for filename in sorted(os.listdir(directory + val_directory + '/Исходные')):    # Проходим по всем файлам в каталоге по указанному пути                  
    val_images.append(image.load_img(os.path.join(directory + val_directory + '/Исходные', filename), 
                                     target_size=(img_width, img_height)))      # Читаем очередную картинку и добавляем ее в список изображения с указанным target_size   
print ('Проверочная выборка загржуена. Время загрузки: ', round(time.time() - cur_time, 2), 'c', sep='') # Отображаем время загрузки картинок проверочной выборки
print ('Количество изображений: ', len(val_images))                             # Отображаем количество элементов в проверочной выборке
```
```
train_segments = [] # Создаем пустой список для хранений оригинльных изображений обучающей выборки
val_segments = []   # Создаем пустой список для хранений оригинльных изображений проверочной выборки

cur_time = time.time()  # Засекаем текущее время
for filename in sorted(os.listdir(directory + train_directory + '/Размеченные')):   # Проходим по всем файлам в каталоге по указанному пути     
    train_segments.append(image.load_img(os.path.join(directory + train_directory + '/Размеченные', filename),
                                       target_size=(img_width, img_height)))        # Читаем очередную картинку и добавляем ее в список изображения с указанным target_size                                                      
print('Обучающая выборка загржуена. Время загрузки: ', round(time.time() - cur_time, 2), 'c', sep='')       # Отображаем время загрузки картинок обучающей выборки
print('Количество изображений: ', len(train_segments))                              # Отображаем количество элементов в обучающем наборе сегментированных изображений

cur_time = time.time()  # Засекаем текущее время

# Проходим по всем файлам в каталоге по указанному пути 
for filename in sorted(os.listdir(directory + val_directory + '/Размеченные')):
    # Читаем очередную картинку и добавляем ее в список изображения с указанным target_size                                                      
    val_segments.append(image.load_img(os.path.join(directory + val_directory + '/Размеченные', filename), 
                                     target_size=(img_width, img_height)))          # Читаем очередную картинку и добавляем ее в список изображения с указанным target_size   
print ('Проверочная выборка загржуена. Время загрузки: ', round(time.time() - cur_time, 2), 'c', sep='')    # Отображаем время загрузки картинок проверочной выборки
print ('Количество изображений: ', len(val_segments))                               # Отображаем количество элементов в проверочном наборе сегментированных изображений
```
```
xTrain = []                     # Создаем пустой список под обучающую выборку
for img in train_images:        # Проходим по всем изображениям из train_images
  x = image.img_to_array(img)   # Переводим изображение в numpy-массив размерностью: высота - ширина - количество каналов
  xTrain.append(x)              # Добавляем очередной элемент в xTrain
xTrain = np.array(xTrain)       # Переводим в numpy

xVal = []                       # Создаем пустой список под проверочную выборку
for img in val_images:          # Проходим по всем изображениям из val_images
  x = image.img_to_array(img)   # Переводим изображение в numpy-массив размерностью: высота - ширина - количество каналов
  xVal.append(x)                # Добавляем очередной элемент в xTrain
xVal = np.array(xVal)           # Переводим в numpy

print(xTrain.shape)             # Размерность обучающей выборки
print(xVal.shape)               # Размерность проверочной выборки
```
```
cur_time = time.time()                          # Засекаем текущее время
yTrain = yt_prep(train_segments, num_classes)   # Создаем yTrain
print('Время обработки: ', round(time.time() - cur_time, 2),'c') # Выводим время работы
```
```
cur_time = time.time()                          # Засекаем текущее время
yVal = yt_prep(val_segments, num_classes)       # Создаем yVal
print('Время обработки: ', round(time.time() - cur_time, 2),'c') # Выводим время работы
```
[:arrow_up:Оглавление](#4)
<a name="2"></a>
## Линейная модель
```
def linearSegmentationNet(
      num_classes = 11,
      input_shape = (144, 156, 3)
      ):
    img_input = Input(input_shape)                                          # Создаем входной слой с размерностью input_shape
    x = Conv2D(128, (3, 3), padding='same', name='block1_conv1')(img_input) # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same', name='block1_conv2')(x)         # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)                                             # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                               # Добавляем слой Activation

    x = Conv2D(num_classes,(3, 3), activation='softmax', padding='same')(x) # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    model = Model(img_input, x)                                             # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель
    model.compile(optimizer=Adam(lr=1e-3),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    return model # Возвращаем сформированную модель
```
Определяем модель, запускаем обучение.
```
modelL = linearSegmentationNet(num_classes, (img_width, img_height, 3))                         # Создаем моель linearSegmentationNet
history = modelL.fit(xTrain, yTrain, epochs=20, batch_size=32, validation_data=(xVal, yVal))    # Обучаем модель на выборке по трем классам
processImage(modelL, 5, num_classes)
```
![Иллюстрация к проекту](https://github.com/maximAI/Autoencoder/blob/main/Screenshot_2.jpg)
[:arrow_up:Оглавление](#4)
<a name="3"></a>
## U-net
```
def unet(num_classes = 11, input_shape= (144, 256, 3)):
    img_input = Input(input_shape)                                         # Создаем входной слой с размерностью input_shape

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input) # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)         # Добавляем Conv2D-слой с 64-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    block_1_out = Activation('relu')(x)                                    # Добавляем слой Activation и запоминаем в переменной block_1_out

    x = MaxPooling2D()(block_1_out)                                        # Добавляем слой MaxPooling2D

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)        # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)        # Добавляем Conv2D-слой с 128-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    block_2_out = Activation('relu')(x)                                    # Добавляем слой Activation и запоминаем в переменной block_2_out

    x = MaxPooling2D()(block_2_out)                                        # Добавляем слой MaxPooling2D

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)        # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)        # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)        # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    block_3_out = Activation('relu')(x)                                    # Добавляем слой Activation и запоминаем в переменной block_3_out

    x = MaxPooling2D()(block_3_out)                                        # Добавляем слой MaxPooling2D

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(x)        # Добавляем Conv2D-слой с 512-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)        # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)        # Добавляем Conv2D-слой с 256-нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    block_4_out = Activation('relu')(x)                                    # Добавляем слой Activation и запоминаем в переменной block_4_out
    x = block_4_out 

    # UP 2
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(x)    # Добавляем слой Conv2DTranspose с 256 нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = concatenate([x, block_3_out])                                      # Объединем текущий слой со слоем block_3_out
    x = Conv2D(256, (3, 3), padding='same')(x)                             # Добавляем слой Conv2D с 256 нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(256, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    # UP 3
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(x)    # Добавляем слой Conv2DTranspose с 128 нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = concatenate([x, block_2_out])                                      # Объединем текущий слой со слоем block_2_out
    x = Conv2D(128, (3, 3), padding='same')(x)                             # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)                                            # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                                              # Добавляем слой Activation

    x = Conv2D(128, (3, 3), padding='same')(x)  # Добавляем слой Conv2D с 128 нейронами
    x = BatchNormalization()(x)                 # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                   # Добавляем слой Activation

    # UP 4
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(x)      # Добавляем слой Conv2DTranspose с 64 нейронами
    x = BatchNormalization()(x)                 # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                   # Добавляем слой Activation

    x = concatenate([x, block_1_out])           # Объединем текущий слой со слоем block_1_out
    x = Conv2D(64, (3, 3), padding='same')(x)   # Добавляем слой Conv2D с 64 нейронами
    x = BatchNormalization()(x)                 # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                   # Добавляем слой Activation

    x = Conv2D(64, (3, 3), padding='same')(x)   # Добавляем слой Conv2D с 64 нейронами
    x = BatchNormalization()(x)                 # Добавляем слой BatchNormalization
    x = Activation('relu')(x)                   # Добавляем слой Activation

    x = Conv2D(num_classes, (3, 3), activation='softmax', padding='same')(x)  # Добавляем Conv2D-Слой с softmax-активацией на num_classes-нейронов

    model = Model(img_input, x) # Создаем модель с входом 'img_input' и выходом 'x'

    # Компилируем модель 
    model.compile(optimizer=Adam(),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])
    
    return model # Возвращаем сформированную модель
```
Объявляем модель, запускаем обучение.
```
modelUnet = unet(num_classes, (img_width, img_height, 3)) # Создаем модель unet
history = modelUnet.fit(xTrain, yTrain, epochs=20, batch_size=16, validation_data = (xVal, yVal)) # Обучаем модель на выборке по трем классам
processImage(modelUnet, 5, num_classes)
```
![Иллюстрация к проекту](https://github.com/maximAI/Autoencoder/blob/main/Screenshot_2.jpg)
[:arrow_up:Оглавление](#4)

[Ноутбук](https://colab.research.google.com/drive/1cvM-dnxKJK6JTYVjq2CbJrh2t03kJejY?usp=sharing)
