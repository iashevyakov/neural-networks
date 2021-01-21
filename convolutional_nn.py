from keras_preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D


batch_size = 32
# Количество классов изображений
nb_classes = 4
# Количество эпох для обучения
nb_epoch = 25
# Размер изображений
img_rows, img_cols = 32, 32
# Количество каналов в изображении: RGB
img_channels = 3

datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = datagen.flow_from_directory('./cnn_data/train', target_size=(img_rows, img_cols),
                                              batch_size=batch_size,
                                              class_mode='categorical')
val_generator = datagen.flow_from_directory('./cnn_data/val', target_size=(img_rows, img_cols), batch_size=batch_size,
                                            class_mode='categorical')
test_generator = datagen.flow_from_directory('./cnn_data/test', target_size=(img_rows, img_cols), batch_size=batch_size,
                                             class_mode='categorical')

model = Sequential()
# Первый сверточный слой
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(32, 32, 3), activation='relu'))
# Второй сверточный слой
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# Первый слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
# Третий сверточный слой
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
# Четвертый сверточный слой
model.add(Conv2D(64, (3, 3), activation='relu'))
# Второй слой подвыборки
model.add(MaxPooling2D(pool_size=(2, 2)))
# Слой регуляризации Dropout
model.add(Dropout(0.25))
# Слой преобразования данных из 2D представления в плоское
model.add(Flatten())
# Полносвязный слой для классификации
model.add(Dense(512, activation='relu'))
# Слой регуляризации Dropout
model.add(Dropout(0.5))
# Выходной полносвязный слой
model.add(Dense(nb_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# Обучаем модель

model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.samples // batch_size,
                    epochs=nb_epoch,
                    validation_data=val_generator,
                    validation_steps=val_generator.samples // batch_size)

# Оцениваем качество обучения модели на тестовых данных
scores = model.evaluate_generator(test_generator)
print("Точность работы на тестовых данных: %.2f%%" % (scores[1] * 100))
