from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

target_size = (300, 150)
input_shape = (target_size[0], target_size[1], 3)  


num_of_authors = 8
max_num_of_words_per_author = 700
max_num_for_test = 10

train_images = [] 
train_labels_author = []
train_labels_word = [] 

test_images = []
test_labels_author = []
test_labels_word = []



for author_no in range(num_of_authors):
    file_desc_name = "HPT_handwritten_polish_text_dataset_final/author" + str(author_no + 1) + "/word_places.txt"
    file_desc_ptr = open(file_desc_name, 'r')
    text = file_desc_ptr.read()
    lines = text.split('\n')
    number_of_lines = len(lines) - 1
    row_values = lines[0].split()
    number_of_values = len(row_values)

    num_of_words = 0
    image_file_name_prev = ""
    for i in range(number_of_lines):
        row_values = lines[i].split()
        
        if row_values[0] != '%':
            number_of_values = len(row_values)
            image_file_name = "HPT_handwritten_polish_text_dataset_final/author" + str(author_no + 1) + "\\" + row_values[0][1:-1]

            if image_file_name != image_file_name_prev:   
                # Load the image
                image = cv2.imread(str(image_file_name))
                # Convert the image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_file_name_prev = image_file_name
            
            word = row_values[1]
            row1, column1, row2, column2 = int(row_values[2]), int(row_values[3]), \
                int(row_values[4]), int(row_values[5])
            subimage = gray_image[row1:row2,column1:column2] 
            if subimage.size > 0:
                subimage_resized = cv2.resize(subimage, target_size)
                # Convert to 3 channels for consistency with the input_shape
                subimage_resized = cv2.cvtColor(subimage_resized, cv2.COLOR_GRAY2RGB)
                train_images.append(subimage_resized)
                train_labels_author.append(author_no+1) 
                train_labels_word.append(word)

            num_of_words += 1

        if num_of_words >= max_num_of_words_per_author: 
            break

    file_desc_ptr.close()

train_images = np.array(train_images)
train_labels_author = np.array(train_labels_author)
train_labels_word = np.array(train_labels_word)

X_train, X_test, y_train, y_test = train_test_split(train_images, train_labels_author, test_size=0.3, random_state=42)

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),  
    keras.layers.MaxPooling2D((2, 2)),  
    keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),  
    keras.layers.MaxPooling2D((2, 2)),  
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(num_of_authors+1, activation='softmax')
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
