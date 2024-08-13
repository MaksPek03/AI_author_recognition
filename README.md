This project implements an artificial neural network (ANN) classifier for recognizing the author of Polish handwritten text using the HPT (Handwritten Polish Text) dataset. The model is trained using backpropagation to classify handwritten samples into one of eight authors.


Data Preparation: Handwritten text images are preprocessed by converting to grayscale, cropping words based on bounding box coordinates, resizing them, and converting them back to RGB format.

Model Architecture: A Convolutional Neural Network (CNN) with multiple convolutional layers, max-pooling layers, and dense layers is used to classify the images into the corresponding authors.

Training: The model is trained using the Adam optimizer and sparse_categorical_crossentropy loss function over 20 epochs, with a validation split to evaluate performance.

Evaluation: The model's accuracy is evaluated on a separate test set, providing insight into its performance in recognizing handwritten text authors.
