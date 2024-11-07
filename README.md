🤖 AI Author Recognition for Polish Handwritten Text ✍️
This project implements an Artificial Neural Network (ANN) for author recognition of Polish handwritten text using the HPT (Handwritten Polish Text) dataset. The model utilizes Convolutional Neural Networks (CNNs) trained with backpropagation to classify handwriting samples into different authors.

🚀 Features:
Preprocessing: Converts images to grayscale, resizes, and normalizes.
Model: CNN architecture with convolutional, pooling, and dense layers.
Training: Uses Adam optimizer and sparse categorical cross-entropy loss.
Evaluation: Accuracy evaluation on the test dataset.
📦 Installation
Clone the repository:

bash
Copy code
git clone https://github.com/MaksPek03/AI_author_recognition.git
cd AI_author_recognition
Install dependencies:

Create a virtual environment and install the necessary libraries:

bash
Copy code
python3 -m venv venv
source venv/bin/activate  # On Windows, use venv\Scripts\activate
pip install -r requirements.txt
Run the model:

To train or evaluate the model, simply run:

bash
Copy code
python main.py
📊 Dataset
The project uses the HPT dataset, which contains images of Polish handwritten text. You need to download and preprocess this dataset before starting the training process.
