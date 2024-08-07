This project focuses on building a deep learning model to classify skin diseases using the Dermamnist dataset.
The dataset consists of dermatoscopic images of seven different skin diseases, which are preprocessed and resized to 28x28 pixels for this project.
The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras libraries. It includes:  
Three convolutional layers, each followed by a max-pooling layer.
- A flatten layer.
- Two dense layers.
  
The final dense layer has 10 units, corresponding to the 10 classes of skin diseases.
The model is trained using the Adam optimizer with a learning rate of 0.001, and the loss function used is Sparse Categorical Crossentropy. The model's performance is evaluated based on its accuracy.  During the training process, the training and validation loss and accuracy are visualized over the epochs. These visualizations are saved as a PNG image. Finally, the model's performance is evaluated on a test set, and the test loss and accuracy are printed.  This project is a great example of applying deep learning to medical imaging data for disease classification. It demonstrates the use of CNNs in TensorFlow and Keras, as well as data preprocessing, model training, and performance evaluation.
