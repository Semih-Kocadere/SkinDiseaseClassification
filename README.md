This project focuses on building a deep learning model to classify skin diseases using the Dermamnist dataset.
The dataset consists of dermatoscopic images of seven different skin diseases, which are preprocessed and resized to 28x28 pixels for this project.
The model is a Convolutional Neural Network (CNN) built using TensorFlow and Keras libraries. It includes:  
Three convolutional layers, each followed by a max-pooling layer.
- A flatten layer.
- Two dense layers.
  
The final dense layer has 10 units, corresponding to the 10 classes of skin diseases.
The model is trained using the Adam optimizer with a learning rate of 0.001, and the loss function used is Sparse Categorical Crossentropy. The model's performance is evaluated based on its accuracy.  During the training process, the training and validation loss and accuracy are visualized over the epochs. These visualizations are saved as a PNG image. Finally, the model's performance is evaluated on a test set, and the test loss and accuracy are printed.  This project is a great example of applying deep learning to medical imaging data for disease classification. It demonstrates the use of CNNs in TensorFlow and Keras, as well as data preprocessing, model training, and performance evaluation.


                                                                DISEASES
![Diseases](https://github.com/user-attachments/assets/0542e48d-3229-4ffe-b945-e1479013d96e)


                                                                LAYERS
![LAY](https://github.com/user-attachments/assets/adf98156-2a7d-4d1f-b05e-61346842fdd4)


                                                                EPOCHS
In the following, we have some part of epochs.
![epoch](https://github.com/user-attachments/assets/878066fd-4458-4bd5-b023-860b15196ada)
![epoch1](https://github.com/user-attachments/assets/85ac89fd-77ae-4866-8bf0-b0470ab8d32c)

                                                                Accuracy

In the last part, we have looked for the accuracy. Our accuracy is pretty good.
![loss](https://github.com/user-attachments/assets/dc1ff247-2a6d-4207-9f0f-cf5a86d30229)
![accuracy](https://github.com/user-attachments/assets/2afdb38b-d3a8-4a6b-a226-ffc8c4b242ad)


