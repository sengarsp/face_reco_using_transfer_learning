# face_reco_using_transfer_learning

# Face Recognition using the Transfer Learning and VGG16
Face-Recognition-
In this we have used the Transfer Learning for training the model and recognition the faces.
# TASK 4:-
To create a model which detect the face or make a face detection model using the transfer learning

# Solution :-
We have used the VGG16 to train the model with our images .
# Step 1:-
We have import the pre-trained model or load the VGG16 model.
![Screenshot (314)](https://user-images.githubusercontent.com/55234454/87397164-7fbab280-c5d1-11ea-8230-c002edfb249f.png)
# Step 2:-
Freeze all layers of the model expext the last layers as we have to make changes in that layer.
![Screenshot (316)](https://user-images.githubusercontent.com/55234454/87397395-ec35b180-c5d1-11ea-8f73-30f228cd37ab.png)
# Step 3:-
Here we create the layer creation to train our model.
![Screenshot (318)](https://user-images.githubusercontent.com/55234454/87397509-17200580-c5d2-11ea-88ec-e931c51abc04.png)

# Step 4:-
 Now add the layer which we have created to the VGG16 layers where we have freezed all the layers.
 ![Screenshot (321)](https://user-images.githubusercontent.com/55234454/87397606-3a4ab500-c5d2-11ea-92f2-47c0e6e8fad6.png)

# Step 5:-
Now load the datasets which you have to train and this should be in the folder and the output depends on the folder name.
![Screenshot (323)](https://user-images.githubusercontent.com/55234454/87397913-b80ec080-c5d2-11ea-9fb8-3a815272f8b9.png)

# Step 6:-
Now compile the model and we have use image data-generator to generate our images for the training.
![Screenshot (325)](https://user-images.githubusercontent.com/55234454/87397993-dd033380-c5d2-11ea-82c9-1531de758313.png)

# Step 7:-
Now finally train the model and after the epochs completed then save that model.
![Screenshot (327)](https://user-images.githubusercontent.com/55234454/87398133-16d43a00-c5d3-11ea-80b5-c869444f3577.png)

# Step 8:
Now save our model and load the saved model to test our model.

![Screenshot (329)](https://user-images.githubusercontent.com/55234454/87398233-3b301680-c5d3-11ea-83e6-4d036c47d1b7.png)

# Step 9:-
Test our model with the testing images we have putted in the folder. Here we have made a small python programming to do so and used cv2 module to show the images.

Here the python program is using the random photo from the test folder for testing.
![Screenshot (331)](https://user-images.githubusercontent.com/55234454/87398312-5733b800-c5d3-11ea-990a-a3c56bf33c23.png)

# OUTPUT:-
 # SRI DEVI
![Uploading Screenshot (309).png…]()
![Uploading Screenshot (305).png…]()
![Uploading Screenshot (301).png…]()




