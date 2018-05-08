# Flower-species-Recognition-using-Computer-vision-and-Machine-Learning

## Project idea

What if -

● You build an intelligent system that was trained with massive dataset of
flower/plant images.

● Your system predicts the label/class of the flower/plant using Computer Vision
techniques and Machine Learning algorithms.

● Your system searches the web for all the flower/plant related data after
predicting the label/class of the captured image.

● Your system helps gardeners and farmers to increase their productivity and yield
with the help of automating tasks in garden/farm.

● Your system applies the recent technological advancements such as Internet of
Things (IoT) and Machine Learning in the agricultural domain.

● You build such a system for your home or your garden to monitor your plants
using a Raspberry Pi.

“All the above scenarios need a common task to be done at the first place - Image
Classification”.

Yeah! It is classifying a flower/plant into it’s corresponding class or category. For
example, when our awesome intelligent assistant looks into a Sunflower image, it must
label or classify it as a “Sunflower”.

## Aim

Our aim from the project is to understand how to use Deep Learning models to solve a
Supervised Image Classification problem.

We will be using the pre-trained Deep Neural Nets trained on the ImageNet challenge
that are made publicly available in Keras.

## Dataset

We will specifically use FLOWERS17 dataset of University of Oxford. This dataset is a
highly challenging dataset with 17 classes of flower species, each having 80 images.
So, totally we have 1360 images to train our model.

## Feature extraction using Deep Convolutional Neural Networks

The pre-trained models we will consider are VGG16, VGG19, Inception-v3, Xception,
ResNet50, InceptionResNetv2 and MobileNet. Instead of creating and training deep
neural nets from scratch, what if we use the pre-trained weights of these deep neural
net architectures (trained on ImageNet dataset) and use it for our own dataset.

Traditional machine learning approach uses feature extraction for images using Global
descriptors such as Local Binary Patterns, Histogram of Oriented Gradients, Color
Histograms etc. or Local descriptors such as SIFT, SURF, ORB etc. These are
hand-crafted features that requires domain level expertise.

But here comes Deep Neural Networks! Instead of using hand-crafted features, Deep
Neural Nets automatically learns these features from images in a hierarchical fashion.
Lower layers learn low-level features such as Corners, Edges whereas middle layers
learn color, shape etc. and higher layers learn high-level features representing the object
in the image.

Thus, we can use a Convolutional Neural Network as a Feature Extractor by taking the
activations available before the last fully connected layer of the network (i.e. ​before​ the
final softmax classifier). These activations will be acting as the feature vector for a
machine learning classifier which further learns to classify it. This type of approach is
well suited for Image Classification problems, where instead of training a CNN from
scratch (which is time-consuming and tedious), a pre-trained CNN co
We find the documentation and GitHub repo of Keras well maintained and easy to
understand.

## Beautiful keras

Keras is an amazing library to quickly start Deep Learning for people entering into this
field. Developed by François Chollet, it offers simple understandable functions and
syntax to start building Deep Neural Nets right away instead of worrying too much on
the programming part. Keras is a wrapper for Deep Learning libraries namely Theano
and TensorFlow.

## Training and Testing

After extracting, concatenating and saving features and labels from our training dataset
using ConvNets, it’s time to train our system. To do that, we need to create our Machine
Learning models. For creating our machine learning models, we take the help of
scikit-learn.

We will choose Logistic Regression, K-Nearest Neighbors, Decision Trees, Random
Forests, Gaussian Naive Bayes, Support Vector Machine and even neural networks as our
machine learning models.

Furthermore, we will use train_test_split function provided by scikit-learn to split our
training dataset into train_data and test_data. By this way, we train the models with the
train_data and test the trained model with the unseen test_data.

We will also use a technique called K-fold cross validation, a model-validation technique
which is the best way to predict ML model’s accuracy. In short, if we choose K = 10,
then we split the entire data into 9 parts for training and 1 part for testing uniquely over
each round upto 10 times.

Finally, we train each of our machine learning model and check the cross-validation
results and take the best machine learning model.
Finally this trained machine learning model will be predicted on the unseen data.

## Dependencies

We will need the following Python packages to do the project.

● Theano or TensorFlow - Backend engine to run keras on top of it.

● Keras - It is a wrapper for Deep Learning libraries namely Theano or TensorFlow
and to build Deep Neural Nets.

● NumPy - It is used for different mathematical and data manipulation operations.

● Scikit-learn - A framework to build several machine learning models.

● Matplotlib - It is a plotting library for the Python programming language and its
numerical mathematics extension NumPy.

● Seaborn - It is a Python visualization library based on matplotlib. It provides a
high-level interface for drawing attractive statistical graphics.

● H5py - It lets you store huge amounts of numerical data, basically we will use
H5py to save our extracted features from ConvNets and used the saved
features to further train our machine learning models.

● Pickle - We will make use of it to save our machine learning models to file and
load it later in order to make predictions

## Conclusion

To learn to use the pre-trained Deep Convolutional Neural Nets to extract features from
the images with keras.

To learn how to make use of different python libraries like NumPy, H5py, Pickle for
complex data manipulation tasks and mathematical operations.

To learn how to design several machine learning models, test their accuracies &
compare them while avoiding overfitting with the help of cross validation techniques.

Finally, the goal is that for any supervised image classification problem we can have the same and ideas can be implemented.
