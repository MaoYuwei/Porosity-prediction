# detection_prediction_framework

First use detection module to get detected labels, then use prediction module to predict the next layer labels.

The example input data is test.npy file, where have 15 layers. Using the first 6 layers, the framework can predict the next layer labels, i.e. 7-th layer. Then, the 8-th layer can be predicted using the information of 2-7 layers.

