# CNN-Model-Trainer-Comparator
This project is based on PyTorch library and can be used to train a model / compare two different CNN models.

The following are prerequisites of this project:
1. PyTorch package
2. MatPlotLib - to make graphical comparison for comparing training details of two different models.
3. numpy

Using this project, one can train a CNN model for CIFAR-10 dataset.

The following are some of the insights obtained from my model training:
1. Different learning rate for different models. There is no universal learning rate that can be applied for all models
2. Data Augmentation in data pre-processing improves model's test accuracy which makes a model less prone to overfitting.
3. Dropout helps in obtaining a better model. With optimal dropout rate, model's test accuracy greatly improved while its loss is not so low. In some cases, the loss went very much below 0.01 but its test accuracy stood between 35-45 percent. 
After using dropout, test precision increased to about 55-60% with the loss value between 0.2-0.4
NOTE: Dropout was implemented in a very basic CNN model (with few Convolutional & Max-pooling layers, 1 FC layer)
4. As the dropout percent / the number of dropout layer is increased, the model's performance started deteriorating. This was observed after introducing a dropout after every Convolutional layer in my model.
Hence, optimal dropout is necessary for a better model training.
5. BatchNorm layers are highly useful since they improved a model's convergence
6. Inception layers got potential in shaping the accuracy of a model. The introduction of two Inception layers rose my model's test precision from around 76% to 83.74%.
7. Fine-tuning of hyper-paramters like epoch, batch size and weight initialization techniques are essential handlings in every model training.

This project supports the comparison of two different models and provide details like test accuracy and mean loss after each epoch. A sample graph generated from this project is given below:
![model_comparison](https://user-images.githubusercontent.com/29046579/128039375-e9580296-eb75-4c8b-8892-e56570ad2ade.png)

In the above image, the model training of the second model is very slow becuase of the low learning rate(0.001). The same model shows good improvements when using higher learning rate of the order of (0.01 or 0.1)
