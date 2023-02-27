# A Gradient Boosting Approach for Training Convolutional and Deep Neural Networks 	

GB-CNN is the python library for working with Gradient Boosted - Convolutional Neural Networks (GB-CNN) and Gradient Boosted Deep Neural Networks (GB-DNN).

* **GB-CNN** is designed to implement the Gradient Boosted Convolutional layers for the *image* datasets.

* **GB-DNN** is designed to implement the Gradient Boosted Dense layers for the big dimension *tabular* datasets.


# Citing

Please use the following bibtex for citing the paper in your research:

**The paper title:** A Gradient Boosting Approach for Training Convolutional and Deep Neural Networks

```
@article{emami2023gbcnn,
  title={A Gradient Boosting Approach for Training Convolutional and Deep Neural Networks},
  author={Emami, Seyedsaman and Martínez-Muñoz, Gonzalo},
  journal={https://arxiv.org/abs/2302.11327},
  year={2023}
}
```
Or use the [CITATION](CITATION.cff) to cite the package and codes.

License
=======

The package is licensed under the [GNU Lesser General Public License v2.1](https://github.com/GAA-UAM/GBNN/blob/main/LICENSE).

Development
-----------

Our latest algorithm is available on the `main` branch of the repository.

Contributing
------------

Issues can be reported.

If you want to implement any new features, please have a discussion about it on the issue tracker or open a pull request.



# Dependencies

Deep_GBNN has the following non-optional dependencies:

- Python 3.7 or higher
- TensorFlow 2.11 or higher
- Numpy 1.24 or higher
- scipy

Installation
============

To install GB-CNN from the repository:

```
$ git clone https://github.com/GB-CNN
$ cd GB-CNN/
$ pip install -r requirements.txt
$ python setup.py install
```

Examples
========

### GB-CNN
An examples that demonstrate GB-CNN training on CIFAR-10, 2D-image dataset.

 To find out more about the model's ability, please refer to the algorithm and corresponding paper.

```Python
import tensorflow as tf
from models.gbcnn import GBCNN
from keras.utils import np_utils
from Libs.config import get_config

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()


def prepare_data(X, size, channels):
    X = X.reshape(X.shape[0], size, size, channels)
    X = X.astype("float32")
    return X/255.


X_train = prepare_data(x_train, 32, 3)
X_test = prepare_data(x_test, 32, 3)

Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

params = {'config': Namespace(seed=111,
                              boosting_epoch=40,
                              boosting_eta=1e-3,
                              save_records=False,
                              additive_epoch=100,
                              batch=128,
                              units=20,
                              additive_eta=1e-3,
                              patience=2)}

model = GBCNN(config=get_config())
model.set_params(**params)
print(model.get_params())
model.fit(X_train, Y_train, X_test, Y_test)

# Note that the X_test and Y_test in the fit are optional. If None, the validation report will not be generated.
```


### GB-DNN

To train on the tabular dataset, the Deep-GBNN model should be trained.

```Python
from models.gbdnn import GBDNNClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_informative=16, n_classes=10, n_samples=5000)
x_train, x_test, y_train, y_test = train_test_split(X, y)

model = GBDNNClassifier(config=get_config())
model.set_params(**params)
model.fit(x_train, y_train)
print(f"GB-DNN SCORE:{model.score(x_test, y_test)}")

```

> The models developed in a way that the different hyper parameter optimization such as grid search can be easily applied through the config.
To now more about the hyperparameters, check the config.


# Version

0.0.1

# Date-released

03.Feb.2023
