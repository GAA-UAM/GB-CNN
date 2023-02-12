# GB-CNN

Gradient Boosted - Convolutional Neural Network

GB-CNN is the python library for working with Gradient Boosted - Convolutional Neural Network.

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

Development
-----------

Our latest algorithm is available on the `main` branch of the repository.

Contributing
------------

Issues can be reported.

If you want to implement any new features, please have a discussion about it on the issue tracker or open a pull request.

Examples
========

In the following, you can find a snippet code that demonstrates a simple GB-CNN training on CIFAR-10, 2D-image dataset. To find out more about the model's ability, please refer to the algorithm and corresponding paper.

```python
import tensorflow as tf
from model.gbcnn import GBCNN
from argparse import Namespace
from keras.utils import np_utils
from Libs.config import get_config

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

def prepare_data(X, size, channels):
    X = X.reshape(X.shape[0], size, size, channels)
    X = X.astype("float32")
    return X/255.

X_train = prepare_data(x_train, 32, 3)
X_test = prepare_data(x_test, 32, 3)

Y_train = np_utils.to_categorical(y_train, 100)
Y_test = np_utils.to_categorical(y_test, 100)

with tf.device('/gpu:0'):
    model = GBCNN(config=get_config())

    params = {'config': Namespace(seed=111,
                                boosting_epoch=40,
                                boosting_eta=1e-3,
                                boosting_patience=4,
                                save_check_points=False,
                                additive_epoch=200,
                                additive_batch=128,
                                additive_units=20,
                                additive_eta=1e-3,
                                additive_patience=3)}

    model.set_params(**params)
    print(model.get_params())
    model.fit(X_train, Y_train, X_test, Y_test)
  
print(f"GB-CNN SCORE:{model.score(X_test, Y_test)}")
```

# Citing

Please use the following bibtex for citing `GB-CNN` in your research:

```
@inproceedings{gbcnn,
  title={Gradient Boosted - Convolutional Neural Network},
  author={Seyedsaman Emami, Gonzalo Martínez-Muñoz},
  year={2022},
  organization={UAM}
}
```

License
=======

The package is licensed under the [GNU Lesser General Public License v2.1](https://github.com/GAA-UAM/GBNN/blob/main/LICENSE).

# Version

0.0.1

# Date-released

03.Feb.2023
