import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import Regularizer


class Lasso(Regularizer):
    """Regularizer for L21 regularization.
    # Arguments
        C: Float; L21 regularization factor.
    """

    def __init__(self, C=0.):
        self.C = K.cast_to_floatx(C)

    def __call__(self, x):
        const_coeff = np.sqrt(K.int_shape(x)[1])
        return self.C*const_coeff*K.sum(K.sqrt(K.sum(K.square(x), axis=1)))

    def get_config(self):
        return {'C': float(self.C)}