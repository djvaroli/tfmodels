

from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model


class _ModelExtension(Model):
    def __init__(self, *args, **kwargs):
        super(_ModelExtension, self).__init__(*args, **kwargs)

    def plot(self):
        return plot_model(self, show_shapes=True)