import tensorflow as tf
from constants import tf_params as params

class VisualCortex(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(VisualCortex, self).__init__()
        self.layers = [ 
            layer['class'](**layer['kwargs']) \
                for layer in params['visual_cortex']
        ]   

    def call(self, image):
        out = image
        for layer in self.layers:
            out = layer(out)
        return out 

class ProprioreceptiveCortex(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(ProprioreceptiveCortex, self).__init__()
        self.layers = [
            layer['class'](**layer['kwargs']) \
                for layer in params['proprioreceptive_cortex']
        ]
        
    def call(self, sensors):
        out = sensors
        for layer in self.layers:
            out = layer(out)
        return out
    
class ActionPreprocessing(tf.keras.layers.Layer):
    def __init__(self, trainable=True, name=None, dtype=None, dynamic=False, **kwargs):
        super(ActionPreprocessing, self).__init__()
        self.layers = [
            layer['class'](**layer['kwargs']) \
                for layer in params['action_preprocessing_layers']
        ]
        
    def call(self, action):
        out = action
        for layer in self.layers:
            out = layer(out)

        return out
