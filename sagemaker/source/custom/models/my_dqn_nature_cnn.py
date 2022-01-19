from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.visionnet_v1 import _get_filter_config
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.framework import try_import_tf

from ray.rllib.models import ModelCatalog

tf = try_import_tf()

def get_conv_activation(model_config):
    if model_config.get("conv_activation") == "linear":
        activation = None
    else:
        activation = getattr(tf.nn, model_config.get("conv_activation"))
    return activation


def conv_layers(x, model_config, obs_space, prefix=""):
    filters = model_config.get("conv_filters")
    if not filters:
        filters = _get_filter_config(obs_space.shape)

    activation = get_conv_activation(model_config)

    for i, (out_size, kernel, stride) in enumerate(filters, 1):
        x = tf.keras.layers.Conv2D(
            out_size,
            kernel,
            strides=(stride, stride),
            activation=activation,
            padding="same",
            data_format="channels_last",
            name=f"{prefix}conv{i}",
        )(x)
    return x


class MyDQNNatureCNN(TFModelV2):
    
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyDQNNatureCNN, self).__init__(
            obs_space, action_space, num_outputs, model_config, name
        )

        inputs = tf.keras.layers.Input(shape=obs_space.shape, name="observations")
        last_layer = inputs
        # Build the conv layers
        last_layer = conv_layers(last_layer, model_config, obs_space)
        # Flatten the last conv layer
        last_layer = tf.keras.layers.Flatten()(last_layer)
        
        self.base_model = tf.keras.Model(inputs, last_layer)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        # explicit cast to float32 needed in eager
        logits = self.base_model(tf.cast(input_dict["obs"], tf.float32))
        return logits, state

