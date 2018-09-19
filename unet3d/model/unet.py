import numpy as np
from keras import backend as K
from keras import initializers, regularizers, constraints
from keras.utils.generic_utils import get_custom_objects
from keras.engine import Layer, InputSpec
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.optimizers import Adam
import sys,os



# from unet3d.metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient
from metrics import dice_coefficient_loss, get_label_dice_coefficient_function, dice_coefficient


"""  Begin OF KERAS-CONTRIB  """

class InstanceNormalization(Layer):
    """Instance normalization layer (Lei Ba et al, 2016, Ulyanov et al., 2016).
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if (self.axis is not None):
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'InstanceNormalization': InstanceNormalization})


class BatchRenormalization(Layer):
    """Batch renormalization layer (Sergey Ioffe, 2017).

    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.

    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchRenormalization`.
        momentum: momentum in the computation of the
            exponential average of the mean and standard deviation
            of the data, for feature-wise normalization.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        epsilon: small float > 0. Fuzz parameter.
            Theano expects epsilon >= 1e-5.
        r_max_value: Upper limit of the value of r_max.
        d_max_value: Upper limit of the value of d_max.
        t_delta: At each iteration, increment the value of t by t_delta.
        weights: Initialization weights.
            List of 2 Numpy arrays, with shapes:
            `[(input_shape,), (input_shape,)]`
            Note that the order of this list is [gamma, beta, mean, std]
        beta_initializer: name of initialization function for shift parameter
            (see [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        gamma_initializer: name of initialization function for scale parameter (see
            [initializers](../initializers.md)), or alternatively,
            Theano/TensorFlow function to use for weights initialization.
            This parameter is only relevant if you don't pass a `weights` argument.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        gamma_regularizer: instance of [WeightRegularizer](../regularizers.md)
            (eg. L1 or L2 regularization), applied to the gamma vector.
        beta_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the beta vector.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    def __init__(self, axis=-1, momentum=0.99, center=True, scale=True, epsilon=1e-3,
                 r_max_value=3., d_max_value=5., t_delta=1e-3, weights=None, beta_initializer='zero',
                 gamma_initializer='one', moving_mean_initializer='zeros',
                 moving_variance_initializer='ones', gamma_regularizer=None, beta_regularizer=None,
                 beta_constraint=None, gamma_constraint=None, **kwargs):
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.momentum = momentum
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.initial_weights = weights
        self.r_max_value = r_max_value
        self.d_max_value = d_max_value
        self.t_delta = t_delta
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

        super(BatchRenormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape,
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint,
                                         name='{}_gamma'.format(self.name))
        else:
            self.gamma = None

        if self.center:
            self.beta = self.add_weight(shape,
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint,
                                        name='{}_beta'.format(self.name))
        else:
            self.beta = None

        self.running_mean = self.add_weight(shape, initializer=self.moving_mean_initializer,
                                            name='{}_running_mean'.format(self.name),
                                            trainable=False)

        self.running_variance = self.add_weight(shape, initializer=self.moving_variance_initializer,
                                                name='{}_running_std'.format(self.name),
                                                trainable=False)

        self.r_max = K.variable(1, name='{}_r_max'.format(self.name))

        self.d_max = K.variable(0, name='{}_d_max'.format(self.name))

        self.t = K.variable(0, name='{}_t'.format(self.name))

        self.t_delta_tensor = K.constant(self.t_delta)

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, inputs, training=None):
        assert self.built, 'Layer must be built before being called'
        input_shape = K.int_shape(inputs)

        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        mean_batch, var_batch = K.moments(inputs, reduction_axes, shift=None, keep_dims=False)
        std_batch = (K.sqrt(var_batch + self.epsilon))

        r = std_batch / (K.sqrt(self.running_variance + self.epsilon))
        r = K.stop_gradient(K.clip(r, 1 / self.r_max, self.r_max))

        d = (mean_batch - self.running_mean) / K.sqrt(self.running_variance + self.epsilon)
        d = K.stop_gradient(K.clip(d, -self.d_max, self.d_max))

        if sorted(reduction_axes) == range(K.ndim(inputs))[:-1]:
            x_normed_batch = (inputs - mean_batch) / std_batch
            x_normed = (x_normed_batch * r + d) * self.gamma + self.beta
        else:
            # need broadcasting
            broadcast_mean = K.reshape(mean_batch, broadcast_shape)
            broadcast_std = K.reshape(std_batch, broadcast_shape)
            broadcast_r = K.reshape(r, broadcast_shape)
            broadcast_d = K.reshape(d, broadcast_shape)
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)

            x_normed_batch = (inputs - broadcast_mean) / broadcast_std
            x_normed = (x_normed_batch * broadcast_r + broadcast_d) * broadcast_gamma + broadcast_beta

        # explicit update to moving mean and standard deviation
        self.add_update([K.moving_average_update(self.running_mean, mean_batch, self.momentum),
                         K.moving_average_update(self.running_variance, std_batch ** 2, self.momentum)], inputs)

        # update r_max and d_max
        r_val = self.r_max_value / (1 + (self.r_max_value - 1) * K.exp(-self.t))
        d_val = self.d_max_value / (1 + ((self.d_max_value / 1e-3) - 1) * K.exp(-(2 * self.t)))

        self.add_update([K.update(self.r_max, r_val),
                         K.update(self.d_max, d_val),
                         K.update_add(self.t, self.t_delta_tensor)], inputs)

        if training in {0, False}:
            return x_normed
        else:
            def normalize_inference():
                if sorted(reduction_axes) == range(K.ndim(inputs))[:-1]:
                    x_normed_running = K.batch_normalization(
                        inputs, self.running_mean, self.running_variance,
                        self.beta, self.gamma,
                        epsilon=self.epsilon)

                    return x_normed_running
                else:
                    # need broadcasting
                    broadcast_running_mean = K.reshape(self.running_mean, broadcast_shape)
                    broadcast_running_std = K.reshape(self.running_variance, broadcast_shape)
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                    broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
                    x_normed_running = K.batch_normalization(
                        inputs, broadcast_running_mean, broadcast_running_std,
                        broadcast_beta, broadcast_gamma,
                        epsilon=self.epsilon)

                    return x_normed_running

            # pick the normalized form of inputs corresponding to the training phase
            # for batch renormalization, inference time remains same as batchnorm
            x_normed = K.in_train_phase(x_normed, normalize_inference, training=training)

            return x_normed

    def get_config(self):
        config = {'epsilon': self.epsilon,
                  'axis': self.axis,
                  'center': self.center,
                  'scale': self.scale,
                  'momentum': self.momentum,
                  'gamma_regularizer': initializers.serialize(self.gamma_regularizer),
                  'beta_regularizer': initializers.serialize(self.beta_regularizer),
                  'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
                  'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
                  'beta_constraint': constraints.serialize(self.beta_constraint),
                  'gamma_constraint': constraints.serialize(self.gamma_constraint),
                  'r_max_value': self.r_max_value,
                  'd_max_value': self.d_max_value,
                  't_delta': self.t_delta}
        base_config = super(BatchRenormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


get_custom_objects().update({'BatchRenormalization': BatchRenormalization})


class GroupNormalization(Layer):
    """Group normalization layer

    Group Normalization divides the channels into groups and computes within each group
    the mean and variance for normalization. Group Normalization's computation is independent
     of batch sizes, and its accuracy is stable in a wide range of batch sizes.

    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical to
    Layer Normalization.

    Relation to Instance Normalization:
    If the number of groups is set to the input dimension (number of groups is equal
    to number of channels), then this operation becomes identical to Instance Normalization.

    # Arguments
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=32,
                 axis=-1,
                 epsilon=1e-5,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(GroupNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        dim = input_shape[self.axis]

        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

        if dim < self.groups:
            raise ValueError('Number of groups (' + str(self.groups) + ') cannot be '
                             'more than the number of channels (' +
                             str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError('Number of groups (' + str(self.groups) + ') must be a '
                             'multiple of the number of channels (' +
                             str(dim) + ').')

        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, **kwargs):
        input_shape = K.int_shape(inputs)
        tensor_input_shape = K.shape(inputs)

        # Prepare broadcasting shape.
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(1, self.groups)

        reshape_group_shape = K.shape(inputs)
        group_axes = [reshape_group_shape[i] for i in range(len(input_shape))]
        group_axes[self.axis] = input_shape[self.axis] // self.groups
        group_axes.insert(1, self.groups)

        # reshape inputs to new group shape
        group_shape = [group_axes[0], self.groups] + group_axes[2:]
        group_shape = K.stack(group_shape)
        inputs = K.reshape(inputs, group_shape)

        group_reduction_axes = list(range(len(group_axes)))
        mean, variance = K.moments(inputs, group_reduction_axes[2:], keep_dims=True)
        inputs = (inputs - mean) / (K.sqrt(variance + self.epsilon))

        # prepare broadcast shape
        inputs = K.reshape(inputs, group_shape)

        outputs = inputs

        # In this case we must explicitly broadcast all parameters.
        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            outputs = outputs * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            outputs = outputs + broadcast_beta

        # finally we reshape the output back to the input shape
        outputs = K.reshape(outputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups': self.groups,
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(GroupNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


get_custom_objects().update({'GroupNormalization': GroupNormalization})

"""  END OF KERAS-CONTRIB  """

K.set_image_data_format("channels_first")

print("     ")
print(os.getcwd())
print("     ")

sys.path.append('./keras-contrib')

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate


def unet_model_3d(input_shape, pool_size=(2, 2, 2), n_labels=1, initial_learning_rate=0.00001, deconvolution=False,
                  depth=4, n_base_filters=32, include_label_wise_dice_coefficients=False, metrics=dice_coefficient,
                  batch_normalization=False, activation_name="sigmoid"):
    """
    Builds the 3D UNet Keras model.f
    :param metrics: List metrics to be calculated during model training (default is dice coefficient).
    :param include_label_wise_dice_coefficients: If True and n_labels is greater than 1, model will report the dice
    coefficient for each label as metric.
    :param n_base_filters: The number of filters that the first layer in the convolution network will have. Following
    layers will contain a multiple of this number. Lowering this number will likely reduce the amount of memory required
    to train the model.
    :param depth: indicates the depth of the U-shape for the model. The greater the depth, the more max pooling
    layers will be added to the model. Lowering the depth may reduce the amount of memory required for training.
    :param input_shape: Shape of the input data (n_chanels, x_size, y_size, z_size). The x, y, and z sizes must be
    divisible by the pool size to the power of the depth of the UNet, that is pool_size^depth.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of up-sampling. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model
    """
    inputs = Input(input_shape)
    current_layer = inputs
    levels = list()

    # add levels with max pooling
    for layer_depth in range(depth):
        layer1 = create_convolution_block(input_layer=current_layer, n_filters=n_base_filters*(2**layer_depth),
                                          batch_normalization=batch_normalization)
        layer2 = create_convolution_block(input_layer=layer1, n_filters=n_base_filters*(2**layer_depth)*2,
                                          batch_normalization=batch_normalization)
        if layer_depth < depth - 1:
            current_layer = MaxPooling3D(pool_size=pool_size)(layer2)
            levels.append([layer1, layer2, current_layer])
        else:
            current_layer = layer2
            levels.append([layer1, layer2])

    # add levels with up-convolution or up-sampling
    for layer_depth in range(depth-2, -1, -1):
        up_convolution = get_up_convolution(pool_size=pool_size, deconvolution=deconvolution,
                                            n_filters=current_layer._keras_shape[1])(current_layer)
        concat = concatenate([up_convolution, levels[layer_depth][1]], axis=1)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=concat, batch_normalization=batch_normalization)
        current_layer = create_convolution_block(n_filters=levels[layer_depth][1]._keras_shape[1],
                                                 input_layer=current_layer,
                                                 batch_normalization=batch_normalization)

    final_convolution = Conv3D(n_labels, (1, 1, 1))(current_layer)
    act = Activation(activation_name)(final_convolution)
    model = Model(inputs=inputs, outputs=act)

    if not isinstance(metrics, list):
        metrics = [metrics]

    if include_label_wise_dice_coefficients and n_labels > 1:
        label_wise_dice_metrics = [get_label_dice_coefficient_function(index) for index in range(n_labels)]
        if metrics:
            metrics = metrics + label_wise_dice_metrics
        else:
            metrics = label_wise_dice_metrics

    model.compile(optimizer=Adam(lr=initial_learning_rate), loss=dice_coefficient_loss, metrics=metrics)
    return model


def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """

    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        layer = InstanceNormalization(axis=1)(layer)
    #     try:
    #         # from keras_contrib.layers.normalization import InstanceNormalization
    #         from keras_contrib.layers.normalization import InstanceNormalization
    #     except ImportError:
    #         pass
            # raise ImportError("Install keras_contrib in order to use instance normalization."
            #                   "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")

        # layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)
