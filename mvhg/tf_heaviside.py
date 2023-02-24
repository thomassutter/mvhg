import tensorflow as tf

import uuid


@tf.RegisterGradient("HeavisideGrad")
def _heaviside_grad(unused_op: tf.Operation, grad: tf.Tensor):
    return tf.maximum(0.0, 1.0 - tf.abs(unused_op.inputs[0])) * grad


def heaviside(x: tf.Tensor, g: tf.Graph = tf.compat.v1.get_default_graph()):
    custom_grads = {"Identity": "HeavisideGrad"}
    with g.gradient_override_map(custom_grads):
        i = tf.identity(x, name="identity_" + str(uuid.uuid1()))
        ge = tf.greater_equal(x, 0, name="ge_" + str(uuid.uuid1()))
        # tf.stop_gradient is needed to exclude tf.to_float from derivative
        step_func = i + tf.stop_gradient(tf.cast(ge, tf.float32) - i)
        return step_func
