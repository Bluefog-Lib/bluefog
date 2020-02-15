import tensorflow as tf
import bluefog.tensorflow as bf

# Bluefog: initialize Bluefog.
bf.init()

# Bluefog: pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[bf.local_rank()], 'GPU')

(mnist_images, mnist_labels), _ = \
    tf.keras.datasets.mnist.load_data(path='mnist-%d.npz' % bf.rank())

dataset = tf.data.Dataset.from_tensor_slices(
    (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
     tf.cast(mnist_labels, tf.int64))
)
dataset = dataset.repeat().shuffle(10000).batch(128)

mnist_model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, [3, 3], activation='relu'),
    tf.keras.layers.Conv2D(64, [3, 3], activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
loss = tf.losses.SparseCategoricalCrossentropy()

# Bluefog: adjust learning rate based on number of GPUs.
opt = tf.optimizers.SGD(0.01 * bf.size())

@tf.function
def training_step(imgs, labels_, is_fisrt_batch):
    with tf.GradientTape() as tape:
        probs = mnist_model(imgs, training=True)
        _loss_value = loss(labels_, probs)
    tf.print("_loss_value: ", _loss_value)

    # Bluefog: add Bluefog Distributed GradientTape.
    tape = bf.DistributedGradientTape(tape)

    grads = tape.gradient(_loss_value, mnist_model.trainable_variables)
    opt.apply_gradients(zip(grads, mnist_model.trainable_variables))
    # Bluefog: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    #
    # Note: broadcast should be done after the first gradient step to ensure optimizer
    # initialization.
    if is_fisrt_batch:
        bf.broadcast_variables(
            mnist_model.variables + opt.variables(), root_rank=0)
        tf.print("broadcast is Done!")

    return _loss_value

tf.config.experimental_run_functions_eagerly(False)

#TODO(ybc) Unfortunately, the tensorflow code won't work because the tensorflow is triggered
# through the graph flow engine so that the parallel ops can occur with any order. The mismatch
# of allreduce/broadcast order will cause the procedure crash or hang. 
# One-sided ops can avoid this problem in theory.

# Bluefog: adjust number of steps based on number of GPUs.
for batch, (images, labels) in enumerate(dataset.take(10000 // bf.size())):
    loss_value = training_step(images, labels, batch == 0)
    if batch % 2 == 0 and bf.local_rank() == 0:
        print('Step #%d\tLoss: %.6f' % (batch, loss_value))
