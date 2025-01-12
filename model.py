#第一次用GitHub，略微试试能不能改点东西然后加进去
import keras
from keras import layers
from keras import layers
from keras import regularizers
import os
os.environ["KERAS_BACKEND"] = "jax" 
from keras import backend as K
import tensorflow as tf
import numpy as np
from keras import regularizers

def T_get_edge_feature(point_cloud_series, nn_idx, k=5):
#     """Construct edge feature for each point
#     Please refer to https://github.com/WangYueFt/dgcnn/blob/master/tensorflow/utils/tf_util.py
#     Args:
#     point_cloud_series: (batch_size, time_step, num_points, 1, num_dims)
#                      or (batch_size, time_step, num_points   , num_dims)
#     nn_idx: (batch_size, num_points, k)
#     k: int

#     Returns:
#     edge features: (batch_size, time_step, num_points, k, num_dims)
#     """

    assert len(nn_idx.get_shape().as_list()) == 3
    if point_cloud_series.get_shape().as_list()[-2] == 1:
        point_cloud_series = tf.squeeze(point_cloud_series, -2)

    point_cloud_central = point_cloud_series

    point_cloud_shape = point_cloud_series.get_shape()
    batch_size = tf.shape(point_cloud_series)[0]
    time_step = point_cloud_shape[-3].value
    num_points = point_cloud_shape[-2].value
    num_dims = point_cloud_shape[-1].value

    # Shared graph for all subjects in the batch and all time-frames
    nn_idx = tf.expand_dims(nn_idx, axis=1)
    
    # Create the shared graph

    # Copy the neighborhood definition for each time step
    nn_idx = tf.tile(nn_idx, [1, time_step, 1, 1]) # https://www.tensorflow.org/api_docs/python/tf/tile
    nn_idx = tf.cast(nn_idx, dtype=tf.int32)

    # Create the shared graph for all batches
    idx_ = tf.range(batch_size*time_step) * num_points
    idx_ = tf.reshape(idx_, [batch_size, time_step, 1, 1]) 
    idx_ = tf.cast(idx_, dtype=tf.int32)

    point_cloud_flat = tf.reshape(point_cloud_series, [-1, num_dims])
    # point_cloud_neighbors defined by k-NN
    point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)
    point_cloud_central = tf.expand_dims(point_cloud_central, axis=-2)

    # Copy the central point data for k times
    point_cloud_central = tf.tile(point_cloud_central, [1, 1, 1, k, 1])

    # For each neighbor, one dimension is x_i as the global features,
    # the difference between neighbors and the central point: x_j - x_i, as the local interaction
    # Therefore, feature * 2
    edge_feature = tf.concat([point_cloud_central, point_cloud_neighbors-point_cloud_central], axis=-1)
    return edge_feature

def T_conv_bn_max(edge_feature, kernel=2, activation_fn='relu'):
#     """TimeDistributed conv with max as aggregation
#     Args:
#     edge_feature: (batch_size, time_step, num_points, k, num_dims)
#     kernel: conv kernel units
#     activation_fn: non-linear activation
#
#     Returns:
#     conv with max aggregation: (batch_size, time_step, num_points, 1, kernel)

    net = layers.TimeDistributed(layers.Conv2D(kernel, (1,1)))(edge_feature)
    # net = TimeDistributed(BatchNormalization(axis=-1))(net) # BatchNorm, can be enabled
    if activation_fn is not None:
        net = layers.TimeDistributed(layers.Activation(activation_fn))(net)
    return layers.TimeDistributed(layers.Lambda(lambda x: tf.reduce_max(x, axis=-2, keep_dims=True)))(net)

def T_edge_conv(point_cloud_series, graph, kernel=2, activation_fn='relu', k=5):
    """
    A simplified T_edge_conv that:
    1) tiles the adjacency matrix for each sample in the batch,
    2) constructs edge_feature,
    3) does conv + max-pool,
    4) returns the result.
    """
    def tile_graph_output_shape(input_shapes):
        """
        Suppose graph shape = (ROI_N, ROI_N)
        The output shape => (None, ROI_N, ROI_N).
        """
        graph_shape, pc_shape = input_shapes
        ROI_N1, ROI_N2 = (236,236)  # e.g. (236, 236)
        return (None, ROI_N1, ROI_N2)

    # 1) tile graph
    graph_tiled = layers.Lambda(
        lambda x: tf.tile(
            tf.expand_dims(x[0], axis=0),  # => (1, ROI_N, ROI_N)
            [tf.shape(x[1])[0], 1, 1]      # => (batch_size, ROI_N, ROI_N)
        ),
        output_shape=(1, 1, 236, 8)
    )([graph, point_cloud_series])

    # 2) Build edge features
    edge_feature = layers.Lambda(
        lambda x: T_get_edge_feature(point_cloud_series=x[0], nn_idx=x[1], k=k), 
        output_shape=(1, 1, 236, 8)
    )([point_cloud_series, graph_tiled])

    # 3) Conv + max
    out = T_conv_bn_max(edge_feature, kernel=kernel, activation_fn=activation_fn)

    return out

######################## Model description ########################

def get_model(
    graph_path,
    ROI_N, 
    frames, 
    kernels=[8,8,8,16,32,32], 
    k=5, 
    l2_reg=1e-4, 
    dp=0.5,
    num_classes=100,
    weight_path=None, 
    skip=[0,0]
):
    ############ load static FC matrix ##############
    print('load graph:', graph_path)
    adj_matrix = np.load(graph_path)                             # shape = (ROI_N, ROI_N)
    graph = adj_matrix.argsort(axis=1)[:, ::-1][:, 1:k+1]        # shape = (ROI_N, k)

    ############ define model ############
    # 1) Input for your data
    main_input = layers.Input((frames, ROI_N, 1), name='points')

    # 2) Input for the adjacency matrix
    #    (We expect shape=(ROI_N, ROI_N) if that's truly your adjacency form.)
    static_graph_input = layers.Input(
        shape=(ROI_N, ROI_N),
        dtype=tf.int32,
        name='graph'
    )
    
    # 4 stacking conv layers
    net1 = T_edge_conv(main_input,   graph=static_graph_input, kernel=kernels[0], k=k)
    net2 = T_edge_conv(net1,         graph=static_graph_input, kernel=kernels[1], k=k)
    net3 = T_edge_conv(net2,         graph=static_graph_input, kernel=kernels[2], k=k)
    net4 = T_edge_conv(net3,         graph=static_graph_input, kernel=kernels[3], k=k)
    net  = layers.Lambda(lambda x: tf.concat([x[0], x[1], x[2], x[3]], axis=-1))([net1, net2, net3, net4])

    # 1 final conv layer
    net = T_edge_conv(net, graph=static_graph_input, kernel=kernels[4], k=k)

    # TimeDistributed
    net = layers.TimeDistributed(layers.Dropout(dp))(net)
    net = layers.ConvLSTM2D(kernels[5], kernel_size=(1,1), padding='same', 
                            return_sequences=True, 
                            recurrent_regularizer=regularizers.l2(l2_reg))(net)
    net = layers.TimeDistributed(layers.BatchNormalization())(net)
    net = layers.TimeDistributed(layers.Activation('relu'))(net)
    net = layers.TimeDistributed(layers.Flatten())(net)
    net = layers.TimeDistributed(layers.Dropout(dp))(net)

    # Dense layer with softmax activation
    net = layers.TimeDistributed(
        layers.Dense(num_classes, activation='softmax', 
                     kernel_regularizer=regularizers.l2(l2_reg))
    )(net)
    # Mean prediction from each time frame
    net = layers.Lambda(lambda x: K.mean(x, axis=1))(net)

    output_layer = net
    model = keras.models.Model(inputs=[main_input, static_graph_input], outputs=output_layer)

    # Optionally load weights
    if weight_path:
        print('Load weight:', weight_path)
        pre_model = keras.models.load_model(
            weight_path,
            custom_objects={
                'tf': tf,
                'T_conv_bn_max': T_conv_bn_max,
                'T_edge_conv': T_edge_conv,
                'T_get_edge_feature': T_get_edge_feature
            }
        )
        # Transfer weights
        for i in range(skip[0], len(model.layers)-skip[1]):
            model.layers[i].set_weights(pre_model.layers[i].get_weights())

    return model

if __name__ == "__main__":
    # Overfit on small random datasets
    ROI_N = 236
    random_FC = np.random.rand(ROI_N, ROI_N)
    random_FC[np.diag_indices(ROI_N)] = 1
    np.save('FC_random', random_FC)
    
    N = 50
    frames = 100
    x_train = np.random.normal(0, 1, size=(N, frames, ROI_N, 1))
    x_test = np.random.normal(0, 1, size=(N, frames, ROI_N, 1))
    print('train data shape:', x_train.shape) # train data shape: (50, 100, 236, 1)
    print('test data shape:', x_test.shape) # test data shape: (50, 100, 236, 1)

    num_classes = 2
    label = np.arange(num_classes).repeat(N // num_classes)
    y_train = y_test = keras.utils.to_categorical(label, num_classes)
    print('label shape:', y_train.shape) # label shape: (50, 2)
    for label, count in enumerate(y_train.sum(0)):
        print('Label %d: %d/%d (%.1f%%)'%(label, count, y_train.shape[0], 100.0 * count / y_train.shape[0]))

    model = get_model(
        graph_path='FC_random.npy',
        ROI_N=ROI_N,
        frames=frames, 
        kernels=[8,8,8,16,32,32], 
        k=3, 
        l2_reg=0, 
        dp=0.5,
        num_classes=num_classes, 
        weight_path=None, 
        skip=[0,0])
    model.summary()
    model.compile(loss=['categorical_crossentropy'], 
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])
    checkpointer = keras.callbacks.ModelCheckpoint(monitor='val_acc', filepath='tmp.hdf5', 
                                                verbose=1, save_best_only=True)
    model.fit(x_train, y_train,
            shuffle=True,
            batch_size=4,
            validation_data=(x_test, y_test),
            epochs=50,
            callbacks=[checkpointer])
    # Best Train acc: ~100% (random: 50%).
    # Best Test acc: ~50% (random: 50%).
