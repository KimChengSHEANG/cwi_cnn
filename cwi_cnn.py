import tensorflow as tf


class CWI_CNN(object):
    # def conv2D(self, x, num_filters, filter_size):

    def __init__(self, sequence_length, num_classes, embedding_dims, filter_sizes, num_filters,
                 l2_reg_lambda=0.0):
        print("Loading model...")
        self.input_x = tf.placeholder(tf.float32, [None, sequence_length, embedding_dims], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        # print(self.input_x.shape)

        # initializer = tf.keras.initializers.glorot_normal
        # Keeping track of l2 regularization loss (Optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.variable_scope("text-embedding"):
            # self.W = tf.Variable(tf.random_uniform([vocab_size, embedding_dims], -1.0, 1.0), name="W")
            # print("input_x: ", self.input_x.shape)
            # self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            self.embedded_chars_expanded = tf.expand_dims(self.input_x, -1)
        #
        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for filter_size in filter_sizes:
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_dims, 1, num_filters]
                # W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                W = tf.Variable(tf.random_normal(filter_shape), name="W")
                b = tf.Variable(tf.random_normal([num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded, W, strides=[1, 1, 1, 1], padding="VALID", name="conv")
                # Apply nonlinarity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding="VALID", name="pool")

                # h_drop = tf.nn.dropout(pooled, self.dropout_keep_prob)
                # print("pooled: ", pooled.shape)
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, len(filter_sizes))
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])


        # Add dropout
        with tf.variable_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final scores and predictions
        with tf.name_scope("output"):
            self.dense1 = tf.contrib.layers.fully_connected(
                inputs=self.h_drop,
                num_outputs=256,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.random_normal_initializer(),
                trainable=False
            )
            self.h_drop1 = tf.nn.dropout(self.dense1, self.dropout_keep_prob)
            self.dense2 = tf.contrib.layers.fully_connected(
                inputs=self.h_drop1,
                num_outputs=64,
                activation_fn=tf.nn.relu,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.random_normal_initializer(),
                trainable=False
            )
            self.h_drop2 = tf.nn.dropout(self.dense2, self.dropout_keep_prob)
            self.output = tf.contrib.layers.fully_connected(
                inputs=self.h_drop2,
                num_outputs=num_classes,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.random_normal_initializer(),
                trainable=False
            )
            # self.h_drop = tf.nn.dropout(self.out3, self.dropout_keep_prob)

            # self.predictions = tf.cast(self.output > 0.5, tf.int64)
            # self.predictions = tf.nn.softmax(self.output, name="predictions")

            # W = tf.get_variable("W", shape=[num_classes], initializer=tf.contrib.layers.xavier_initializer())
            # b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
            # l2_loss += tf.nn.l2_loss(W)
            # l2_loss += tf.nn.l2_loss(b)
            # self.output = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # self.output = tf.nn.sigmoid(self.out3)
            # self.predictions = tf.argmax(self.output, 1, name="predictions")
            # self.predictions = tf.nn.softmax_cross_entropy_with_logits_v2()
            # self.predictions = tf.cast(tf.nn.softmax(self.output), tf.float32)

            # Model-Predict
            self.predictions = tf.argmax(tf.nn.softmax(self.output), 1, name="predictions")
            # self.predictions = tf.nn.softmax(self.model, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            # losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.input_y)
            losses = tf.nn.weighted_cross_entropy_with_logits(targets=self.input_y, logits=self.output, pos_weight=1.5)
            # self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

            self.l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            # self.l2 = tf.nn.l2_loss(W)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * self.l2
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
