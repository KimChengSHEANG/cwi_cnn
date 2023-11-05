import tensorflow as tf
import numpy as np
import os
import datetime
import time

from cwi_cnn import CWI_CNN
import data_helpers as utils
from configure import parse_args
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error
from evaluate import evaluate
import warnings
import sklearn.exceptions


# warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

# tf.logging.set_verbosity(tf.logging.ERROR)


def train(FLAGS):
    train_file_name = os.path.basename(FLAGS.TRAIN_PATH).split('.')[0]
    dev_file_name = os.path.basename(FLAGS.DEV_PATH).split('.')[0]
    train_file = './data/dumps/' + train_file_name + '.pk'
    val_file = './data/dumps/' + dev_file_name + '.pk'

    if not os.path.exists(train_file):
        print("Train dump not found. Preparing data...")
        utils.create_dump(FLAGS.TRAIN_PATH, train_file)

    if not os.path.exists(val_file):
        print("Validation dump not found. Preparing data...")
        utils.create_dump(FLAGS.DEV_PATH, val_file)

    print('Loading dataset from ./data/dumps/...')
    x_train, y_train = utils.fetch(train_file)
    x_val, y_val = utils.fetch(val_file)
    print("x_train.shape:", x_train.shape)
    print("y_train.shape:", y_train.shape)

    # Shuffle training data
    np.random.seed(10)
    shuff_idx = np.random.permutation(np.arange(len(y_train)))
    x_train, y_train = x_train[shuff_idx], y_train[shuff_idx]

    # Define directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))
    print("Writing to {}\n".format(out_dir))
    # write hyperparameters to a file
    # if not os.path.exists(out_dir):
    #     os.makedirs(out_dir)

    utils.write_configure_to_file(FLAGS, out_dir)

    print("Generating graph and starting training...")

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.ALLOW_SOFT_PLACEMENT,
            log_device_placement=FLAGS.LOG_DEVICE_PLACEMENT
        )
        # session_conf.gpu_options.allow_growth = FLAGS.gpu_allow_growth
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cwi = CWI_CNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                embedding_dims=FLAGS.TEXT_EMBEDDING_DIM,
                filter_sizes=list(map(int, FLAGS.FILTER_SIZES.split(","))),
                num_filters=FLAGS.NUM_FILTERS,
                l2_reg_lambda=FLAGS.L2_REG_LAMBDA
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.LEARNING_RATE)
            # optimizer = tf.train.AdagradOptimizer(1e-3)
            # optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)

            grads_and_vars = optimizer.compute_gradients(cwi.loss)
            # capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grads_and_vars]
            # train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)


            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cwi.loss)
            acc_summary = tf.summary.scalar("accuracy", cwi.accuracy)

            # Train summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.NUM_CHECKPOINTS)

            # Initialize all variables
            # sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            sess.run(tf.global_variables_initializer())

            # sess.run(cwi.embedding_init, feed_dict={cwi.embedding_placeholder: embedding})

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                    cwi.input_x: x_batch,
                    cwi.input_y: y_batch,
                    cwi.dropout_keep_prob: FLAGS.DROPOUT_KEEP_PROB
                }
                # print("input_x[0]", x_batch[0])
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cwi.loss, cwi.accuracy], feed_dict
                )
                print("step {}, loss {:g}, accuracy {:g}".format(step, loss, accuracy))
                # print("predictions: ", predictions)
                train_summary_writer.add_summary(summaries, step)
                # step, loss, accuracy = sess.run([global_step, cwi.loss, cwi.accuracy], feed_dict)
                # time_str = datetime.datetime.now().isoformat()
                # print("{}: step {}, loss {:g}, accuracy {:g}". format(time_str, step, loss, accuracy))

            def val_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                global best_accuracy

                feed_dict = {
                    cwi.input_x: x_batch,
                    cwi.input_y: y_batch,
                    cwi.dropout_keep_prob: 1.0
                }
                # print("input_x_val[0]", x_batch[0])
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cwi.loss, cwi.accuracy], feed_dict
                )
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, accuracy {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

                # Save checkpoint
                if best_accuracy < accuracy:
                    best_accuracy = accuracy
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

            # Generate batches
            batches = utils.batch_iter(list(zip(x_train, y_train)), FLAGS.BATCH_SIZE, FLAGS.NUM_EPOCHS)

            val_score = 0.0
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % FLAGS.EVALUATE_EVERY == 0:
                    print("\nEvaluation:")
                    # Randomly draw a validation batch
                    shuff_idx = np.random.permutation(np.arange(FLAGS.BATCH_SIZE))
                    # print("shuff_idx:", shuff_idx)

                    x_batch_val, y_batch_val = x_val[shuff_idx], y_val[shuff_idx]

                    # val_step(x_batch_val, y_batch_val, writer=dev_summary_writer)
                    feed_dict = {
                        cwi.input_x: x_batch_val,
                        cwi.input_y: y_batch_val,
                        cwi.dropout_keep_prob: 1.0
                    }
                    # print("input_x_val[0]", x_batch[0])
                    step, summaries, loss, accuracy, predictions, out = sess.run(
                        [global_step, dev_summary_op, cwi.loss, cwi.accuracy, cwi.predictions, cwi.output], feed_dict
                    )
                    # print(out)
                    # time_str = datetime.datetime.now().isoformat()
                    print(" step {}, loss {:g}, accuracy {:g}".format(step, loss, accuracy))
                    if dev_summary_writer:
                        dev_summary_writer.add_summary(summaries, step)

                    # Save checkpoint
                    # if best_accuracy < accuracy:
                    #     best_accuracy = accuracy
                    #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #     print("Saved model checkpoint to {}\n".format(path))

                    f1 = f1_score(np.argmax(y_batch_val, 1), predictions, average='macro')
                    # tf.reduce_mean(tf.cast(correct_predictions, tf.float32)
                    if val_score < f1 or val_score == 0:
                        val_score = f1
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

                    # mae = mean_absolute_error(np.argmax(y_batch_val, 1), predictions)
                    # if mae <= val_score or val_score == 0:
                    #     val_score = mae
                    #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #     print("Saved model checkpoint to {}\n".format(path))

                    # if val_score > loss or val_score == 0:
                    #     val_score = loss
                    #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    #     print("Saved model checkpoint to {}\n".format(path))
                    print("")
                # if current_step % FLAGS.num_checkpoints == 0:
                #     path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                #     print("Saved model checkpoint to {}\n".format(path))

    return timestamp


def trains(LIST_LEARNING_RATES, LIST_FILTER_SIZES, LIST_NUM_FILTERS, LIST_DROPOUT_KEEP_PROB, LIST_L2_REG_LAMBDA, LIST_BATCH_SIZE, LIST_NUM_EPOCHS):
    params = parse_args()
    # params.TRAIN_PATH = './data/CWI 2018 Training Set/english/All_Train.tsv'
    # params.DEV_PATH = './data/CWI 2018 Training Set/english/All_Dev.tsv'

    # params.TRAIN_PATH = './data/CWI 2018 Training Set/english/Wikipedia_Train.tsv'
    # params.DEV_PATH = './data/CWI 2018 Training Set/english/Wikipedia_Dev.tsv'

    for learning_rate in LIST_LEARNING_RATES:
        for filter_size in LIST_FILTER_SIZES:
            for num_filter in LIST_NUM_FILTERS:
                for dropout in LIST_DROPOUT_KEEP_PROB:
                    for l2_reg in LIST_L2_REG_LAMBDA:
                        for batch_size in LIST_BATCH_SIZE:
                            for epoch in LIST_NUM_EPOCHS:
                                params.NUM_EPOCHS = epoch
                                params.BATCH_SIZE = batch_size
                                params.DROPOUT_KEEP_PROB = dropout
                                params.L2_REG_LAMBDA = l2_reg
                                params.NUM_FILTERS = num_filter
                                params.FILTER_SIZES = filter_size
                                params.LEARNING_RATE = learning_rate

                                checkpoint_dir = train(params)
                                # Run evaluation on different datasets
                                evaluate(checkpoint_dir, './data/english/Wikipedia_Test.tsv', params.BATCH_SIZE)
                                evaluate(checkpoint_dir, './data/english/WikiNews_Test.tsv', params.BATCH_SIZE)
                                evaluate(checkpoint_dir, './data/english/News_Test.tsv', params.BATCH_SIZE)


def main(_):
    FLAGS = parse_args()
    train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
