import os
import argparse
import numpy as np
import tensorflow as tf

import data_helpers as utils
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error
from sklearn.metrics import classification_report

tf.logging.set_verbosity(tf.logging.ERROR)


def evaluate(checkpoint_dir, dataset, batch_size=128):

    print("======================================================================")

    file_name = os.path.basename(dataset).split('.')[0]
    test_file = './data/dumps/' + file_name + '.pk'

    print("Check point dir: ", checkpoint_dir)

    # prepare and load test data
    if not os.path.exists(test_file):
        print("Test dump not found. Preparing data....")
        utils.create_dump(dataset, test_file)

    print("Loading dataset from {} ...".format(test_file))
    x_test, y_test = utils.fetch(test_file)
    y_test = np.argmax(y_test, axis=1)

    # checkpoint directory from training run
    load_checkpoint_dir = "./runs/" + checkpoint_dir + "/checkpoints/"
    print("Loading graph from {}".format(load_checkpoint_dir))

    batch_size = int(batch_size)

    # Evaluation
    checkpoint_file = tf.train.latest_checkpoint(load_checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False
        )
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            # scores = graph.get_operation_by_name("output/scores").outputs[0]

            # Generate batches for one epoch
            batches = utils.batch_iter(list(x_test), batch_size, 1)

            # Collect the prediction scores here
            all_predictions = []

            for x_test_batch in batches:
                # x_test_batch = zip(*batch)
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                # print(batch_predictions)
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                # print(all_predictions)

    # Print accuracy if y_test is defined
    if y_test is not None:
        print("all_prediction.shape:", all_predictions.shape)
        print("y_test.shape:", y_test.shape)

        # correct_predictions = tf.equal(tf.cast(all_predictions, tf.int64), tf.argmax(y_test, 1))
        # accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
        # print("Total number of test examples: {}".format(len(y_test)))
        # print("Accuracy: {:g}".format(accuracy))
        # print("Accuracy: {:g}".format(correct_predictions / float(len(y_test))))

        # correct_predictions = float(sum(all_predictions == y_test))
        # print("Total number of test examples: {}".format(len(y_test)))
        # print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

        print(classification_report(y_test, all_predictions))
        # from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error
        print("Accuracy: {:.3f}".format(accuracy_score(y_test, all_predictions)))
        print("f1 score: {:.3f}".format(f1_score(y_test, all_predictions)))
        print("Precision: {:.3f}".format(precision_score(y_test, all_predictions)))
        print("Recall: {:.3f}".format(recall_score(y_test, all_predictions)))
        print("Mean absolute error: {:.3f}".format(mean_absolute_error(y_test, all_predictions)))

        # save evaluation results to a file
        fin = open("./runs/" + checkpoint_dir + "/configs.txt")
        configs = fin.readlines()
        fin.close()
        out_dir = "./reports/" + file_name
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        fout = open(out_dir + "/" + checkpoint_dir + "_report.txt", "w")
        fout.write(classification_report(y_test, all_predictions) + "\n")
        # from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_absolute_error
        fout.write("Accuracy: {:.3f}\n".format(accuracy_score(y_test, all_predictions)))
        fout.write("f1 score: {:.3f}\n".format(f1_score(y_test, all_predictions)))
        fout.write("Precision: {:.3f}\n".format(precision_score(y_test, all_predictions)))
        fout.write("Recall: {:.3f}\n".format(recall_score(y_test, all_predictions)))
        fout.write("Mean absolute error: {:.3f}\n".format(mean_absolute_error(y_test, all_predictions)))

        fout.write("\n==============================================\n")
        fout.write(checkpoint_dir + "\n")
        fout.writelines(configs)

        fout.close()

        print("Save report to a file. Completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckptdir', type=str, default=None, help='Model checkpoint directory')
    parser.add_argument('--dataset', type=str, default='News_Test', help='TSV test file name')
    parser.add_argument('--bsize', type=int, default=128, help='Test batch size')
    opt = parser.parse_args()
    evaluate(opt.ckptdir, opt.dataset, opt.bsize)

