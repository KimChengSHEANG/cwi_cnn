import argparse
import sys


class DefaultConfiguration:
    # TRAIN_PATH = "./data/CWI 2018 Training Set/english/News_Train.tsv"
    # DEV_PATH = "./data/CWI 2018 Training Set/english/News_Dev.tsv"
    TRAIN_PATH = './data/CWI 2018 Training Set/english/All_Train.tsv'
    DEV_PATH = './data/CWI 2018 Training Set/english/All_Dev.tsv'
    TEST_PATH = "./data/CWI 2018 Test Set/english/News_Test.tsv"
    EMBEDDING_PATH = "~/upf/11_resources/dataset/glove/glove.6B.300d.txt"
    DEV_SAMPLE_PERCENTAGE = 0.1
    MAX_SENTENCE_LENGTH = 600
    MAX_TARGET_WORDS = 8
    TEXT_EMBEDDING_DIM = 300
    FILTER_SIZES = "3,4,5"
    NUM_FILTERS = 128
    DROPOUT_KEEP_PROB = 0.75
    L2_REG_LAMBDA = 1e-5
    BATCH_SIZE = 128
    NUM_EPOCHS = 200
    DISPLAY_EVERY = 10
    EVALUATE_EVERY = 20
    NUM_CHECKPOINTS = 5
    LEARNING_RATE = 1e-3
    DECAY_RATE = 0.9
    CHECKPOINT_DIR = ""
    ALLOW_SOFT_PLACEMENT = True
    LOG_DEVICE_PLACEMENT = False
    GPU_ALLOW_GROWTH = True


def parse_args():
    """
    Parse input arguments
    """
    params = DefaultConfiguration()
    parser = argparse.ArgumentParser()

    # Data loading params
    parser.add_argument("--train_path", default=params.TRAIN_PATH,
                        type=str, help="Path of train data")
    parser.add_argument("--dev_path", default=params.TRAIN_PATH,
                        type=str, help="Path of dev data")
    parser.add_argument("--test_path",
                        default=params.TEST_PATH,
                        type=str, help="Path of test data")

    parser.add_argument("--dev_sample_percentage", default=params.DEV_SAMPLE_PERCENTAGE, type=float,
                        help="Percentage of the training data to use for validation")
    parser.add_argument("--max_sentence_length", default=params.MAX_SENTENCE_LENGTH, type=int, help="Max sentence length in data")
    parser.add_argument("--max_target_words", default=params.MAX_TARGET_WORDS, type=int, help="Maximum number of words in the target words.")
    # Model Hyper-parameters
    # Embeddings
    parser.add_argument("--embedding_path", default=params.EMBEDDING_PATH,
                        type=str, help="Path of pre-trained word embeddings (word2vec)")
    parser.add_argument("--text_embedding_dim", default=params.TEXT_EMBEDDING_DIM,
                        type=int, help="Dimensionality of word embedding (default: 300)")
    # CNN
    parser.add_argument("--filter_sizes", default=params.FILTER_SIZES,
                        type=str, help="Comma-separated filter sizes (Default: 3,4,5)")
    parser.add_argument("--num_filters", default=params.NUM_FILTERS,
                        type=int, help="Number of filters per filter size (Default: 128)")

    # Misc
    parser.add_argument("--dropout_keep_prob", default=params.DROPOUT_KEEP_PROB,
                        type=float, help="Dropout keep probability of output layer (default: 0.2)")
    parser.add_argument("--l2_reg_lambda", default=params.L2_REG_LAMBDA,
                        type=float, help="L2 regularization lambda (default: 1e-5)")

    # Training parameters
    parser.add_argument("--batch_size", default=params.BATCH_SIZE,
                        type=int, help="Batch Size (default: 20)")
    parser.add_argument("--num_epochs", default=params.NUM_EPOCHS,
                        type=int, help="Number of training epochs (Default: 100)")
    parser.add_argument("--display_every", default=params.DISPLAY_EVERY,
                        type=int, help="Number of iterations to display training information")
    parser.add_argument("--evaluate_every", default=params.EVALUATE_EVERY,
                        type=int, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument("--num_checkpoints", default=params.NUM_CHECKPOINTS,
                        type=int, help="Number of checkpoints to store (default: 50)")
    parser.add_argument("--learning_rate", default=params.LEARNING_RATE,
                        type=float, help="Which learning rate to start with (Default: 1.0)")
    parser.add_argument("--decay_rate", default=params.DECAY_RATE,
                        type=float, help="Decay rate for learning rate (Default: 0.9)")

    # Testing parameters
    parser.add_argument("--checkpoint_dir", default=params.CHECKPOINT_DIR,
                        type=str, help="Checkpoint directory from training run")

    # Misc Parameters
    parser.add_argument("--allow_soft_placement", default=params.ALLOW_SOFT_PLACEMENT,
                        type=bool, help="Allow device soft device placement")
    parser.add_argument("--log_device_placement", default=params.LOG_DEVICE_PLACEMENT,
                        type=bool, help="Log placement of ops on devices")
    parser.add_argument("--gpu_allow_growth", default=params.GPU_ALLOW_GROWTH,
                        type=bool, help="Allow gpu memory growth")

    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)

    print("")
    args = parser.parse_args()
    for arg in vars(args):
        print("{}={}".format(arg.upper(), getattr(args, arg)))
    print("")

    params.TRAIN_PATH = args.train_path
    params.DEV_PATH = args.dev_path
    params.TEST_PATH = args.test_path
    params.EMBEDDING_PATH = args.embedding_path
    params.DEV_SAMPLE_PERCENTAGE = args.dev_sample_percentage
    params.MAX_SENTENCE_LENGTH = args.max_sentence_length
    params.MAX_TARGET_WORDS = args.max_target_words
    params.TEXT_EMBEDDING_DIM = args.text_embedding_dim
    params.FILTER_SIZES = args.filter_sizes
    params.NUM_FILTERS = args.num_filters
    params.DROPOUT_KEEP_PROB = args.dropout_keep_prob
    params.L2_REG_LAMBDA = args.l2_reg_lambda
    params.BATCH_SIZE = args.batch_size
    params.NUM_EPOCHS = args.num_epochs
    params.DISPLAY_EVERY = args.display_every
    params.EVALUATE_EVERY = args.evaluate_every
    params.NUM_CHECKPOINTS = args.num_checkpoints
    params.LEARNING_RATE = args.learning_rate
    params.DECAY_RATE = args.decay_rate
    params.CHECKPOINT_DIR = args.checkpoint_dir
    params.ALLOW_SOFT_PLACEMENT = args.allow_soft_placement
    params.LOG_DEVICE_PLACEMENT = args.log_device_placement
    params.GPU_ALLOW_GROWTH = args.gpu_allow_growth

    return params


# FLAGS = parse_args()

if __name__ == '__main__':
    parse_args()
