import tensorflow as tf
from train import *
from evaluate import evaluate
import os
import glob

LIST_LEARNING_RATES = [1e-3, 1e-2]
LIST_FILTER_SIZES = ["3,4,5", "2,3,4"]
LIST_NUM_FILTERS = [64, 128, 256]
LIST_DROPOUT_KEEP_PROB = [0.5, 0.75, 0.9]

# LIST_L2_REG_LAMBDA = [1e-4, 1e-5, 1e-6]
# LIST_BATCH_SIZE = [128, 256]
# LIST_NUM_EPOCHS = [100, 200]


training_body = """
import tensorflow as tf
from train import *

LIST_LEARNING_RATES = [<LEARNING_RATE>]
LIST_FILTER_SIZES = ["<FILTER_SIZE>"]
LIST_NUM_FILTERS = [<NUM_FILTER>]
LIST_DROPOUT_KEEP_PROB = [<DROPOUT>]

LIST_L2_REG_LAMBDA = [1e-3, 1e-4, 1e-5]
LIST_BATCH_SIZE = [64, 128, 256]
LIST_NUM_EPOCHS = [100, 200, 300]

def main(_):
    trains(LIST_LEARNING_RATES, LIST_FILTER_SIZES, LIST_NUM_FILTERS, LIST_DROPOUT_KEEP_PROB, LIST_L2_REG_LAMBDA, LIST_BATCH_SIZE, LIST_NUM_EPOCHS)


if __name__ == '__main__':
    tf.app.run()

"""

slurm_body = """#!/bin/bash
#SBATCH -J CWI_NN
#SBATCH -t 0-8:00
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --ntasks 2
#SBATCH --threads-per-core 1
#SBATCH --mem=8G
#SBATCH -o %N.%J.out # STDOUT
#SBATCH -e %N.%J.err # STDERR
#SBATCH --partition=medium
##SBATCH -C intel

module load Python/3.6.4-foss-2017a
module load zlib/1.2.9-foss-2017a
module load Tensorflow/1.12.0-foss-2017a-Python-3.6.4
module load numpy/1.14.0-foss-2017a-Python-3.6.4
module load NLTK/3.3.0-foss-2017a-Python-3.6.4
module load scikit-learn/0.19.1-foss-2017a-Python-3.6.4


cd $PWD
python <TRAINING_FILE>
"""


if __name__ == '__main__':
    # delete old files
    for f in glob.glob("slurm/*.*"):
        os.remove(f)
        # print("Delete : " + f)
    if not os.path.exists("slurm"):
        os.mkdir("slurm")
    # generate slurm files
    index = 0
    for learning_rate in LIST_LEARNING_RATES:
        for filter_size in LIST_FILTER_SIZES:
            for num_filter in LIST_NUM_FILTERS:
                for dropout in LIST_DROPOUT_KEEP_PROB:
                    code = training_body
                    code = code.replace("<LEARNING_RATE>", str(learning_rate))
                    code = code.replace("<FILTER_SIZE>", filter_size)
                    code = code.replace("<NUM_FILTER>", str(num_filter))
                    code = code.replace("<DROPOUT>", str(dropout))

                    train_file = "slurm/train." + str(index + 1) + ".py"
                    ftrain = open(train_file, 'w')
                    ftrain.write(code)
                    ftrain.close()

                    slurm_content = slurm_body
                    slurm_content = slurm_body.replace("<TRAINING_FILE>", train_file)
                    slurm_file = "slurm/slurm." + str(index + 1) + ".txt"
                    fslurm = open(slurm_file, 'w')
                    fslurm.write(slurm_content)
                    fslurm.close()

                    index += 1

    fscript = open("slurm/submit_slurm_jobs.sh", 'w')
    fscript.write("#!/usr/bin/env bash \n")

    for i in range(0, index):
        fscript.write("sbatch slurm/slurm." + str(i + 1) + ".txt\n")

    fscript.close()
