import tensorflow as tf
import os
import numpy as np
import random
import base
from dataset import KBDataset
from models import TorusE, TransE
import argparse


def get_parameters(reproduce=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('-reproduce', default="toruse-wn18", type=str,
                        help="toruse-fb15k, toruse-wn18, toruse-fb15k-small, toruse-wn18-small")
    parser.add_argument('-restore', default=-1, type=int,
                        help="load pretrained embedding at epoch i, if not loading, i = -1")
    parser.add_argument('-gpu', default=0, type=int, help="GPU ID")
    parser.add_argument('-data', default="fb15k", type=str, help="fb15k, fb15k237, wn18, wn18rr")
    parser.add_argument('-model', default="toruse", type=str, help="TransE, TorusE")
    parser.add_argument('-emb_dim', default=10000, type=int, help="embedding dimension")
    parser.add_argument('-epoch', default=1000, type=int, help="number of epochs")
    parser.add_argument('-lr', default=0.0005, type=float, help="learning rate")
    parser.add_argument('-nbatches', default=100, type=int, help="number of batches")
    parser.add_argument('-reg', default="l1", type=str, help="regularization method {l1,l2,el2}")
    parser.add_argument('-margin', default=1., type=float, help="margin")
    parser.add_argument('-opt', default="SGD", type=str, help="Optimization method")
    parser.add_argument('-save_steps', default=100, type=int, help="save every k epochs")
    parser.add_argument('-valid_steps', default=100, type=int, help="validate every k epochs")
    parser.add_argument('-early_stopping', default=5, type=int,
                        help="early stopping after some validation steps")
    args = parser.parse_args()

    args.save_dir = "results/%s/%s.ckpt" % (args.data.lower(), args.model.lower())
    args.data = args.data.lower()
    args.model = args.model.lower()

    if reproduce is None:
        reproduce = args.reproduce.lower()

    if reproduce.lower() == "toruse-wn18":
        args.data = "wn18"
        args.model = "toruse"
        args.emb_dim = 10000
        args.lr = 0.0005
        args.epoch = 400
        args.nbatches = 100
        args.margin = 2000
        args.reg = "l1"
    elif reproduce.lower() == "toruse-fb15k":
        args.data = "fb15k"
        args.model = "toruse"
        args.emb_dim = 10000
        args.lr = 0.0005
        args.epoch = 1000
        args.nbatches = 100
        args.margin = 500
        args.reg = "el2"
    elif reproduce.lower() == "transe-wn18":
        args.data = "wn18"
        args.model = "transe"
        args.emb_dim = 200
        args.lr = 0.001
        args.epoch = 2000
        args.nbatches = 100
        args.margin = 0.1 * args.emb_dim
        args.reg = "l2"
    elif reproduce.lower() == "transe-fb15k":
        args.data = "fb15k"
        args.model = "transe"
        args.emb_dim = 200
        args.lr = 0.001
        args.epoch = 2000
        args.nbatches = 100
        args.margin = 0.1 * args.emb_dim
        args.reg = "l2"

    if int(args.restore) >= 0:
        # args.restore = args.save_dir
        restore_dir = "checkpoints/%s/%d/%s.ckpt" % (args.data, int(args.restore), args.model)
        args.restore = os.path.join(os.getcwd(), restore_dir)
    else:
        args.restore = ""

    args.save_steps = 100
    args.valid_steps = 100
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    return args


def main(_):
    # Init
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    np.random.seed(1)
    random.seed(a=1, version=2)

    # Load configuration
    config = get_parameters("toruse-wn18")
    print(config)

    # Load Dataset
    data = KBDataset(config.data)

    # tensorflow config
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)

    # Model Loading
    if config.model == "transe":
        model = TransE(config, data.nent, data.nrel)
    else:
        model = TorusE(config, data.nent, data.nrel)
    optimizer = tf.train.GradientDescentOptimizer(config.lr)
    # global_step = tf.Variable(0, name="gb", trainable=False)
    cal_gradient = optimizer.compute_gradients(model.loss)
    train_opt = optimizer.apply_gradients(cal_gradient)

    # Config Saver and Session
    saver = tf.train.Saver(max_to_keep=100)
    sess.run(tf.global_variables_initializer())

    # Training
    base.train(data, model, train_opt, config, sess, saver)

    # Testing
    base.test(data, model, sess)


if __name__ == "__main__":
    tf.app.run()
