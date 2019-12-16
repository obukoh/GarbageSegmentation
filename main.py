import argparse
import random
import tensorflow as tf
from util import loader as ld
from util import model
from util import repoter as rp
from PIL import Image
import numpy as np


def load_dataset(train_rate):
    loader = ld.Loader(dir_original="dataset_unity/newBefore2",
                       dir_segmented="dataset_unity/newAfter2")
    return loader.load_train_test(train_rate=train_rate, shuffle=True)

def compute_class_accuracies(pred, label, num_classes):
    total = []
    for val in range(num_classes):
        print("label", label)
        print("num_classes", num_classes)
        print("val", val)

    count = [0.0] * num_classes
    for i in range(len(label)):
        if pred[i] == label[i]:
            count[int(pred[i])] = count[int(pred[i])] + 1.0

    # If there are no pixels from a certain class in the GT,
    # it returns NAN because of divide by zero
    # Replace the nans with a 1.0.
    accuracies = []
    for i in range(len(total)):
        if total[i] == 0:
            accuracies.append(1.0)
        else:
            accuracies.append(count[i] / total[i])

    return accuracies

def train(parser):
    # 訓練とテストデータを読み込みます
    # return train_set, test_set
    train, test = load_dataset(train_rate=parser.trainrate)
    valid = train.perm(0, 30) #decide on batch size of valid
    test = test.perm(0, 150) #decide on batch size of test

    # 結果保存用のインスタンスを作成します
    reporter = rp.Reporter(parser=parser)
    accuracy_fig = reporter.create_figure("Accuracy", ("epoch", "accuracy"), ["train", "test"])
    loss_fig = reporter.create_figure("Loss", ("epoch", "loss"), ["train", "test"])

    # GPUを使用するか
    gpu = parser.gpu

    # モデルの生成
    model_unet = model.UNet(l2_reg=parser.l2reg).model

    # 誤差関数とオプティマイザの設定をします
    # tf.reduce_mean : 平均を出している
    #https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits
    # Computes softmax cross entropy between logits and labels.
    # labels: Each vector along the class dimension should hold a valid probability distribution e.g. for the case in which labels are of shape [batch_size, num_classes], each row of labels[i] must be a valid probability distribution.
    # logits: Per-label activations, typically a linear output. These activation energies are interpreted as unnormalized log probabilities.
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=model_unet.teacher,
                                                                           logits=model_unet.outputs))

    # the GraphKeys class contains many standard names for collections.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 精度の算出をします
    # 尤もらしいClass、argmax(model_unet.outputs, 3)が教師のラベルと等しいか = true or false
    #teacher = tf.placeholder(tf.float32, [None(arg0), size[0](arg1), size[1](arg2), len(ld.DataSet.CATEGORY)(arg3)])
    correct_prediction = tf.equal(tf.argmax(model_unet.outputs, 3), tf.argmax(model_unet.teacher, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    all_labels_true = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
    mean_accuracy = tf.reduce_mean(all_labels_true)

    # セッションの初期化をします
    gpu_config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.7), device_count={'GPU': 1},
                                log_device_placement=False, allow_soft_placement=True)
    sess = tf.InteractiveSession(config=gpu_config) if gpu else tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # モデルの訓練
    epochs = parser.epoch
    batch_size = parser.batchsize
    is_augment = parser.augmentation
    train_dict = {model_unet.inputs: valid.images_original, model_unet.teacher: valid.images_segmented,
                  model_unet.is_training: False}
    test_dict = {model_unet.inputs: test.images_original, model_unet.teacher: test.images_segmented,
                 model_unet.is_training: False}

    for epoch in range(epochs):
        for batch in train(batch_size=batch_size, augment=is_augment):
            # バッチデータの展開
            inputs = batch.images_original
            teacher = batch.images_segmented
            # Training
            sess.run(train_step, feed_dict={model_unet.inputs: inputs, model_unet.teacher: teacher,
                                            model_unet.is_training: True})

        # 評価
        if epoch % 1 == 0:
            loss_train = sess.run(cross_entropy, feed_dict=train_dict)
            loss_test = sess.run(cross_entropy, feed_dict=test_dict)
            accuracy_train = sess.run(mean_accuracy, feed_dict=train_dict)
            accuracy_test = sess.run(mean_accuracy, feed_dict=test_dict)
            print("Epoch:", epoch)
            print("[Train] Loss:", loss_train, " Accuracy:", accuracy_train)
            print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)
            accuracy_fig.add([accuracy_train, accuracy_test], is_update=True)
            loss_fig.add([loss_train, loss_test], is_update=True)
            if epoch % 3 == 0:
                idx_train = random.randrange(10)
                idx_test = random.randrange(100)
                outputs_train = sess.run(model_unet.outputs,
                                         feed_dict={model_unet.inputs: [train.images_original[idx_train]],
                                                    model_unet.is_training: False})
                outputs_test = sess.run(model_unet.outputs,
                                        feed_dict={model_unet.inputs: [test.images_original[idx_test]],
                                                   model_unet.is_training: False})
                train_set = [train.images_original[idx_train], outputs_train[0], train.images_segmented[idx_train]]
                test_set = [test.images_original[idx_test], outputs_test[0], test.images_segmented[idx_test]]
                # reporter.save_image_from_ndarray(train_set, test_set, train.palette, epoch,
                #                               index_void=len(ld.DataSet.CATEGORY)-1)
                reporter.save_image_from_ndarray(train_set, test_set, train.palette, epoch, index_void=0)

    # 訓練済みモデルの評価
    loss_test = sess.run(cross_entropy, feed_dict=test_dict)
    accuracy_test = sess.run(accuracy, feed_dict=test_dict)
    print("Result")
    print("[Test]  Loss:", loss_test, "Accuracy:", accuracy_test)

    sess.close()


def get_parser():
    parser = argparse.ArgumentParser(
        prog='Image segmentation using U-Net',
        usage='python main.py',
        description='This module demonstrates image segmentation using U-Net.',
        add_help=True
    )

    parser.add_argument('-g', '--gpu', action='store_true', help='Using GPUs')
    parser.add_argument('-e', '--epoch', type=int, default=200, help='Number of epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=32, help='Batch size')
    parser.add_argument('-t', '--trainrate', type=float, default=0.85, help='Training rate')
    #parser.add_argument('-t', '--trainrate', type=float, default=0.01, help='Training rate')
    parser.add_argument('-a', '--augmentation', action='store_true', help='Number of epochs')
    parser.add_argument('-r', '--l2reg', type=float, default=0.001, help='L2 regularization')

    return parser


if __name__ == '__main__':
    parser = get_parser().parse_args()
    train(parser)
