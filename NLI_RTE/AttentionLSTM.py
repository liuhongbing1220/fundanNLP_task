# from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility
import os
import keras as K
from keras.regularizers import l2
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import *
# from visualizer import *
from keras.models import *
from keras.optimizers import *
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score
from keras.layers.core import Lambda,Dropout,RepeatVector,Flatten,Permute
from keras.layers import Input, Embedding, LSTM, Dense, TimeDistributed,Concatenate,Activation,Multiply,Add,Reshape
import argparse
# from keras.utils.visualize_util import plot  # THIS IS BAD
# from data_reader import *
from reader import load_data,get_vocab
from my_utils import map_to_txt
import logging
from datetime import datetime


# from myconfig import DATAPATH,MYPATH

def get_params():
    
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lstm', action="store", default=150, dest="lstm_units", type=int)
    parser.add_argument('-epochs', action="store", default=20, dest="epochs", type=int)
    parser.add_argument('-batch', action="store", default=32, dest="batch_size", type=int)
    parser.add_argument('-emb', action="store", default=100, dest="emb", type=int)
    parser.add_argument('-xmaxlen', action="store", default=20, dest="xmaxlen", type=int)
    parser.add_argument('-ymaxlen', action="store", default=20, dest="ymaxlen", type=int)
    parser.add_argument('-maxfeat', action="store", default=43444, dest="max_features", type=int)
    parser.add_argument('-classes', action="store", default=351, dest="num_classes", type=int)
    parser.add_argument('-sample', action="store", default=1, dest="samples", type=int)
    parser.add_argument('-nopad', action="store", default=False, dest="no_padding", type=bool)
    parser.add_argument('-lr', action="store", default=0.001, dest="lr", type=float)
    parser.add_argument('-load', action="store", default=False, dest="load_save", type=bool)
    parser.add_argument('-verbose', action="store", default=False, dest="verbose", type=bool)
    parser.add_argument('-train', action="store", default="train_all.txt", dest="train")
    parser.add_argument('-test', action="store", default="test_all.txt", dest="test")
    parser.add_argument('-dev', action="store", default="dev.txt", dest="dev")
    opts = parser.parse_args(sys.argv[1:])
    print ("lstm_units", opts.lstm_units)
    print ("epochs", opts.epochs)
    print ("batch_size", opts.batch_size)
    print ("emb", opts.emb)
    print ("samples", opts.samples)
    print ("xmaxlen", opts.xmaxlen)
    print ("ymaxlen", opts.ymaxlen)
    print ("max_features", opts.max_features)
    print ("no_padding", opts.no_padding)
    return opts


class AccCallBack(Callback):
    def __init__(self, xtrain, ytrain, xdev, ydev, xtest, ytest, vocab, opts):
        self.xtrain = xtrain
        self.ytrain = ytrain
        self.xdev = xdev
        self.ydev = ydev
        self.xtest = xtest
        self.ytest = ytest
        self.vocab = vocab
        self.opts = opts

    def on_epoch_end(self, epoch, logs={}):
        train_acc = compute_acc(self.xtrain, self.ytrain, self.vocab, self.model, self.opts)
        dev_acc = compute_acc(self.xdev, self.ydev, self.vocab, self.model, self.opts)
        test_acc = compute_acc(self.xtest, self.ytest, self.vocab, self.model, self.opts)
        logging.info('----------------------------------')
        logging.info('Epoch ' + str(epoch) + ' train loss:' + str(logs.get('loss')) + ' - Validation loss: ' + str(
            logs.get('val_loss')) + ' train acc: ' + str(train_acc) + ' dev acc: ' + str(
            dev_acc) +  'test acc: ' + str(test_acc) )
        logging.info('----------------------------------')


def get_H_n(X):
    ans = X[:, -1, :]  # get last element from time dim
    return ans


def get_Y(X, xmaxlen):
    return X[:, :xmaxlen, :]  # get first xmaxlen elem from time dim


def get_R(X):
    Y, alpha = X[0], X[1]
    ans = K.T.batched_dot(Y, alpha)
    return ans


def get_sum(X):
    return K.sum(X, axis=2)

def build_model(opts, verbose=False):
    k = 2 * opts.lstm_units  # 300
    L = opts.xmaxlen  # 20
    N = opts.xmaxlen + opts.ymaxlen + 1  # H P sentence length 
    
    print ("x len", L, "total len", N)
    print ("k", k, "L", L)

    main_input = Input(shape=(N,), dtype='int32', name='main_input')
    x = Embedding(output_dim=opts.emb, input_dim=opts.max_features, input_length=N, name='x')(main_input)
    drop_out = Dropout(0.1, name='dropout')(x)
    
    lstm_fwd = LSTM(opts.lstm_units, return_sequences=True, name='lstm_fwd')(drop_out)
    lstm_bwd = LSTM(opts.lstm_units, return_sequences=True, go_backwards=True, name='lstm_bwd')(drop_out)
    print(lstm_fwd)
    print(lstm_bwd)
    bilstm = Concatenate()([lstm_fwd, lstm_bwd])
    drop_out = Dropout(0.1)(bilstm)
    
    """
    The premise LSTM produce output matrix [h1---hk]--k X L
    """
    Y = Lambda(get_Y, arguments={"xmaxlen": L}, name="Y", output_shape=(L, k))(drop_out)
    
    h_n = Lambda(get_H_n, output_shape=(k,), name="h_n")(drop_out)
    Whn = Dense(k, W_regularizer=l2(0.01), name="Wh_n")(h_n)
    Whn_x_e = RepeatVector(L, name="Wh_n_x_e")(Whn)
    
    WY = TimeDistributed(Dense(k, W_regularizer=l2(0.01)), name="WY")(Y)
    merged = Concatenate()([Whn_x_e, WY])
    M = Activation('tanh', name="M")(merged)
    
    """
    对M矩阵-- L X K 进行Dense连接后变成 L维向量 
    """
    alpha_ = TimeDistributed(Dense(1, activation='linear'), name="alpha_")(M)
    flat_alpha = Flatten(name="flat_alpha")(alpha_)
    alpha = Dense(L, activation='softmax', name="alpha")(flat_alpha)

    Y_trans = Permute((2, 1), name="y_trans")(Y)  # of shape (None,300,20)
    print("Y:", Y)
    print("Y_trans:", Y_trans)
    print("alpha:", alpha)

#    r_ = merge([Y_trans, alpha], output_shape=(k, 1), name="r_", mode=get_R)z
    r_ = Lambda(get_sum, output_shape=(k, 1), name='r_')(Multiply()([Y_trans, alpha]))
    print('r_: ',r_)
    r = Reshape((k,), name="r")(r_)
    
    Wr = Dense(k, W_regularizer=l2(0.01))(r)
    Wh = Dense(k, W_regularizer=l2(0.01))(h_n)
    merged = Add()([Wr, Wh])
    
    h_star = Activation('tanh')(merged)
    out = Dense(3, activation='softmax')(h_star)
    output = out
    
    model = Model(input=[main_input], output=output)
    if verbose:
        model.summary()
    # plot(model, 'model.png')
    # # model.compile(loss={'output':'binary_crossentropy'}, optimizer=Adam())
    # model.compile(loss={'output':'categorical_crossentropy'}, optimizer=Adam(options.lr))
    model.compile(loss='categorical_crossentropy',optimizer=Adam(options.lr))
    return model


def compute_acc(X, Y, vocab, model, opts):
    scores = model.predict(X, batch_size=options.batch_size)
    prediction = np.zeros(scores.shape)
    for i in range(scores.shape[0]):
        l = np.argmax(scores[i])
        prediction[i][l] = 1.0
    assert np.array_equal(np.ones(prediction.shape[0]), np.sum(prediction, axis=1))
    plabels = np.argmax(prediction, axis=1)
    tlabels = np.argmax(Y, axis=1)
    acc = accuracy_score(tlabels, plabels)
    
    return acc


def getConfig(opts):
    conf = [opts.xmaxlen,
            opts.ymaxlen,
            opts.batch_size,
            opts.emb,
            opts.lr,
            opts.samples,
            opts.lstm_units,
            opts.epochs]
    if opts.no_padding:
        conf.append("no-pad")
    return "_".join(map(lambda x: str(x), conf))


def save_model(model, wtpath, archpath, mode='yaml'):
    if mode == 'yaml':
        yaml_string = model.to_yaml()
        open(archpath, 'w').write(yaml_string)
    else:
        with open(archpath, 'w') as f:
            f.write(model.to_json())
    model.save_weights(wtpath)


def load_model(wtpath, archpath, mode='yaml'):
    if mode == 'yaml':
        model = model_from_yaml(open(archpath).read())  # ,custom_objects={"MyEmbedding": MyEmbedding})
    else:
        with open(archpath) as f:
            model = model_from_json(f.read())  # , custom_objects={"MyEmbedding": MyEmbedding})
    model.load_weights(wtpath)
    return model


def concat_in_out(X, Y, vocab):
    numex = X.shape[0]  # num examples
    glue = vocab["delimiter"] * np.ones(numex).reshape(numex, 1)
    inp_train = np.concatenate((X, glue, Y), axis=1)
    return inp_train


def setup_logger(config_str):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=datetime.now().strftime('mylogfile_%H_%M_%d_%m_%Y.log'),
                        filemode='w')


if __name__ == "__main__":
    
#    tf.reset_default_graph()
    options = get_params()
    root="/Users/liuhongbing/Documents/tensorflow/data/snli_1.0/"
    train=[l.strip().split('\t') for l in open(root+'snli_1.0_train.txt')]
    dev=[l.strip().split('\t') for l in open(root+'snli_1.0_dev.txt')]
    test=[l.strip().split('\t') for l in open(root+'snli_1.0_test.txt')]
    vocab = get_vocab(train)
    print("vocab (incr. maxfeatures accordingly):",len(vocab))
    
    X_train,Y_train,Z_train=load_data(train,vocab)
    X_dev,Y_dev,Z_dev=load_data(dev,vocab)
    X_test,Y_test,Z_test=load_data(test,vocab)
    print('Build model...')
    
    model = build_model(options)

    config_str = getConfig(options)
    MODEL_ARCH = root+"/Attention_neural/arch_att" + config_str + ".yaml"
    MODEL_WGHT = root+"/Attention_neural/weights_att" + config_str + ".weights"

    MAXLEN = options.xmaxlen
    X_train = pad_sequences(X_train, maxlen=MAXLEN, value=vocab["unk"], padding='pre')
    X_dev = pad_sequences(X_dev, maxlen=MAXLEN, value=vocab["unk"], padding='pre')
    X_test = pad_sequences(X_test, maxlen=MAXLEN, value=vocab["unk"], padding='pre')
    
    Y_train = pad_sequences(Y_train, maxlen=MAXLEN, value=vocab["unk"], padding='post')
    Y_dev = pad_sequences(Y_dev, maxlen=MAXLEN, value=vocab["unk"], padding='post')
    Y_test = pad_sequences(Y_test, maxlen=MAXLEN, value=vocab["unk"], padding='post')

    net_train = concat_in_out(X_train, Y_train, vocab)
    net_dev = concat_in_out(X_dev, Y_dev, vocab)
    net_test = concat_in_out(X_test, Y_test, vocab)

    Z_train = to_categorical(Z_train, num_classes=3)
    Z_dev = to_categorical(Z_dev, num_classes=3)
    Z_test = to_categorical(Z_test, num_classes=3)

    print(X_train.shape, Y_train.shape, net_train.shape)
    print(map_to_txt(net_train[0], vocab), Z_train[0])
    print(map_to_txt(net_train[1], vocab), Z_train[1])
    setup_logger(config_str)

    assert net_train[0][options.xmaxlen] == 1
    
    train_dict = {'input': net_train, 'output': Z_train}
    dev_dict = {'input': net_dev, 'output': Z_dev}
    print('Build model...')
    model = build_model(options)

    logging.info(vars(options))
    logging.info(
        "train size: " + str(len(net_train)) + " dev size: " + str(len(net_dev)) + " test size: " + str(len(net_test)))
    
    if options.load_save and os.path.exists(MODEL_ARCH) and os.path.exists(MODEL_WGHT):
        print("Loading pre-trained model from", MODEL_WGHT)
        load_model(MODEL_WGHT, MODEL_ARCH, 'json')
        train_acc = compute_acc(net_train, Z_train, vocab, model, options)
        dev_acc = compute_acc(net_dev, Z_dev, vocab, model, options)
        test_acc = compute_acc(net_test, Z_test, vocab, model, options)
        print(train_acc, dev_acc, test_acc)
    else:
        history = model.fit(net_train,Z_train,
                            batch_size=options.batch_size,
                            nb_epoch=options.epochs,
                            validation_data=(net_dev,Z_dev),
                            callbacks=[
                                AccCallBack(net_train, Z_train, net_dev, Z_dev, net_test, Z_test, vocab, options)]
                            )
        save_model(model, MODEL_WGHT, MODEL_ARCH)