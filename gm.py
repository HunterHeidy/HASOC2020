from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Activation, Input,  BatchNormalization, Concatenate,Add
from keras.layers import Convolution1D, Flatten, Dropout, MaxPool1D, GlobalAveragePooling1D
from keras.layers import Conv1D, Bidirectional, LSTM, GRU,Dot
from keras.engine.topology import Layer
from keras import regularizers
from keras import backend as K
import gensim
from on_lstm_keras import ONLSTM
import pandas as pd
import numpy as np
from lookhead import Lookahead
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
# from Layers.on_lstm_keras import ONLSTM
from nltk.tokenize import RegexpTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
from keras.engine.topology import InputSpec
import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.callbacks import Callback
from keras.optimizers import Adam, SGD


def load_data(train_file,test_file):
    clean_questions = pd.read_csv(train_file)
    clean_test = pd.read_csv(test_file)

    tokenizer = RegexpTokenizer(r'\w+')

    clean_questions["tokens"] = clean_questions['comment_text'].astype(str).apply(tokenizer.tokenize)
    clean_test["tokens"] = clean_test['comment_text'].astype(str).apply(tokenizer.tokenize)

    all_words = [word for tokens in clean_questions["tokens"] for word in tokens]
    for tokens in clean_questions["tokens"]:
        for word in tokens:
            all_words.append(word)
    print(all_words[-1])

    sentence_lengths = [len(tokens) for tokens in clean_questions["tokens"]]
    VOCAB = sorted(list(set(all_words)))
    print("%s words total,with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
    print("Max sentence length is %s" % max(sentence_lengths))
    max_sequence_length = max(sentence_lengths) + 1
    num_words = len(VOCAB)
    VALIDATION_SPLIT = .2

    tokenizer = Tokenizer(num_words)
    tokenizer.fit_on_texts(clean_questions['comment_text'].astype(str).tolist())
    tokenizer.fit_on_texts(clean_test['comment_text'].astype(str).tolist())

    sequences_train = tokenizer.texts_to_sequences(clean_questions['comment_text'].astype(str).tolist())
    sequences_test = tokenizer.texts_to_sequences(clean_test['comment_text'].astype(str).tolist())

    train_data = pad_sequences(sequences_train, maxlen=max_sequence_length)
    test_data = pad_sequences(sequences_test, maxlen=max_sequence_length)

    clean_questions['to_task1'] = [t1[w] for w in clean_questions['task_1']]
    clean_questions['to_task2'] = [t2[w] for w in clean_questions['task_2']]
    train_labelsA = to_categorical(np.array(clean_questions['to_task1']))
    train_labelsB = to_categorical(np.array(clean_questions['to_task2']), 4)
    from collections import Counter
    print('train_labelsB',Counter(clean_questions['to_task1']))
    print('train_labelsB',Counter(  clean_questions['to_task2'] ))



    clean_test['to_task1'] = [t1[w] for w in clean_test['task_1']]
    clean_test['to_task2'] = [t2[w] for w in clean_test['task_2']]
    teat_labelsA = to_categorical(np.array(clean_test['to_task1']))
    teat_labelsB = to_categorical(np.array(clean_test['to_task2']), 4)



    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    indices = range(train_data.shape[0])
    np.random.shuffle(list(indices))
    train_data = train_data[indices]
    train_labelsA = train_labelsA[indices]
    train_labelsB = train_labelsB[indices]
    # num_validation_samples = int(VALIDATION_SPLIT * cnn_data.shape[0])

    train_x,train_ya,train_yb=[],[],[]
    val_x,val_ya,val_yb=[],[],[]
    for i in range(len(train_data)):
        if i % split == 0:
            val_x.append(train_data[i])
            val_ya.append(train_labelsA[i])
            val_yb.append(train_labelsB[i])
        else:
            train_x.append(train_data[i])
            train_ya.append(train_labelsA[i])
            train_yb.append(train_labelsB[i])

    data = [[train_data,train_labelsA,train_labelsB], [test_data, teat_labelsA, teat_labelsB, clean_test['task_3'], clean_test['id']]]
    # data = [[train_x,train_ya,train_yb], [val_x,val_ya,val_yb], [test_data, teat_labelsA, teat_labelsA]]
    return data, word_index,num_words,max_sequence_length
class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())
def load_glove(word_index):
    EMBEDDING_FILE = 'glove.840B.300d.txt'

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')

    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf-8"))

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]
    count = 0

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index) + 1)
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            count += 1
    print("缺失值", count)
    return embedding_matrix
def load_fasttext(word_index):
    # EMBEDDING_FILE = '/home/bin_lab/桌面/n2c2-1/data/crawl-300d-2M.vec'
    # def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    # embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    EMBEDDING_FILE = 'cc.de.300.vec'
    def get_coefs(word,*arr): return word, np.asarray(arr, dtype='float32')
    #embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding='gb18030'))

    # EMBEDDING_FILE = '/media/bin_lab/C4F6073207B3A949/Linux/data/glove.840B.300d.txt'
    # def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE,encoding='utf-8') if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index)+1)
    count=0
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        else :count+=1
    print("queshizhi",count)

    return embedding_matrix

class OurLayer(Layer):
    """定义新的Layer，增加reuse方法，允许在定义Layer时调用现成的层
    """

    def reuse(self, layer, *args, **kwargs):
        if not layer.built:
            if len(args) > 0:
                inputs = args[0]
            else:
                inputs = kwargs['inputs']
            if isinstance(inputs, list):
                input_shape = [K.int_shape(x) for x in inputs]
            else:
                input_shape = K.int_shape(inputs)
            layer.build(input_shape)
        outputs = layer.call(*args, **kwargs)
        for w in layer.trainable_weights:
            if w not in self._trainable_weights:
                self._trainable_weights.append(w)
        for w in layer.non_trainable_weights:
            if w not in self._non_trainable_weights:
                self._non_trainable_weights.append(w)
        for u in layer.updates:
            if not hasattr(self, '_updates'):
                self._updates = []
            if u not in self._updates:
                self._updates.append(u)
        return outputs
def squash(x, axis=-1):
    # s_squared_norm is really small
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    scale = K.sqrt(s_squared_norm) / (0.5 + s_squared_norm)
    return scale * x
#    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
#    scale = K.sqrt(s_squared_norm + K.epsilon())
#    return x / scale
def extract_seq_patches(x, kernel_size, rate):
    """x.shape = [None, seq_len, seq_dim]
    滑动地把每个窗口的x取出来，为做局部attention作准备。
    """
    seq_dim = K.int_shape(x)[-1]
    seq_len = K.shape(x)[1]
    k_size = kernel_size + (rate - 1) * (kernel_size - 1)
    p_right = (k_size - 1) // 2
    p_left = k_size - 1 - p_right
    x = K.temporal_padding(x, (p_left, p_right))
    xs = [x[:, i: i + seq_len] for i in range(0, k_size, rate)]
    x = K.concatenate(xs, 2)
    return K.reshape(x, (-1, seq_len, kernel_size, seq_dim))
def to_mask(x, mask, mode='mul'):
    """通用mask函数
    这里的mask.shape=[batch_size, seq_len]或[batch_size, seq_len, 1]
    """
    if mask is None:
        return x
    else:
        for _ in range(K.ndim(x) - K.ndim(mask)):
            mask = K.expand_dims(mask, K.ndim(mask))
        if mode == 'mul':
            return x * mask
        else:
            return x - (1 - mask) * 1e10
class SparseSelfAttention(OurLayer):
    """稀疏多头自注意力机制
    来自文章《Generating Long Sequences with Sparse Transformers》
    说明：每个元素只跟相对距离为rate的倍数的元素、以及相对距离不超过rate的元素有关联。
    """

    def __init__(self, heads, size_per_head, rate=2,
                 key_size=None, mask_right=False, **kwargs):
        super(SparseSelfAttention, self).__init__(**kwargs)
        self.supports_masking = True
        self.heads = heads
        self.size_per_head = size_per_head
        self.out_dim = heads * size_per_head
        self.key_size = key_size if key_size else size_per_head
        assert rate != 1, u'if rate=1, please use SelfAttention directly'
        self.rate = rate
        self.neighbors = rate - 1
        self.mask_right = mask_right
        # self.routings = 9
        # self.activation = squash

    def build(self, input_shape):
        super(SparseSelfAttention, self).build(input_shape)
        self.q_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.k_dense = Dense(self.key_size * self.heads, use_bias=False)
        self.v_dense = Dense(self.out_dim, use_bias=False)
        # self.W = Dense(self.out_dim, use_bias=False)

    def call(self, inputs):
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        seq_dim = K.int_shape(x)[-1]
        # 补足长度，保证可以reshape
        seq_len = K.shape(x)[1]
        pad_len = self.rate - seq_len % self.rate
        x = K.temporal_padding(x, (0, pad_len))
        if x_mask is not None:
            x_mask = K.temporal_padding(x_mask, (0, pad_len))
        new_seq_len = K.shape(x)[1]
        x = K.reshape(x, (-1, new_seq_len, seq_dim))  # 经过padding后shape可能变为None，所以重新声明一下shape
        # 线性变换
        qw = self.reuse(self.q_dense, x)
        kw = self.reuse(self.k_dense, x)
        vw = self.reuse(self.v_dense, x)
        # 提取局部特征
        kernel_size = 1 + 2 * self.neighbors
        kwp = extract_seq_patches(kw, kernel_size, self.rate)  # shape=[None, seq_len, kernel_size, out_dim]
        vwp = extract_seq_patches(vw, kernel_size, self.rate)  # shape=[None, seq_len, kernel_size, out_dim]
        if x_mask is not None:
            xp_mask = extract_seq_patches(x_mask, kernel_size, self.rate)
        # 形状变换
        qw = K.reshape(qw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.key_size))
        kw = K.reshape(kw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.key_size))
        vw = K.reshape(vw, (-1, new_seq_len // self.rate, self.rate, self.heads, self.size_per_head))
        kwp = K.reshape(kwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.key_size))
        vwp = K.reshape(vwp, (-1, new_seq_len // self.rate, self.rate, kernel_size, self.heads, self.size_per_head))
        if x_mask is not None:
            x_mask = K.reshape(x_mask, (-1, new_seq_len // self.rate, self.rate, 1, 1))
            xp_mask = K.reshape(xp_mask, (-1, new_seq_len // self.rate, self.rate, kernel_size, 1, 1))
        # 维度置换
        qw = K.permute_dimensions(qw, (0, 3, 2, 1, 4))  # shape=[None, heads, r, seq_len // r, size]
        kw = K.permute_dimensions(kw, (0, 3, 2, 1, 4))
        vw = K.permute_dimensions(vw, (0, 3, 2, 1, 4))
        qwp = K.expand_dims(qw, 4)
        kwp = K.permute_dimensions(kwp,
                                   (0, 4, 2, 1, 3, 5))  # shape=[None, heads, r, seq_len // r, kernel_size, out_dim]
        vwp = K.permute_dimensions(vwp, (0, 4, 2, 1, 3, 5))
        if x_mask is not None:
            x_mask = K.permute_dimensions(x_mask, (0, 3, 2, 1, 4))
            xp_mask = K.permute_dimensions(xp_mask, (0, 4, 2, 1, 3, 5))
        # Attention1
        a = K.batch_dot(qw, kw, [4, 4]) / self.key_size ** 0.5
        a = K.permute_dimensions(a, (0, 1, 2, 4, 3))
        a = to_mask(a, x_mask, 'add')
        a = K.permute_dimensions(a, (0, 1, 2, 4, 3))
        if self.mask_right:
            ones = K.ones_like(a[: 1, : 1, : 1])
            mask = (ones - K.tf.matrix_band_part(ones, -1, 0)) * 1e10
            a = a - mask
        # Attention2
        ap = K.batch_dot(qwp, kwp, [5, 5]) / self.key_size ** 0.5
        ap = K.permute_dimensions(ap, (0, 1, 2, 3, 5, 4))
        if x_mask is not None:
            ap = to_mask(ap, xp_mask, 'add')
        ap = K.permute_dimensions(ap, (0, 1, 2, 3, 5, 4))
        if self.mask_right:
            mask = np.ones((1, kernel_size))
            mask[:, - self.neighbors:] = 0
            mask = (1 - K.constant(mask)) * 1e10
            for _ in range(4):
                mask = K.expand_dims(mask, 0)
            ap = ap - mask
        ap = ap[..., 0, :]
        # 合并两个Attention
        A = K.concatenate([a, ap], -1)
        A = K.softmax(A)
        a, ap = A[..., : K.shape(a)[-1]], A[..., K.shape(a)[-1]:]
        # 完成输出1
        o1 = K.batch_dot(a, vw, [4, 3])
        # 完成输出2
        ap = K.expand_dims(ap, -2)
        o2 = K.batch_dot(ap, vwp, [5, 4])
        o2 = o2[..., 0, :]
        # 完成输出
        o = o1 + o2
        o = to_mask(o, x_mask, 'mul')
        o = K.permute_dimensions(o, (0, 3, 2, 1, 4))
        o = K.reshape(o, (-1, new_seq_len, self.out_dim))
        o = o[:, : - pad_len]
        # return self.activation(o)

        return o
class KMaxPooling(Layer):
    """
    k-max-pooling
    """

    def __init__(self, k=1, **kwargs):
        super().__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.k = k
        self.supports_masking = True

    def compute_output_shape(self, input_shape):
        return (input_shape[0], (input_shape[2] * self.k))

    def call(self, inputs):
        # swap last two dimensions since top_k will be applied along the last dimension
        if isinstance(inputs, list):
            x, x_mask = inputs
        else:
            x, x_mask = inputs, None
        shifted_input = tf.transpose(x, [0, 2, 1])
        top_k = tf.nn.top_k(shifted_input, k=self.k, sorted=True, name=None)[0]
        return Flatten()(top_k)

def f1(y_true, y_pred):
    '''
    metric from here
    https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
    '''

    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])
class data_generator:
    def __init__(self, X, Y, batch_size=16):
        self.x = X
        self.y = Y
        self.batch_size = batch_size
        self.steps = len(self.x) // self.batch_size
        # print(len(self.data['sent']))
        if len(self.x) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = range(len(self.x))
            np.random.shuffle(list(idxs))
            sent, Y = [], []
            for i in idxs:
                # dataset = self.data
                s = self.x[i]
                y = self.y[i]

                sent.append(s)
                Y.append(y)
                if len(sent) == self.batch_size or i == idxs[-1]:
                    sent = seq_padding(sent)
                    Y = seq_padding(Y)
                    yield [sent], Y
                    [sent, Y] = [], []
from keras import regularizers

def SSAmodel(embed,num_words,max_sequence_length, num_class=2, trainable=False):
# def SSAmodel(num_words,max_sequence_length, num_class=2, trainable=False):
#     glove,fasttext
    sequence_input = Input(shape=(max_sequence_length,))
    sent = Embedding(input_dim=embed.shape[0], output_dim=embed.shape[1], weights=[embed], trainable=False,
                     name="sent")(sequence_input)
    # # print(embed.shape)
    # # mask_s = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(sequence_input)
    # sent = Embedding(input_dim=num_words, output_dim=300, trainable=trainable,
    #                  name="sent")(sequence_input)
    # sent = ONLSTM(300, 30, True, dropconnect=0.1)(sent)

    # sent = Bidirectional(LSTM(150, recurrent_dropout=0.25, activation='relu', return_sequences=True))(sent)
    output1 = SparseSelfAttention(10,30)(sent)
    output2 = SparseSelfAttention(10,30)(sent)
    output = Concatenate()([output1, output2])
    # output = Dropout(0.1)(output)



    # output = Capsule(num_capsule=self.sent_lenth, dim_capsule=200, routings=3)(output)
    output = KMaxPooling(3)(output)
    # output = Flatten()(output)
    output = Dense(units=128)(output)
    output = Activation("tanh")(output)

    output = Dense(units=num_class)(output)
    output = Activation("softmax")(output)
    model = Model(inputs=sequence_input, outputs=output)
    # sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  # optimizer='sgd',
                  metrics=['acc', f1])
    # lookhead= Lookahead(k=5,alpha=0.5)
    # lookhead.inject(model)
    # model.summary()
    return model
num_words = 0
embedding_dim = 300
max_sequence_length = 0
max_features = 3000000
epochs = 5
split = 5 ##da train,val
batch_size=16
NUM_SPLIT = 4
lr = 1e-4
t1 = {'NOT': 0, 'HOF': 1}
t_1 = {0: '4NOt', 1: 'HOF'}
t2 = {'HATE': 0, 'OFFN': 1, 'PRFN': 2, 'NONE': 3}
t_2 = {0: 'HATE', 1: 'OFFN', 2: 'PRFN', 3: 'NONE'}
train_file = "data/clean_hasoc_2020_de_train.csv"
test_file = "data/clean_german_test_1509.csv"

data, word_index,num_words,max_sequence_length = load_data(train_file, test_file)
# embedding_matrix = load_glove(word_index)
embedding_matrix = load_fasttext(word_index)
train_x, train_ya, train_yb = data[0][0], data[0][1], data[0][2]
test_x, test_ya , test_yb, ID, tweet_id = data[1][0], data[1][1], data[1][2], data[1][3], data[1][4]
# val_x, val_ya, val_yb = data[1][0], data[1][1], data[1][2]
# test_x, test_ya , test_yb, ID, tweet_id = data[2][0], data[2][1], data[2][2], data[2][3], data[2][4]

test_ya = np.argmax(test_ya, axis=-1)
test_yb = np.argmax(test_yb, axis=-1)
#callbacks
from keras.callbacks import ModelCheckpoint,EarlyStopping
DATA_SPLIT_SEED = 218
clr = CyclicLR(base_lr=0.001, max_lr=0.006,
               step_size=300., mode='exp_range',
               gamma=0.99994)
filepath = "emd/h5/modeltask1on0.1-30_{epoch:03d}-{val_loss:.4f}.h5" #避免文件名称重复
save_best_only = True #保存所有模型
patience = 20

# 保存训练过程中的最佳模型权重
cp = ModelCheckpoint(filepath,
                     monitor='val_loss',
                     verbose=2,
                     save_best_only=save_best_only,
                     mode='min')
es = EarlyStopping(monitor='val_loss',
                   # min_delta=1.0,
                   patience=patience,
                   verbose=2,
                   mode='min')
# save_weights_only=True
callbacks = [es,clr]
test_ameta=[]
test_bmeta=[]
for k in range(NUM_SPLIT):
    print('-'*40)
    print("Fold %d/%d" % (k+1, NUM_SPLIT))
    validationSize = int(len(train_x)/NUM_SPLIT)
    index1 = validationSize * k
    index2 = validationSize * (k+1)
    x_train = np.vstack((train_x[:index1], train_x[index2:]))
    x_val = train_x[index1:index2]


    ya_train = np.vstack((train_ya[:index1], train_ya[index2:]))
    y_val = train_ya[index1:index2]
    modela = SSAmodel(embedding_matrix,
                      num_words,
                      max_sequence_length,
                      num_class=2,
                      trainable=True)
    # modela = SSAmodel(num_words,
    #                   max_sequence_length,
    #                   num_class=2,
    #                   trainable=True)

    traina = data_generator(x_train, ya_train)
    vala = data_generator(x_val, y_val)
    modela.fit_generator(traina.__iter__(), steps_per_epoch=len(traina), epochs=epochs,
                         validation_data=vala.__iter__(),
                         validation_steps=len(vala),
                         callbacks=callbacks,
                         verbose=2)
    predictions_testa = modela.predict(test_x)
    test_ameta.append(predictions_testa)


    yb_train = np.vstack((train_yb[:index1], train_yb[index2:]))
    yb_val = train_yb[index1:index2]
    # modelb = SSAmodel(embedding_matrix, num_class=4, trainable=True)
    modelb = SSAmodel(embedding_matrix,
                      num_words,
                      max_sequence_length,
                      num_class=4,
                      trainable=True)
    # modelb = SSAmodel(
    #                   num_words,
    #                   max_sequence_length,
    #                   num_class=4,
    #                   trainable=True)
    # modelb.fit(x_train, yb_train,validation_data=(x_val, yb_val),epochs=epochs, batch_size=batch_size,verbose=2)

    trainb = data_generator(x_train, yb_train)
    valb = data_generator(x_val, yb_val)
    modelb.fit_generator(trainb.__iter__(), steps_per_epoch=len(trainb), epochs=epochs,
                         validation_data=valb.__iter__(),
                         validation_steps=len(valb),
                         callbacks=callbacks,
                         verbose=2)
    predictions_testb = modelb.predict(test_x)
    test_bmeta.append(predictions_testb)

def save_result(ID, tweet_id, test_data, task, file_name):
    if task == 'task1':
        for t in test_data:
            t = t_1[t]
    if task == 'task2':
        for t in test_data:
            t = t_2[t]
    result_df = pd.DataFrame({'tweet_id': tweet_id, task: test_data, 'ID': ID})
    result_df.to_csv(file_name, index=False)

# test_a = np.argmax(testa,axis=-1)
test_a = np.argmax(sum(test_ameta)/NUM_SPLIT, axis=-1)
save_result(ID,tweet_id,test_a,task='task1',file_name='submission_de_task1.csv')
p, r, f, _ = precision_recall_fscore_support(test_ya, test_a, average='macro')
print("Macro task1:precision_recall_fscore_support--precision:%.3f,recall:%.3f,f1_score:%.3f"%(p, r, f))

test_b = np.argmax(sum(test_bmeta)/NUM_SPLIT, axis=-1)
# test_b = modelb.predict(test_x)
# test_b = np.argmax(test_b, axis=-1)
save_result(ID, tweet_id, test_b, task='task2', file_name='submission_de_task2.csv')
p, r, f, _ = precision_recall_fscore_support(test_yb, test_b, average='macro')
print("Macro task2:precision_recall_fscore_support--precision:%.3f,recall:%.3f,f1_score:%.3f"%(p, r, f))


