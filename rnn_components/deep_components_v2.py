from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import xavier_initializer

import math

from utils.common import infer_shape
from utils.layer import linear

def get_init_method(stddev):
    # init_method_for_rnn=tf.random_normal_initializer(stddev=0.1)
    init_method_for_layers = tf.random_normal_initializer(stddev=stddev)
    init_method_for_rnn = tf.random_normal_initializer(stddev=stddev)
    # init_method_for_rnn=None
    # init_method_for_layers=tf.keras.initializers.he_normal()
    init_method_for_bias = tf.random_normal_initializer(stddev=stddev)
    return init_method_for_layers,init_method_for_rnn,init_method_for_bias


def residual_block(x,fx):
    return x+fx


def get_dropout_rnn_cell(rnn_units,cell_type,keep_prob,init_method_for_rnn):
    if cell_type=='gru':
        rnn_cell = rnn.GRUCell(rnn_units,kernel_initializer=init_method_for_rnn)
    elif cell_type=='lstm':
        rnn_cell = rnn.LSTMCell(rnn_units,initializer=init_method_for_rnn)
    else:
        raise ValueError('wrong rnn cell type:',cell_type)
    dropout_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
        rnn_cell, output_keep_prob=keep_prob)
    return dropout_rnn_cell

class multi_layer_encoder(object):
    def __init__(self,rnn_units,keep_prob,init_method_for_rnn,is_bidirectional=True,layer_num=3,cell_type='gru'):
        self.is_bidirectional=is_bidirectional
        self.rnn_units=rnn_units
        self.layer_num=layer_num
        self.keep_prob=keep_prob
        self.cell_type=cell_type
        fw_cells = [get_dropout_rnn_cell(self.rnn_units, self.cell_type, self.keep_prob,init_method_for_rnn) for i in
                    range(0, self.layer_num)]
        self.fw_multilayer_cell = rnn.MultiRNNCell(fw_cells, state_is_tuple=True)
        if self.is_bidirectional:
            bw_cells = [get_dropout_rnn_cell(self.rnn_units, self.cell_type, self.keep_prob,init_method_for_rnn) for i in
                        range(0, self.layer_num)]
            self.bw_multilayer_cell = rnn.MultiRNNCell(bw_cells, state_is_tuple=True)


    def __call__(self, seq_embedding, seq_len, init_state_fw=None, init_state_bw=None):
        if self.is_bidirectional:
            out, state = tf.nn.bidirectional_dynamic_rnn(self.fw_multilayer_cell, self.bw_multilayer_cell,
                                                         inputs=seq_embedding, sequence_length=seq_len,
                                                         initial_state_fw=init_state_fw,
                                                         initial_state_bw=init_state_bw,
                                                         dtype=tf.float32)
            if self.cell_type == 'gru':
                rnn_out = tf.concat(out, axis=2)
                rnn_state = [tf.concat([state[0][i], state[1][i]], axis=1) for i in range(0, self.layer_num)]
            elif self.cell_type == 'lstm':
                rnn_out = tf.concat(out, axis=2)
                rnn_state = [tf.concat([state[0][i][1], state[1][i][1]], axis=1) for i in range(0, self.layer_num)]
            else:
                raise ValueError('wrong rnn cell type:', self.cell_type)
            rnn_state = tf.concat(rnn_state, axis=1)
        else:
            out, state =tf.nn.dynamic_rnn(self.fw_multilayer_cell,seq_embedding,seq_len,
                                          dtype=tf.float32)
            if self.cell_type == 'gru':
                rnn_out = out
                rnn_state = [state[i] for i in range(0, self.layer_num)]
            elif self.cell_type == 'lstm':
                rnn_out = out
                rnn_state = [state[i][1] for i in range(0, self.layer_num)]
            else:
                raise ValueError('wrong rnn cell type:', self.cell_type)
            rnn_state = tf.concat(rnn_state,axis=1)
        #rnn_out=residual_block(seq_embedding,rnn_out)
        return rnn_out,rnn_state

class multi_layer_decoder(rnn.RNNCell):
    """
        seq2seq decoder with attention.
        """
    def __init__(self, memory, memory_len,
                 memory_size, keep_prob, rnn_units,vocab_size,embedding_size,
                 init_method_for_rnn, stddev,
                 layer_num=2,cell_type='gru',
                 reuse=None, name='multi_layer_decoder'):
        super(multi_layer_decoder, self).__init__(name=name, _reuse=reuse)
        self.rnn_units = rnn_units
        self.layer_num = layer_num
        self.keep_prob = keep_prob
        self.memory = memory
        self.memory_len = memory_len
        self.memory_size=memory_size
        self.vocab_size=vocab_size
        self.cell_type=cell_type
        self.embedding_size=embedding_size
        if cell_type=='gru':
            self.fw_rnn_cell = rnn.GRUCell(self.rnn_units,kernel_initializer=init_method_for_rnn)
            self.fw_rnn_cell=tf.nn.rnn_cell.DropoutWrapper(
                self.fw_rnn_cell, output_keep_prob=self.keep_prob)
            self.fw_multilayer_cell = rnn.MultiRNNCell([get_dropout_rnn_cell(self.rnn_units, self.cell_type, self.keep_prob, init_method_for_rnn) for i in
                        range(0, self.layer_num)], state_is_tuple=False)
        elif cell_type=='lstm':
            self.fw_rnn_cell=rnn.LSTMCell(self.rnn_units,initializer=init_method_for_rnn)
            self.fw_rnn_cell = tf.nn.rnn_cell.DropoutWrapper(
                self.fw_rnn_cell, output_keep_prob=self.keep_prob)
            self.fw_multilayer_cell = rnn.MultiRNNCell([get_dropout_rnn_cell(self.rnn_units, self.cell_type, self.keep_prob, init_method_for_rnn) for i in
                        range(0, self.layer_num)], state_is_tuple=False)
        else:
            raise ValueError('wrong rnn cell type:',cell_type)
        # params
        self.attn_W = tf.Variable(tf.random_normal(shape=(self.rnn_units*self.layer_num+embedding_size, memory_size.value)) * stddev)
        self.attn_b = tf.Variable(tf.zeros(shape=(self.memory_size.value,)))
        self.attn_combine_W = tf.Variable(
            tf.random_normal(shape=(self.embedding_size + memory_size.value, self.rnn_units)) * stddev)
        self.attn_combine_b = tf.Variable(tf.zeros(shape=(self.rnn_units,)))
        self.linear = tf.Variable(tf.random_normal(shape=(self.rnn_units, self.vocab_size)) * stddev)

    def attention(self, query, key, value, value_length, attn_w, attn_b, max_length=None):
        left = tf.tensordot(query, attn_w, axes=(1, 0),name='attn_tensordot')  # b*(rnn_units)
        left = tf.add(left, attn_b, name='attn_add_bias')
        attn_weights = tf.matmul(tf.expand_dims(left, axis=1), tf.transpose(key, perm=(0, 2, 1)),name='attn_cal_weight')  # b*1*len
        if max_length is None:
            max_length = tf.reduce_max(value_length)
        attn_mask = tf.sequence_mask(value_length, dtype=tf.float32, maxlen=max_length)  # b*max_len
        attn_weights = tf.multiply(attn_weights, tf.expand_dims(attn_mask, axis=1),name='attn_mask')  # b*1*l
        attn_weights = tf.nn.softmax(attn_weights,name='cal_prob_attn_weight')
        output = tf.matmul(attn_weights, value,name='apply_attention')  # b*1*memsize
        return output[:, 0, :], attn_weights

    def linear_func(self, x, w, b):
        """
        linear function, (x * W + b)
        :param x: x input
        :param w: W param
        :param b: b bias
        :return:
        """
        linear_out = tf.add(tf.matmul(x, w), b)
        return linear_out

    @property
    def state_size(self):
        return self.fw_multilayer_cell.state_size

    @property
    def output_size(self):
        return self.fw_multilayer_cell.output_size

    def __call__(self, inputs, state, memory=None, memory_len=None, scope=None, return_attn=False):
        """
        attention decoder using gru.
        :param inputs: word indices(batch, )
        :param encoder_outputs: encoder outputs(batch, max_length, hidden_size)
        :param state: final state
        :return:
        """
        if memory is None:
            attn_out, attn_weights = self.attention(query=tf.concat([state, inputs], axis=1), key=self.memory,
                                                    value=self.memory,
                                                    value_length=self.memory_len, attn_w=self.attn_W,
                                                    attn_b=self.attn_b)
        else:
            assert memory_len is not None,'memory len is None!'
            attn_out, attn_weights = self.attention(query=tf.concat([state, inputs], axis=1), key=memory,
                                                    value=memory,
                                                    value_length=memory_len, attn_w=self.attn_W,
                                                    attn_b=self.attn_b)
        #rnn_input = self.linear_func(tf.concat([inputs, attn_out], axis=1), self.attn_combine_W, self.attn_combine_b)
        rnn_input = tf.concat([inputs, attn_out], axis=1)
        next_out, next_state=self.fw_multilayer_cell(rnn_input,state)
        #next_out = tf.tensordot(next_out, self.linear, axes=[[1], [0]])
        #next_out=residual_block(inputs,next_out)
        #next_state=residual_block(inputs,next_state)
        if return_attn:
            return next_out, next_state, attn_weights
        else:
            return next_out, next_state


class bilstm_encoder(object):
    def __init__(self,embedding,encoder_size):
        self.embedding=embedding
        self.encoder_size=encoder_size
        self.fw_cell=rnn.LSTMCell(self.encoder_size)
        self.bw_cell=rnn.LSTMCell(self.encoder_size)

    def __call__(self,seq_index,seq_len,init_state_fw=None,init_state_bw=None):
        seq_embedding=tf.nn.embedding_lookup(self.embedding,seq_index)
        out,state=tf.nn.bidirectional_dynamic_rnn(self.fw_cell,self.bw_cell,seq_embedding,seq_len,dtype=tf.float32,
                                                  initial_state_fw=init_state_fw,initial_state_bw=init_state_bw)
        combined_out=tf.concat(out,axis=2)
        combined_state=tf.concat([state[0][1],state[1][1]],axis=1)
        return combined_out,combined_state


class simple_decoder(object):
    def __init__(self,embedding,rnn_units,vocab_size):
        self.embedding=embedding
        self.rnn_units=rnn_units
        self.linear=tf.get_variable(name='decoder_linear_projection',
                                    shape=(rnn_units,vocab_size))
        self.gru_cell=rnn.GRUCell(rnn_units)

    def __call__(self, inputs,encoder_out, state):
        inputs_embedding=tf.nn.embedding_lookup(self.embedding,inputs)
        output,state= tf.nn.dynamic_rnn(cell=self.gru_cell,
                                        inputs=tf.expand_dims(inputs_embedding,axis=1),
                                        initial_state=state,dtype=tf.float32)
        output = tf.tensordot(output,self.linear,axes=(2,0))
        return output,state

#Luong attention
def general_attention(query, key, value, value_length, attn_w, max_length=None):
    #hs*W*ht
    left = tf.tensordot(query, attn_w, axes=(1, 0), name='attn_tensordot')  # b*(rnn_units)
    attn_weights = tf.matmul(tf.expand_dims(left, axis=1), tf.transpose(key, perm=(0, 2, 1)),
                             name='attn_cal_weight')  # b*1*len
    if max_length is None:
        max_length = tf.reduce_max(value_length)
    attn_mask = tf.sequence_mask(value_length, dtype=tf.float32, maxlen=max_length)  # b*max_len
    attn_weights = tf.multiply(attn_weights, tf.expand_dims(attn_mask, axis=1), name='attn_mask')  # b*1*l
    attn_weights = tf.nn.softmax(attn_weights, name='cal_prob_attn_weight')
    output = tf.matmul(attn_weights, value, name='apply_attention')  # b*1*memsize
    return output[:, 0, :], attn_weights

def mlp_attention(query, key, value, value_length, attn_w, attn_v, max_length=None):
    #query:[batchsize,vec_len]
    #key:[batchsize,words,vec_len]
    #value=key
    #V*(W*[query;one key]).shape=[1]
    key_len=tf.shape(key)[2]
    stacked_query=tf.expand_dims(query,axis=1)
    stacked_query=tf.tile(stacked_query,[1,key_len,1])
    concated_q_k=tf.concat([stacked_query,key],axis=2)
    first=tf.tensordot(concated_q_k,attn_w,axes=[2,1])
    first=tf.tanh(first)
    second=tf.tensordot(first,attn_v,axes=[2,1])
    attn_weights = tf.transpose(second, perm=(0,2,1))
    if max_length is None:
        max_length = tf.reduce_max(value_length)
    attn_mask = tf.sequence_mask(value_length, dtype=tf.float32, maxlen=max_length)  # b*max_len
    attn_weights = tf.multiply(attn_weights, tf.expand_dims(attn_mask, axis=1), name='attn_mask')  # b*1*l
    attn_weights = tf.nn.softmax(attn_weights, name='cal_prob_attn_weight')
    output = tf.matmul(attn_weights, value, name='apply_attention')  # b*1*memsize
    return output[:, 0, :], attn_weights

"""
Attention Unit
"""

def add_timing_signal(x, min_timescale=1.0, max_timescale=1.0e4, name=None):
    """
    This function adds a bunch of sinusoids of different frequencies to a
    Tensor. See paper: `Attention is all you need'

    :param x: A tensor with shape [batch, length, channels]
    :param min_timescale: A floating point number
    :param max_timescale: A floating point number
    :param name: An optional string

    :returns: a Tensor the same shape as x.
    """
    with tf.name_scope(name, default_name="add_timing_signal", values=[x]):
        length = tf.shape(x)[1]
        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )

        scaled_time = (tf.expand_dims(position, 1) * tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        signal = tf.reshape(signal, [1, length, channels])

        return x + signal


def split_heads(x,
                num_heads,
                name=None):
    """ Split heads

    :param x: A tensor with shape [batch, length, channels]
    :param num_heads: An integer
    :param name: An optional string

    :returns: A tensor with shape [batch, heads, length, channels / heads]
    """
    with tf.name_scope(name, default_name="split_heads", values=[x]):
        x_shape = infer_shape(x)
        m = x_shape[-1]
        if isinstance(m, int) and isinstance(num_heads, int):
            assert m % num_heads == 0
        return tf.transpose(tf.reshape(x, x_shape[:-1] + [num_heads, m // num_heads]), [0, 2, 1, 3])


def combine_heads(x,
                  name=None):
    """ Combine heads

    :param x: A tensor with shape [batch, heads, length, channels]
    :param name: An optional string

    :returns: A tensor with shape [batch, length, heads * channels]
    """

    with tf.name_scope(name, default_name="combine_heads", values=[x]):
        x = tf.transpose(x, [0, 2, 1, 3])
        x_shape = infer_shape(x)
        a, b = x_shape[-2:]
        return tf.reshape(x, x_shape[:-2] + [a * b])


def compute_qkv(queries,
                memories,
                key_size,
                value_size,
                num_heads,
                state=None):
    """Computes query, key and value.

    :param queries: A tensor with shape [batch, length_q, depth_q]
    :param memories: A tensor with shape [batch, length_m, depth_m]
    :param state: design for incremental decoding

    :returns: (q, k, v): [batch, length, depth] tensors
    """
    next_state = {}

    if key_size % num_heads != 0:
        raise ValueError("Key size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (key_size, num_heads))

    if value_size % num_heads != 0:
        raise ValueError("Value size (%d) must be divisible by the number of "
                         "attention heads (%d)." % (value_size, num_heads))

    if memories is None:
        # self attention
        size = key_size * 2 + value_size
        combined = linear(queries, size, scope="qkv_transform")
        q, k, v = tf.split(combined, [key_size, key_size, value_size], axis=-1)

        if state is not None:
            k = tf.concat([state["key"], k], axis=1)
            v = tf.concat([state["value"], v], axis=1)
            next_state["key"] = k
            next_state["value"] = v
    else:
        q = linear(queries, key_size, scope="q_transform")
        combined = linear(memories, key_size + value_size, scope="kv_transform")
        k, v = tf.split(combined, [key_size, value_size], axis=-1)

    return q, k, v, next_state


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=None,
                          name=None):
    """dot-product attention.

    :param q: A tensor with shape [batch, heads, length_q, depth_k]
    :param k: A tensor with shape [batch, heads, length_kv, depth_k]
    :param v: A tensor with shape [batch, heads, length_kv, depth_v]
    :param bias: A tensor for ingoring unreasonable position
    :param dropout_rate: A floating point number
    :param name: An optional string

    :returns: A tensor with shape [batch, heads, length_q, depth_v]
    """

    with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")

        if dropout_rate is not None:
            weights = tf.nn.dropout(weights, 1 - dropout_rate)

        return tf.matmul(weights, v)


def fast_dot_product_attention(q,
                               k,
                               v,
                               bias,
                               dropout_rate=None,
                               name=None):
    """fast dot-product attention.
    deal with special case(the length of q is equal to 1)

    :param q: A tensor with shape [batch, heads, 1, depth_k]
    :param k: A tensor with shape [batch, heads, length_kv, depth_k]
    :param v: A tensor with shape [batch, heads, length_kv, depth_v]

    :returns: A tensor with shape [batch, heads, 1, depth_v]
    """

    with tf.variable_scope(name, default_name="dot_product_attention", values=[q, k, v]):
        # [batch, num_heads, query_length, memory_length]
        logits = tf.expand_dims(tf.reduce_sum(q * k, axis=3), axis=2)
        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")

        if dropout_rate is not None and dropout_rate > 0.0:
            weights = tf.nn.dropout(weights, 1 - dropout_rate)

        weights_shape = infer_shape(weights)
        new_shape = weights_shape[:-2]
        new_shape.append(weights_shape[-1])
        new_shape.append(1)
        weights = tf.reshape(weights, new_shape)
        return tf.expand_dims(tf.reduce_sum(weights * v, axis=2), axis=2)
        # return tf.matmul(weights, v)


def multihead_attention(queries,
                        memories,
                        bias,
                        num_heads,
                        key_size,
                        value_size,
                        output_size,
                        dropout_rate=None,
                        scope=None,
                        state=None):
    """ Multi-head scaled-dot-product attention with input/output
        transformations.

    :param queries: A tensor with shape [batch, length_q, depth_q]
    :param memories: A tensor with shape [batch, length_m, depth_m]
    :param bias: A tensor (see attention_bias)
    :param num_heads: An integer dividing key_size and value_size
    :param key_size: An integer
    :param value_size: An integer
    :param output_size: An integer
    :param dropout_rate: A floating point number in (0, 1]
    :param dtype: An optional instance of tf.DType
    :param scope: An optional string

    :returns: A dict with the following keys:
        weights: A tensor with shape [batch, heads, length_q, length_kv]
        outputs: A tensor with shape [batch, length_q, depth_v]
    """

    with tf.variable_scope(scope, default_name="multihead_attention", values=[queries, memories]):

        q, k, v, next_state = compute_qkv(queries, memories, key_size, value_size, num_heads, state=state)

        # split heads
        q = split_heads(q, num_heads)
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)

        # scale query
        key_depth_per_head = key_size // num_heads
        q *= key_depth_per_head ** -0.5

        # attention
        if state is not None:
            results = fast_dot_product_attention(q, k, v, bias, dropout_rate)
        else:
            results = dot_product_attention(q, k, v, bias, dropout_rate)

        # combine heads
        x = combine_heads(results)
        net_output = linear(x, output_size, scope="output_transform")

        outputs = {"outputs": net_output}
        if state is not None:
            outputs["state"] = next_state

        return outputs


def attention_bias(inputs, mode, inf=-1e9, name=None):
    """ A bias tensor used in attention mechanism
    :param inputs: A tensor
    :param mode: one of "causal", "masking", "proximal" or "distance"
    :param inf: A floating value
    :param name: optional string
    :returns: A 4D tensor with shape [batch, heads, queries, memories]
    """

    with tf.name_scope(name, default_name="attention_bias", values=[inputs]):
        if mode == "causal":
            length = inputs
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            ret = inf * (1.0 - lower_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        elif mode == "masking":
            mask = inputs
            ret = (1.0 - mask) * inf
            return tf.expand_dims(tf.expand_dims(ret, 1), 1)
        elif mode == "proximal":
            length = inputs
            r = tf.to_float(tf.range(length))
            diff = tf.expand_dims(r, 0) - tf.expand_dims(r, 1)
            m = tf.expand_dims(tf.expand_dims(-tf.log(1 + tf.abs(diff)), 0), 0)
            return m
        elif mode == "distance":
            length, distance = inputs
            distance = tf.where(distance > length, 0, distance)
            distance = tf.cast(distance, tf.int64)
            lower_triangle = tf.matrix_band_part(
                tf.ones([length, length]), -1, 0
            )
            mask_triangle = 1.0 - tf.matrix_band_part(
                tf.ones([length, length]), distance - 1, 0
            )
            ret = inf * (1.0 - lower_triangle + mask_triangle)
            return tf.reshape(ret, [1, 1, length, length])
        else:
            raise ValueError("Unknown mode %s" % mode)
