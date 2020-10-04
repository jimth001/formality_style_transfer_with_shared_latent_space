# coding=utf-8

import tensorflow as tf
import nltk
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.contrib.seq2seq import BasicDecoder, sequence_loss
from tensorflow.contrib.seq2seq import TrainingHelper, dynamic_decode
import random
from nltk.translate.bleu_score import corpus_bleu
from rnn_components import beamsearch, deep_components_v2 as deep_components
#from reinforcement_leaning_components import rewards_generator
from utils.embedding_api import load_word_embedding
from sls_settings_v2_FR import *

class seq2seq:
    def __init__(self, vocab_size, sos_token, eos_token, unk_token, struct_para, vocab_hash):
        self.vocab_size = vocab_size
        self.lr = struct_para.learning_rate
        self.hidden_size = struct_para.hidden_size
        self.max_length = struct_para.max_length
        self.layers_num = struct_para.layers_num
        self.embedding_size = struct_para.embedding_size
        self.SOS_token = sos_token
        self.EOS_token = eos_token
        self.UNK_token = unk_token
        self.beam_size = struct_para.beam_size
        self.keep_prob_value = struct_para.keep_prob
        self.lr_decay_rate = struct_para.lr_decay_rate
        self.denoising_swap_rate = struct_para.denoising_swap_rate
        self.max_swap_times = struct_para.max_swap_times
        self.max_lm_masked_num = struct_para.max_lm_masked_num
        self.max_lm_masked_rate = struct_para.max_lm_masked_rate
        #self.weight_generator=rewards_generator()
        self.graph=tf.Graph()
        self.vocab_hash=vocab_hash
        self.vocab_hash['<SOS>']=len(self.vocab_hash)
        self.vocab_hash['<UNK>'] = len(self.vocab_hash)
        self.vocab_hash['<EOS>'] = len(self.vocab_hash)
        self.id2words={}
        init_method_for_layers, init_method_for_rnn, init_method_for_bias=deep_components.get_init_method(struct_para.init_stddev)
        for key in self.vocab_hash.keys():
            assert self.vocab_hash[key] not in self.id2words
            self.id2words[self.vocab_hash[key]]=key
        with self.graph.as_default():
            with tf.variable_scope('placeholder_and_embedding'):
                self.src = tf.placeholder(name='src', shape=(None, None), dtype=tf.int32)
                self.src_len = tf.placeholder(name='src_len', shape=(None,), dtype=tf.int32)
                self.tgt_informal_input = tf.placeholder(name='tgt_informal_input', shape=(None, None),
                                                         dtype=tf.int32)
                self.tgt_informal_target = tf.placeholder(name='tgt_informal_target', shape=(None, None),
                                                          dtype=tf.int32)
                self.tgt_informal_len = tf.placeholder(name='tgt_informal_len', shape=(None,), dtype=tf.int32)
                self.tgt_formal_input = tf.placeholder(name='tgt_formal_input', shape=(None, None),
                                                       dtype=tf.int32)
                self.tgt_formal_target = tf.placeholder(name='tgt_formal_target', shape=(None, None),
                                                        dtype=tf.int32)
                self.tgt_formal_len = tf.placeholder(name='tgt_formal_len', shape=(None,), dtype=tf.int32)
                self.random_selected_sen = tf.placeholder(name='random_selected_sen', shape=(None, None,),
                                                          dtype=tf.int32)
                self.random_selected_sen_len = tf.placeholder(name='random_selected_sen_len', shape=(None,),
                                                              dtype=tf.int32)
                self.mch1 = tf.placeholder(name='mch1', shape=(None, None,), dtype=tf.int32)
                self.mch1_len = tf.placeholder(name='mch1_len', shape=(None,), dtype=tf.int32)
                self.mch2 = tf.placeholder(name='mch2', shape=(None, None,), dtype=tf.int32)
                self.mch2_len = tf.placeholder(name='mch2_len', shape=(None,), dtype=tf.int32)
                self.batch_size = tf.placeholder(name='batch_size', shape=(), dtype=tf.int32)
                self.keep_prob = tf.placeholder(name='keep_prob', shape=(), dtype=tf.float32)
                self.match_label = tf.placeholder(name='match_label', shape=(None,), dtype=tf.int32)
                self.embedding_source_pl = tf.placeholder(name='embedding_pl', dtype=tf.float32,
                                                          shape=(self.vocab_size, struct_para.embedding_size))
                self.sample_weights = tf.placeholder(name='sample_weights', dtype=tf.float32, shape=(None,))
                if not struct_para.share_embedding:
                    embedding_source = tf.get_variable(name='embedding_source',
                                                       shape=(self.vocab_size, struct_para.embedding_size),
                                                       dtype=tf.float32, trainable=False)
                    self.init_embedding_source = embedding_source.assign(self.embedding_source_pl)
                    self.init_embeddings = [self.init_embedding_source]
                    embedding_inf = tf.get_variable(name='embedding_inf',
                                                    shape=(self.vocab_size, struct_para.embedding_size),
                                                    dtype=tf.float32, trainable=True)
                    embedding_fm = tf.get_variable(name='embedding_fm',
                                                   shape=(self.vocab_size, struct_para.embedding_size),
                                                   dtype=tf.float32, trainable=True)
                    self.init_embedding_inf = embedding_inf.assign(self.embedding_source_pl)
                    self.init_embedding_fm = embedding_fm.assign(self.embedding_source_pl)
                    self.init_embeddings.append(self.init_embedding_fm)
                    self.init_embeddings.append(self.init_embedding_inf)
                else:
                    embedding_source = tf.get_variable(name='embedding_source',
                                                       shape=(self.vocab_size, struct_para.embedding_size),
                                                       dtype=tf.float32, trainable=True)
                    self.init_embedding_source = embedding_source.assign(self.embedding_source_pl)
                    self.init_embeddings = [self.init_embedding_source]
                    embedding_inf = embedding_source
                    embedding_fm = embedding_source
                src_max_len = tf.reduce_max(self.src_len)
                self.var_lr = tf.Variable(float(struct_para.learning_rate), trainable=False, dtype=tf.float32)
                self.lr_pl = tf.placeholder(name='lr_pl', shape=(), dtype=tf.float32)
                self.learning_rate_decay_op = self.var_lr.assign(self.lr_pl)
                self.match_lr = struct_para.match_lr
            with tf.variable_scope("encoder"):
                encoder_rnn_units = self.hidden_size
                if struct_para.encoder_is_bidirectional:
                    encoder_rnn_units = self.hidden_size / 2
                self.encoder = deep_components.multi_layer_encoder(encoder_rnn_units, self.keep_prob,
                                                                   layer_num=self.layers_num,
                                                                   cell_type=struct_para.encoder_cell_type,
                                                                   is_bidirectional=struct_para.encoder_is_bidirectional,
                                                                   init_method_for_rnn=init_method_for_rnn)
            with tf.variable_scope('encoding'):
                src_en_out, src_en_state = self.encoder(tf.nn.embedding_lookup(embedding_source, self.src),
                                                        self.src_len)
                mch1_en_out, mch1_en_state = self.encoder(tf.nn.embedding_lookup(embedding_source, self.mch1),
                                                          self.mch1_len)
                mch2_en_out, mch2_en_state = self.encoder(tf.nn.embedding_lookup(embedding_source, self.mch2),
                                                          self.mch2_len)
                rm_en_out, rm_en_state = self.encoder(tf.nn.embedding_lookup(embedding_source, self.random_selected_sen)
                                                      , self.random_selected_sen_len)
            with tf.variable_scope('informal_decoder'):
                self.informal_decoder = deep_components.multi_layer_decoder(memory=src_en_out,
                                                                            memory_len=self.src_len,
                                                                            memory_size=src_en_out.shape[2],
                                                                            vocab_size=vocab_size,
                                                                            embedding_size=self.embedding_size,
                                                                            keep_prob=self.keep_prob,
                                                                            rnn_units=self.hidden_size,
                                                                            layer_num=self.layers_num,
                                                                            init_method_for_rnn=init_method_for_rnn,
                                                                            stddev=struct_para.init_stddev
                                                                            )
            with tf.variable_scope('formal_decoder'):
                self.formal_decoder = deep_components.multi_layer_decoder(memory=src_en_out,
                                                                          memory_len=self.src_len,
                                                                          memory_size=src_en_out.shape[2],
                                                                          vocab_size=vocab_size,
                                                                          embedding_size=self.embedding_size,
                                                                          keep_prob=self.keep_prob,
                                                                          rnn_units=self.hidden_size,
                                                                          layer_num=self.layers_num,
                                                                          init_method_for_rnn=init_method_for_rnn,
                                                                          stddev=struct_para.init_stddev
                                                                          )
            with tf.variable_scope('src_matching_train'):
                q1_state = tf.concat((mch1_en_state, mch2_en_state, rm_en_state), axis=0)
                q2_state = tf.concat((mch2_en_state, rm_en_state, mch1_en_state), axis=0)
                match_fea1 = tf.abs(tf.subtract(q1_state, q2_state))
                match_fea2 = tf.matmul(tf.expand_dims(q1_state, axis=1),
                                       tf.expand_dims(q2_state, axis=2))[:, :, -1]
                match_vector = tf.concat((match_fea1, match_fea2), axis=1)
                hidden_layer1 = tf.layers.dense(match_vector, units=24, activation=tf.nn.relu)
                hidden_layer2 = tf.layers.dense(match_vector, units=24, activation=tf.nn.sigmoid)
                logits = tf.layers.dense(tf.concat((hidden_layer1, hidden_layer2), axis=1), units=2,
                                         activation=tf.nn.tanh)
                self.loss_match = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.match_label, logits=logits))
                self.loss_mse = tf.reduce_mean(tf.square(tf.subtract(q1_state, q2_state)),
                                               axis=1)
                float_label = tf.cast(self.match_label, dtype=tf.float32)
                self.loss_mse = tf.reduce_sum(tf.multiply(self.loss_mse, float_label)) / tf.cast(self.batch_size,
                                                                                                 dtype=tf.float32)
                self.loss_vector_length = tf.reduce_mean(
                    tf.square(1 * self.layers_num - tf.norm(q1_state, axis=1))) + tf.reduce_mean(
                    tf.square(1 * self.layers_num - tf.norm(q2_state, axis=1)))
                self.total_align_loss = struct_para.match_mse_lam * self.loss_mse + struct_para.match_cls_lam * self.loss_match \
                                        + struct_para.match_veclen_lam * self.loss_vector_length
                if struct_para.optimizer == 'sgd':
                    self.matching_train_op = tf.train.GradientDescentOptimizer(learning_rate=self.match_lr).minimize(
                        self.total_align_loss)
                else:
                    self.matching_train_op = tf.train.AdamOptimizer(learning_rate=self.match_lr).minimize(
                        self.total_align_loss)
            with tf.variable_scope("src2formal_decoder_train"):
                formal_decode_out_layer = tf.layers.Dense(self.vocab_size, name='formal_decode_output_layer',
                                                          _reuse=tf.AUTO_REUSE,
                                                          kernel_initializer=init_method_for_layers,
                                                          use_bias=False)
                fm_train_helper = TrainingHelper(tf.nn.embedding_lookup(embedding_fm, self.tgt_formal_input),
                                                 sequence_length=self.tgt_formal_len, name="src2formal_train_helper")
                fm_train_decoder = BasicDecoder(self.formal_decoder, fm_train_helper, initial_state=src_en_state,
                                                output_layer=formal_decode_out_layer)
                fm_dec_output, _, fm_gen_len = dynamic_decode(fm_train_decoder, impute_finished=True,
                                                              maximum_iterations=tf.reduce_max(self.tgt_formal_len))
                fm_gen_max_len = tf.reduce_max(fm_gen_len)

                fm_decoder_outputs = tf.identity(fm_dec_output.rnn_output)

                fm_decoder_target_mask = tf.sequence_mask(self.tgt_formal_len,
                                                          maxlen=fm_gen_max_len, dtype=tf.float32)
                self.src2formal_cost = sequence_loss(logits=fm_decoder_outputs, targets=self.tgt_formal_target,
                                                     weights=fm_decoder_target_mask, average_across_batch=False)
                #self.src2formal_cost = tf.multiply(self.src2formal_cost, tf.nn.softmax(self.sample_weights))
                self.src2formal_cost = tf.multiply(self.src2formal_cost,self.sample_weights)
                self.src2formal_cost = tf.reduce_mean(self.src2formal_cost)
                if struct_para.optimizer == 'sgd':
                    self.src2formal_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.var_lr).minimize(
                        self.src2formal_cost)
                else:
                    self.src2formal_optimizer = tf.train.AdamOptimizer(learning_rate=self.var_lr).minimize(
                        self.src2formal_cost)
            with tf.variable_scope("src2informal_decoder_train"):
                informal_decode_out_layer = tf.layers.Dense(self.vocab_size, name='informal_decode_output_layer',
                                                            kernel_initializer=init_method_for_layers,
                                                            _reuse=tf.AUTO_REUSE, use_bias=False)
                inf_train_helper = TrainingHelper(tf.nn.embedding_lookup(embedding_inf, self.tgt_informal_input),
                                                  sequence_length=self.tgt_informal_len,
                                                  name="src2informal_train_helper")
                inf_train_decoder = BasicDecoder(self.informal_decoder, inf_train_helper, initial_state=src_en_state,
                                                 output_layer=informal_decode_out_layer)
                inf_dec_output, _, inf_gen_len = dynamic_decode(inf_train_decoder, impute_finished=True,
                                                                maximum_iterations=tf.reduce_max(self.tgt_informal_len))
                inf_gen_max_len = tf.reduce_max(inf_gen_len)

                inf_decoder_outputs = tf.identity(inf_dec_output.rnn_output)

                inf_decoder_target_mask = tf.sequence_mask(self.tgt_informal_len,
                                                           maxlen=inf_gen_max_len, dtype=tf.float32)
                self.src2informal_cost = sequence_loss(logits=inf_decoder_outputs, targets=self.tgt_informal_target,
                                                       weights=inf_decoder_target_mask, average_across_batch=False)
                #self.src2informal_cost = tf.multiply(self.src2informal_cost, tf.nn.softmax(self.sample_weights))
                self.src2informal_cost = tf.multiply(self.src2informal_cost, self.sample_weights)
                self.src2informal_cost = tf.reduce_mean(self.src2informal_cost)
                if struct_para.optimizer == 'sgd':
                    self.src2informal_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.var_lr).minimize(
                        self.src2informal_cost)
                else:
                    self.src2informal_optimizer = tf.train.AdamOptimizer(learning_rate=self.var_lr).minimize(
                        self.src2informal_cost)
            with tf.variable_scope("src2formal-copynet"):
                fm_predict_id, fm_score, fm_attention = beamsearch.create_inference_graph(
                    decoding_fn=self.formal_decoder,
                    states=src_en_state,
                    memory=src_en_out,
                    memory_len=self.src_len,
                    decode_length=self.max_length,
                    batch_size=self.batch_size,
                    beam_size=self.beam_size,
                    decode_alpha=struct_para.decode_alpha,
                    bos_id=self.SOS_token,
                    eos_id=self.EOS_token,
                    embedding=embedding_fm,
                    output_layer=formal_decode_out_layer,
                    src_max_len=src_max_len)
                self.fm_id_copy = fm_predict_id
                self.fm_attn_copy = fm_attention
            with tf.variable_scope("src2informal-copynet"):
                inf_predict_id, inf_score, inf_attention = beamsearch.create_inference_graph(
                    decoding_fn=self.informal_decoder,
                    states=src_en_state,
                    memory=src_en_out,
                    memory_len=self.src_len,
                    decode_length=self.max_length,
                    batch_size=self.batch_size,
                    beam_size=self.beam_size,
                    decode_alpha=struct_para.decode_alpha,
                    bos_id=self.SOS_token,
                    eos_id=self.EOS_token,
                    embedding=embedding_inf,
                    output_layer=informal_decode_out_layer,
                    src_max_len=src_max_len)
                self.inf_id_copy = inf_predict_id
                self.inf_attn_copy = inf_attention

    def learning_rate_decay(self, sess):
        self.lr *= self.lr_decay_rate
        sess.run([self.learning_rate_decay_op], feed_dict={self.lr_pl: self.lr})

    def train(self, sess, train_step, inf_input, inf_input_len, fm_input, fm_input_len, rm_input, rm_input_len,
              tgt_inf_input, tgt_inf_output, tgt_inf_len, tgt_fm_input, tgt_fm_output, tgt_fm_len, batch_size,
              apply_sample_weight=0,dataset_weight=1):
        res = []
        feed_dict = {}
        #print(batch_size)
        losses = {'total_aligned_loss': 0.0, 'loss_match': 0.0, 'loss_mse': 0.0, 'loss_vector_length': 0.0,
                  'inf2fm_loss': 0.0, 'fm2inf_loss': 0.0, 'fm2fm_loss': 0.0, 'inf2inf_loss': 0.0,
                  'fm2fm_swap_loss':0.0, 'inf2inf_swap_loss':0.0,
                  'masked_fm_lm_loss': 0.0, 'masked_inf_lm_loss': 0.0,
                  'fm_bt_loss': 0.0, 'inf_bt_loss': 0.0}
        basic_sample_weights=[1.0]*batch_size
        inf_sample_weights=basic_sample_weights
        fm_sample_weights=basic_sample_weights
        if apply_sample_weight!=0:
            pass
            #inf_tok=self.parse_output(tgt_inf_output,with_unk=False,with_eos=False)
            #inf_sample_weights=self.weight_generator.calculate_formality_reward(inf_tok,'inf',alpha=apply_sample_weight)
            #fm_tok=self.parse_output(tgt_fm_output,with_unk=False,with_eos=False)
            #fm_sample_weights = self.weight_generator.calculate_formality_reward(fm_tok,'fm',alpha=apply_sample_weight)
        inf_sample_weights=[i*dataset_weight for i in inf_sample_weights]
        fm_sample_weights = [i * dataset_weight for i in fm_sample_weights]
        if 'match' in train_step:
            res.clear()
            res.append(self.matching_train_op)
            res.append(self.total_align_loss)
            res.append(self.loss_match)
            res.append(self.loss_mse)
            res.append(self.loss_vector_length)
            feed_dict.clear()
            feed_dict[self.match_label] = [int(1)] * batch_size + [int(0)] * (batch_size * 2)
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.mch1] = inf_input
            feed_dict[self.mch1_len] = inf_input_len
            feed_dict[self.mch2] = fm_input
            feed_dict[self.mch2_len] = fm_input_len
            feed_dict[self.random_selected_sen] = rm_input
            feed_dict[self.random_selected_sen_len] = rm_input_len
            feed_dict[self.keep_prob] = self.keep_prob_value
            _, t_l, mch_l, mse_l, len_l = sess.run(res, feed_dict=feed_dict)
            losses['total_aligned_loss'] += t_l
            losses['loss_match'] += mch_l
            losses['loss_mse'] += mse_l
            losses['loss_vector_length'] += len_l
        if 'fm2fm' in train_step:
            res.clear()
            res.append(self.src2formal_optimizer)
            res.append(self.src2formal_cost)
            feed_dict.clear()
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = fm_input
            feed_dict[self.src_len] = fm_input_len
            feed_dict[self.tgt_formal_input] = tgt_fm_input
            feed_dict[self.tgt_formal_target] = tgt_fm_output
            feed_dict[self.tgt_formal_len] = tgt_fm_len
            feed_dict[self.keep_prob] = self.keep_prob_value
            feed_dict[self.sample_weights] = fm_sample_weights
            _, l = sess.run(res, feed_dict=feed_dict)
            losses['fm2fm_loss']=l
        if 'inf2inf' in train_step:
            res.clear()
            res.append(self.src2informal_optimizer)
            res.append(self.src2informal_cost)
            feed_dict.clear()
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = inf_input
            feed_dict[self.src_len] = inf_input_len
            feed_dict[self.tgt_informal_input] = tgt_inf_input
            feed_dict[self.tgt_informal_target] = tgt_inf_output
            feed_dict[self.tgt_informal_len] = tgt_inf_len
            feed_dict[self.keep_prob] = self.keep_prob_value
            feed_dict[self.sample_weights] = inf_sample_weights
            _, l = sess.run(res, feed_dict=feed_dict)
            losses['inf2inf_loss']=l
        if 'fm2fm_swap' in train_step:
            res.clear()
            res.append(self.src2formal_optimizer)
            res.append(self.src2formal_cost)
            feed_dict.clear()
            # denoising:
            tmp_fm_input = copy_list(fm_input)
            for sample,sample_len in zip(tmp_fm_input,fm_input_len):
                swap_times=min(int(sample_len*self.denoising_swap_rate/2),self.max_swap_times)
                for i in range(0,swap_times):
                    index_a=np.random.randint(0,sample_len)
                    index_b=np.random.randint(max(0,index_a-2),min(sample_len,index_a+2))
                    tmp=sample[index_a]
                    sample[index_a]=sample[index_b]
                    sample[index_b]=tmp
            #feed_dict
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = tmp_fm_input
            feed_dict[self.src_len] = fm_input_len
            feed_dict[self.tgt_formal_input] = tgt_fm_input
            feed_dict[self.tgt_formal_target] = tgt_fm_output
            feed_dict[self.tgt_formal_len] = tgt_fm_len
            feed_dict[self.keep_prob] = self.keep_prob_value
            feed_dict[self.sample_weights] = basic_sample_weights
            _, l = sess.run(res, feed_dict=feed_dict)
            losses['fm2fm_swap_loss'] = l
        if 'inf2inf_swap' in train_step:
            res.clear()
            res.append(self.src2informal_optimizer)
            res.append(self.src2informal_cost)
            feed_dict.clear()
            # denoising
            tmp_inf_input = copy_list(inf_input)
            for sample,sample_len in zip(tmp_inf_input,inf_input_len):
                swap_times=min(int(sample_len*self.denoising_swap_rate/2),self.max_swap_times)
                for i in range(0,swap_times):
                    index_a=np.random.randint(0,sample_len)
                    index_b=np.random.randint(max(0,index_a-2),min(sample_len,index_a+2))
                    tmp=sample[index_a]
                    sample[index_a]=sample[index_b]
                    sample[index_b]=tmp
            # feed_dict
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = tmp_inf_input
            feed_dict[self.src_len] = inf_input_len
            feed_dict[self.tgt_informal_input] = tgt_inf_input
            feed_dict[self.tgt_informal_target] = tgt_inf_output
            feed_dict[self.tgt_informal_len] = tgt_inf_len
            feed_dict[self.keep_prob] = self.keep_prob_value
            feed_dict[self.sample_weights] = basic_sample_weights
            _, l = sess.run(res, feed_dict=feed_dict)
            losses['inf2inf_swap_loss'] = l
        if 'masked_fm_lm' in train_step:
            res.clear()
            res.append(self.src2formal_optimizer)
            res.append(self.src2formal_cost)
            feed_dict.clear()
            # masking:
            tmp_fm_input = copy_list(fm_input)
            tmp_tgt_fm_input = copy_list(tgt_fm_input)
            for sample, sample_len in zip(tmp_fm_input, fm_input_len):
                max_mask_times = int(min([self.max_lm_masked_num, self.max_lm_masked_rate * sample_len]))
                if max_mask_times < 2:
                    mask_times = 0
                else:
                    mask_times = np.random.randint(1, max_mask_times + 1)
                for i in range(0, mask_times):
                    index_a = np.random.randint(0, sample_len)
                    sample[index_a] = self.UNK_token
            for sample, sample_len in zip(tmp_tgt_fm_input, tgt_fm_len):
                max_mask_times = int(min([self.max_lm_masked_num, self.max_lm_masked_rate * sample_len]))
                if max_mask_times < 2:
                    mask_times = 0
                else:
                    mask_times = np.random.randint(1, max_mask_times + 1)
                for i in range(0, mask_times):
                    index_a = np.random.randint(1, sample_len)
                    sample[index_a] = self.UNK_token
            # feed_dict
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = tmp_fm_input
            feed_dict[self.src_len] = fm_input_len
            feed_dict[self.tgt_formal_input] = tmp_tgt_fm_input
            feed_dict[self.tgt_formal_target] = tgt_fm_output
            feed_dict[self.tgt_formal_len] = tgt_fm_len
            feed_dict[self.keep_prob] = self.keep_prob_value
            feed_dict[self.sample_weights] = basic_sample_weights
            _, l = sess.run(res, feed_dict=feed_dict)
            losses['masked_fm_lm_loss'] = l
        if 'masked_inf_lm' in train_step:
            res.clear()
            res.append(self.src2informal_optimizer)
            res.append(self.src2informal_cost)
            feed_dict.clear()
            # masking
            tmp_inf_input = copy_list(inf_input)
            tmp_tgt_inf_input = copy_list(tgt_inf_input)
            for sample, sample_len in zip(tmp_inf_input, inf_input_len):
                max_mask_times = int(min([self.max_lm_masked_num, self.max_lm_masked_rate * sample_len]))
                if max_mask_times < 2:
                    mask_times = 0
                else:
                    mask_times = np.random.randint(1, max_mask_times + 1)
                for i in range(0, mask_times):
                    index_a = np.random.randint(0, sample_len)
                    sample[index_a] = self.UNK_token
            for sample, sample_len in zip(tmp_tgt_inf_input, tgt_inf_len):
                max_mask_times = int(min([self.max_lm_masked_num, self.max_lm_masked_rate * sample_len]))
                if max_mask_times < 2:
                    mask_times = 0
                else:
                    mask_times = np.random.randint(1, max_mask_times + 1)
                for i in range(0, mask_times):
                    index_a = np.random.randint(1, sample_len)
                    sample[index_a] = self.UNK_token
            # feed_dict
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = tmp_inf_input
            feed_dict[self.src_len] = inf_input_len
            feed_dict[self.tgt_informal_input] = tmp_tgt_inf_input
            feed_dict[self.tgt_informal_target] = tgt_inf_output
            feed_dict[self.tgt_informal_len] = tgt_inf_len
            feed_dict[self.keep_prob] = self.keep_prob_value
            feed_dict[self.sample_weights] = basic_sample_weights
            _, l = sess.run(res, feed_dict=feed_dict)
            losses['masked_inf_lm_loss'] = l
        if 'inf2fm' in train_step:
            res.clear()
            res.append(self.src2formal_optimizer)
            res.append(self.src2formal_cost)
            feed_dict.clear()
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = inf_input
            feed_dict[self.src_len] = inf_input_len
            feed_dict[self.tgt_formal_input] = tgt_fm_input
            feed_dict[self.tgt_formal_target] = tgt_fm_output
            feed_dict[self.tgt_formal_len] = tgt_fm_len
            feed_dict[self.keep_prob] = self.keep_prob_value
            feed_dict[self.sample_weights] = fm_sample_weights
            _, l = sess.run(res, feed_dict=feed_dict)
            losses['inf2fm_loss'] = l
        if 'fm2inf' in train_step:
            res.clear()
            res.append(self.src2informal_optimizer)
            res.append(self.src2informal_cost)
            feed_dict.clear()
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = fm_input
            feed_dict[self.src_len] = fm_input_len
            feed_dict[self.tgt_informal_input] = tgt_inf_input
            feed_dict[self.tgt_informal_target] = tgt_inf_output
            feed_dict[self.tgt_informal_len] = tgt_inf_len
            feed_dict[self.keep_prob] = self.keep_prob_value
            feed_dict[self.sample_weights] = inf_sample_weights
            _, l = sess.run(res, feed_dict=feed_dict)
            losses['fm2inf_loss'] = l
        if 'fm_bt' in train_step:
            res.clear()
            res.append(self.inf_id_copy)
            feed_dict.clear()
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = fm_input
            feed_dict[self.src_len] = fm_input_len
            feed_dict[self.keep_prob] = 1.0
            inf_gen = sess.run(res, feed_dict=feed_dict)[0]
            inf_gen = self.parse_output(inf_gen[:, 0, 1:])
            inf_gen, inf_gen_len = padding_batch(inf_gen)
            res.clear()
            res.append(self.src2formal_optimizer)
            res.append(self.src2formal_cost)
            feed_dict.clear()
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = inf_gen
            feed_dict[self.src_len] = inf_gen_len
            feed_dict[self.tgt_formal_input] = tgt_fm_input
            feed_dict[self.tgt_formal_target] = tgt_fm_output
            feed_dict[self.tgt_formal_len] = tgt_fm_len
            feed_dict[self.keep_prob] = self.keep_prob_value
            feed_dict[self.sample_weights] = basic_sample_weights
            _, l = sess.run(res, feed_dict=feed_dict)
            losses['fm_bt_loss'] = l
        if 'inf_bt' in train_step:
            res.clear()
            res.append(self.fm_id_copy)
            feed_dict.clear()
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = inf_input
            feed_dict[self.src_len] = inf_input_len
            feed_dict[self.keep_prob] = 1.0
            fm_gen = sess.run(res, feed_dict=feed_dict)[0]
            fm_gen = self.parse_output(fm_gen[:, 0, 1:])
            fm_gen, fm_gen_len = padding_batch(fm_gen)
            res.clear()
            res.append(self.src2informal_optimizer)
            res.append(self.src2informal_cost)
            feed_dict.clear()
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = fm_gen
            feed_dict[self.src_len] = fm_gen_len
            feed_dict[self.tgt_informal_input] = tgt_inf_input
            feed_dict[self.tgt_informal_target] = tgt_inf_output
            feed_dict[self.tgt_informal_len] = tgt_inf_len
            feed_dict[self.keep_prob] = self.keep_prob_value
            feed_dict[self.sample_weights] = basic_sample_weights
            _, l = sess.run(res, feed_dict=feed_dict)
            losses['inf_bt_loss'] = l
        if 'fm_st' in train_step:
            res.clear()
            res.append(self.inf_id_copy)
            feed_dict.clear()
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = fm_input
            feed_dict[self.src_len] = fm_input_len
            feed_dict[self.keep_prob] = 1.0
            inf_gen = sess.run(res, feed_dict=feed_dict)[0]
            inf_input_tgt = self.parse_output(inf_gen[:, 0, :])#with BOS
            inf_output_tgt = self.parse_output(inf_gen[:, 0, 1:],with_eos=True)
            inf_input_tgt, inf_gen_len = padding_batch(inf_input_tgt)
            inf_output_tgt, _ = padding_batch(inf_output_tgt)
            res.clear()
            res.append(self.src2informal_optimizer)
            res.append(self.src2informal_cost)
            feed_dict.clear()
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = fm_input
            feed_dict[self.src_len] = fm_input_len
            feed_dict[self.tgt_informal_input] = inf_input_tgt
            feed_dict[self.tgt_informal_target] = inf_output_tgt
            feed_dict[self.tgt_informal_len] = inf_gen_len
            feed_dict[self.keep_prob] = self.keep_prob_value
            feed_dict[self.sample_weights] = basic_sample_weights
            _, l = sess.run(res, feed_dict=feed_dict)
            losses['fm_st_loss'] = l
        if 'inf_st' in train_step:
            res.clear()
            res.append(self.fm_id_copy)
            feed_dict.clear()
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = inf_input
            feed_dict[self.src_len] = inf_input_len
            feed_dict[self.keep_prob] = 1.0
            fm_gen = sess.run(res, feed_dict=feed_dict)[0]
            fm_input_tgt = self.parse_output(fm_gen[:, 0, :])  # with BOS
            fm_output_tgt = self.parse_output(fm_gen[:, 0, 1:], with_eos=True)
            fm_input_tgt, fm_gen_len = padding_batch(fm_input_tgt)
            fm_output_tgt, _ = padding_batch(fm_output_tgt)
            res.clear()
            res.append(self.src2informal_optimizer)
            res.append(self.src2informal_cost)
            feed_dict.clear()
            feed_dict[self.batch_size] = batch_size
            feed_dict[self.src] = inf_input
            feed_dict[self.src_len] = inf_input_len
            feed_dict[self.tgt_informal_input] = fm_input_tgt
            feed_dict[self.tgt_informal_target] = fm_output_tgt
            feed_dict[self.tgt_informal_len] = fm_gen_len
            feed_dict[self.keep_prob] = self.keep_prob_value
            feed_dict[self.sample_weights] = basic_sample_weights
            _, l = sess.run(res, feed_dict=feed_dict)
            losses['inf_st_loss'] = l
        return losses

    def evaluate(self, sess, evalaute_step, data_num, all_inf_input, all_fm_input, all_rm_input,
                 all_tgt_inf_input, all_tgt_inf_output,
                 all_tgt_fm_input, all_tgt_fm_output, inf2fm_val_data=None, fm2inf_val_data=None,
                 batch_size=256):
        low = 0
        res = []
        losses = {'total_aligned_loss': 0.0, 'loss_match': 0.0, 'loss_mse': 0.0, 'loss_vector_length': 0.0,
                  'inf2fm_loss': 0.0, 'fm2inf_loss': 0.0, 'fm2fm_loss': 0.0, 'inf2inf_loss': 0.0,
                  'fm_bt_loss': 0.0, 'inf_bt_loss': 0.0}
        feed_dict = {}
        inf2fm_gen = []
        fm2inf_gen = []
        while low < data_num:
            n_samples = min([batch_size, data_num - low])
            inf_input, inf_input_len = padding_batch(copy_list(all_inf_input[low:low + n_samples]))
            fm_input, fm_input_len = padding_batch(copy_list(all_fm_input[low:low + n_samples]))
            tgt_fm_input, tgt_fm_len = padding_batch(copy_list(all_tgt_fm_input[low:low + n_samples]))
            tgt_fm_output, _ = padding_batch(copy_list(all_tgt_fm_output[low:low + n_samples]))
            tgt_inf_input, tgt_inf_len = padding_batch(copy_list(all_tgt_inf_input[low:low + n_samples]))
            tgt_inf_output, _ = padding_batch(copy_list(all_tgt_inf_output[low:low + n_samples]))
            basic_sample_weights = [1.0] * n_samples
            if 'match' in evalaute_step:
                idx = [random.randint(0, len(all_rm_input) - 1) for i in range(0, n_samples)]
                rm_input = [all_rm_input[i] for i in idx]
                rm_input, rm_input_len = padding_batch(copy_list(rm_input))
                res.clear()
                res.append(self.total_align_loss)
                res.append(self.loss_match)
                res.append(self.loss_mse)
                res.append(self.loss_vector_length)
                feed_dict.clear()
                feed_dict[self.match_label] = [int(1)] * n_samples + [int(0)] * (n_samples * 2)
                feed_dict[self.batch_size] = n_samples
                feed_dict[self.mch1] = inf_input
                feed_dict[self.mch1_len] = inf_input_len
                feed_dict[self.mch2] = fm_input
                feed_dict[self.mch2_len] = fm_input_len
                feed_dict[self.random_selected_sen] = rm_input
                feed_dict[self.random_selected_sen_len] = rm_input_len
                feed_dict[self.keep_prob] = 1.0
                t_l, mch_l, mse_l, len_l = sess.run(res, feed_dict=feed_dict)
                losses['total_aligned_loss'] += t_l * n_samples
                losses['loss_match'] += mch_l * n_samples
                losses['loss_mse'] += mse_l * n_samples
                losses['loss_vector_length'] += len_l * n_samples
            if 'fm2fm' in evalaute_step:
                res.clear()
                res.append(self.src2formal_cost)
                feed_dict.clear()
                feed_dict[self.batch_size] = n_samples
                feed_dict[self.src] = fm_input
                feed_dict[self.src_len] = fm_input_len
                feed_dict[self.tgt_formal_input] = tgt_fm_input
                feed_dict[self.tgt_formal_target] = tgt_fm_output
                feed_dict[self.tgt_formal_len] = tgt_fm_len
                feed_dict[self.keep_prob] = 1.0
                feed_dict[self.sample_weights] = basic_sample_weights
                l = sess.run(res, feed_dict=feed_dict)
                losses['fm2fm_loss'] += l[0] * n_samples
            if 'inf2inf' in evalaute_step:
                res.clear()
                res.append(self.src2informal_cost)
                feed_dict.clear()
                feed_dict[self.batch_size] = n_samples
                feed_dict[self.src] = inf_input
                feed_dict[self.src_len] = inf_input_len
                feed_dict[self.tgt_informal_input] = tgt_inf_input
                feed_dict[self.tgt_informal_target] = tgt_inf_output
                feed_dict[self.tgt_informal_len] = tgt_inf_len
                feed_dict[self.keep_prob] = 1.0
                feed_dict[self.sample_weights] = basic_sample_weights
                l = sess.run(res, feed_dict=feed_dict)
                losses['inf2inf_loss'] += l[0] * n_samples
            if 'inf2fm' in evalaute_step:
                res.clear()
                res.append(self.src2formal_cost)
                feed_dict.clear()
                feed_dict[self.batch_size] = n_samples
                feed_dict[self.src] = inf_input
                feed_dict[self.src_len] = inf_input_len
                feed_dict[self.tgt_formal_input] = tgt_fm_input
                feed_dict[self.tgt_formal_target] = tgt_fm_output
                feed_dict[self.tgt_formal_len] = tgt_fm_len
                feed_dict[self.keep_prob] = 1.0
                feed_dict[self.sample_weights] = basic_sample_weights
                l = sess.run(res, feed_dict=feed_dict)
                losses['inf2fm_loss'] += l[0] * n_samples
            if 'fm2inf' in evalaute_step:
                res.clear()
                res.append(self.src2informal_cost)
                feed_dict.clear()
                feed_dict[self.batch_size] = n_samples
                feed_dict[self.src] = fm_input
                feed_dict[self.src_len] = fm_input_len
                feed_dict[self.tgt_informal_input] = tgt_inf_input
                feed_dict[self.tgt_informal_target] = tgt_inf_output
                feed_dict[self.tgt_informal_len] = tgt_inf_len
                feed_dict[self.keep_prob] = 1.0
                feed_dict[self.sample_weights] = basic_sample_weights
                l = sess.run(res, feed_dict=feed_dict)
                losses['fm2inf_loss'] += l[0] * n_samples
            if 'fm_bt' in evalaute_step:
                res.clear()
                res.append(self.inf_id_copy)
                feed_dict.clear()
                feed_dict[self.batch_size] = n_samples
                feed_dict[self.src] = fm_input
                feed_dict[self.src_len] = fm_input_len
                feed_dict[self.keep_prob] = 1.0
                inf_gen = sess.run(res, feed_dict=feed_dict)[0]
                inf_gen = self.parse_output(inf_gen[:, 0, 1:])
                inf_gen, inf_gen_len = padding_batch(inf_gen)
                res.clear()
                res.append(self.src2formal_cost)
                feed_dict.clear()
                feed_dict[self.batch_size] = n_samples
                feed_dict[self.src] = inf_gen
                feed_dict[self.src_len] = inf_gen_len
                feed_dict[self.tgt_formal_input] = tgt_fm_input
                feed_dict[self.tgt_formal_target] = tgt_fm_output
                feed_dict[self.tgt_formal_len] = tgt_fm_len
                feed_dict[self.keep_prob] = 1.0
                feed_dict[self.sample_weights] = basic_sample_weights
                l = sess.run(res, feed_dict=feed_dict)
                losses['fm_bt_loss'] += l[0] * n_samples
            if 'inf_bt' in evalaute_step:
                res.clear()
                res.append(self.fm_id_copy)
                feed_dict.clear()
                feed_dict[self.batch_size] = n_samples
                feed_dict[self.src] = inf_input
                feed_dict[self.src_len] = inf_input_len
                feed_dict[self.keep_prob] = 1.0
                fm_gen = sess.run(res, feed_dict=feed_dict)[0]
                fm_gen = self.parse_output(fm_gen[:, 0, 1:])
                fm_gen, fm_gen_len = padding_batch(fm_gen)
                res.clear()
                res.append(self.src2informal_cost)
                feed_dict.clear()
                feed_dict[self.batch_size] = n_samples
                feed_dict[self.src] = fm_gen
                feed_dict[self.src_len] = fm_gen_len
                feed_dict[self.tgt_informal_input] = tgt_inf_input
                feed_dict[self.tgt_informal_target] = tgt_inf_output
                feed_dict[self.tgt_informal_len] = tgt_inf_len
                feed_dict[self.keep_prob] = 1.0
                feed_dict[self.sample_weights] = basic_sample_weights
                l = sess.run(res, feed_dict=feed_dict)
                losses['inf_bt_loss'] += l[0] * n_samples
            if 'inf2fm_bleu' in evalaute_step:
                res.clear()
                res.append(self.fm_id_copy)
                feed_dict.clear()
                feed_dict[self.batch_size] = n_samples
                feed_dict[self.src] = inf_input
                feed_dict[self.src_len] = inf_input_len
                feed_dict[self.keep_prob] = 1.0
                fm_gen = sess.run(res, feed_dict=feed_dict)[0]
                fm_gen = self.parse_output(fm_gen[:, 0, 1:])
                for one_sen_index in fm_gen:
                    inf2fm_gen.append(inf2fm_val_data.indices2sentence(one_sen_index))
            if 'fm2inf_bleu' in evalaute_step:
                res.clear()
                res.append(self.inf_id_copy)
                feed_dict.clear()
                feed_dict[self.batch_size] = n_samples
                feed_dict[self.src] = fm_input
                feed_dict[self.src_len] = fm_input_len
                feed_dict[self.keep_prob] = 1.0
                inf_gen = sess.run(res, feed_dict=feed_dict)[0]
                inf_gen = self.parse_output(inf_gen[:, 0, 1:])
                for one_sen_index in inf_gen:
                    fm2inf_gen.append(fm2inf_val_data.indices2sentence(one_sen_index))
            low += n_samples
        losses['total_aligned_loss'] /= (data_num)
        losses['loss_match'] /= data_num
        losses['loss_mse'] /= data_num
        losses['loss_vector_length'] /= data_num
        losses['fm2inf_loss'] /= data_num
        losses['inf2fm_loss'] /= data_num
        losses['fm2fm_loss'] /= data_num
        losses['inf2inf_loss'] /= data_num
        losses['fm_bt_loss'] /= data_num
        losses['inf_bt_loss'] /= data_num
        if 'inf2fm_bleu' in evalaute_step:
            losses['inf2fm_bleu'] = -inf2fm_val_data.test_bleu(inf2fm_gen)
        if 'fm2inf_bleu' in evalaute_step:
            losses['fm2inf_bleu'] = -fm2inf_val_data.test_bleu(fm2inf_gen)
        return losses

    def generate_informal2formal(self, sess, encoder_inputs, encoder_length, batch_size, res):
        """
        feed data to generate.
        :param sess: session
        :param encoder_inputs: encoder inputs
        :param encoder_length: encoder inputs sequence length, (batch=1, )
        :return:
        """
        if encoder_inputs.ndim == 1:
            encoder_inputs = encoder_inputs.reshape((1, -1))
            encoder_length = encoder_length.reshape((1,))
        result = sess.run(res, feed_dict={self.src: encoder_inputs,
                                          self.src_len: encoder_length,
                                          self.keep_prob: 1.0,
                                          self.batch_size: batch_size,
                                          })
        return result

    def generate_formal2informal(self, sess, encoder_inputs, encoder_length, batch_size, res):
        """
        feed data to generate.
        :param sess: session
        :param encoder_inputs: encoder inputs
        :param encoder_length: encoder inputs sequence length, (batch=1, )
        :return:
        """
        if encoder_inputs.ndim == 1:
            encoder_inputs = encoder_inputs.reshape((1, -1))
            encoder_length = encoder_length.reshape((1,))
        result = sess.run(res, feed_dict={self.src: encoder_inputs,
                                          self.src_len: encoder_length,
                                          self.keep_prob: 1.0,
                                          self.batch_size: batch_size,
                                          })
        return result

    def parse_output(self, token_indices, with_unk=False, is_infer=False, with_eos=False):
        res = []
        unk_counter = 0
        for one_sen in token_indices:
            sen = []
            for token in one_sen:
                if token != self.EOS_token:  # end
                    if token != self.UNK_token or with_unk:
                        sen.append(token)
                    if token == self.UNK_token:
                        unk_counter += 1
                else:
                    break
            if with_eos:
                sen.append(self.EOS_token)
            res.append(sen)
        if is_infer:
            print('unk num:', unk_counter)
        return res


class prepare_data():
    def __init__(self, informal_file, formal_file, embedding_file, to_lower_case=False, ref_list=None):
        self.embedding_file = embedding_file
        self.informal_file = informal_file
        self.formal_file = formal_file
        self.to_lower_case = to_lower_case
        self.embedding, self.vocab_hash = self.load_fasttext_embedding()
        self.SOS_token = np.int32(self.add_term_to_embedding('<SOS>', [0.0] * len(self.embedding[0])))
        self.UNK_token = np.int32(self.add_term_to_embedding('<UNK>', [0.0] * len(self.embedding[0])))
        self.EOS_token = np.int32(self.add_term_to_embedding('<EOS>', [0.0] * len(self.embedding[0])))
        self.index2word = self.gen_index2word_dict()
        self.informal_input, self.formal_input, self.inf_src_tok, self.fm_src_tok = self.process_source_file()
        self.tgt_informal_input, self.tgt_informal_output, self.tgt_formal_input, self.tgt_formal_output = self.process_target_file()
        self.all_sentences = self.informal_input + self.formal_input
        self.refs = []
        if type(ref_list) == type([]):
            for ref_path in ref_list:
                print(ref_path)
                self.refs.append(self.load_origin_ref_data(ref_path))
            self.refs = [[self.refs[i][j] for i in range(0, len(self.refs))] for j in range(0, len(self.refs[0]))]

    def test_bleu(self, gen_results, ngrams=4):
        if len(self.refs) > 0:
            gen = []
            for line in gen_results:
                gen.append(line.strip().lower().replace('@@ ','').split())
            weight = [1.0 / ngrams] * ngrams
            return corpus_bleu(self.refs, gen, weights=weight)
        else:
            raise ValueError('this dataset has no reference')

    def load_origin_ref_data(self, ref_path):
        data = []
        with open(ref_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(nltk.word_tokenize(line.strip().lower()))
        return data

    def padding_and_get_len(self, list, max_len=None):
        len_list = []
        tol_num = len(list)
        cut_num = 0
        if max_len is None:
            pass
        else:
            for i in range(0, len(list)):
                if len(list[i]) < max_len:
                    len_list.append(len(list[i]))
                    list[i] = list[i] + [0] * (max_len - len(list[i]))
                else:
                    len_list.append(max_len)
                    list[i] = list[i][:max_len]
                    cut_num += 1
        print('cut_num:', cut_num)
        print('total_num:', tol_num)
        return len_list

    def process_source_file(self,max_len=None):
        inf_source = []
        inf_source_tok = []
        with open(self.informal_file, 'r', encoding='utf-8') as f:
            for line in f:
                if self.to_lower_case:
                    line = line.lower()
                words = line.strip().split()
                inf_source_tok.append(words)
                inf_source.append(self.sentence2indices(words))
        fm_source = []
        fm_source_tok = []
        with open(self.formal_file, 'r', encoding='utf-8') as f:
            for line in f:
                if self.to_lower_case:
                    line = line.lower()
                words = line.strip().split()
                fm_source_tok.append(words)
                fm_source.append(self.sentence2indices(words))
        return inf_source, fm_source, inf_source_tok, fm_source_tok

    def process_target_file(self):
        inf_target_output = []
        inf_target_input = []
        with open(self.informal_file, 'r', encoding='utf-8') as f:
            for line in f:
                if self.to_lower_case:
                    line = line.lower()
                words = line.strip().split()
                inf_target_input.append(self.sentence2indices(words, with_sos=True))
                inf_target_output.append(self.sentence2indices(words, with_eos=True))
        fm_target_output = []
        fm_target_input = []
        with open(self.formal_file, 'r', encoding='utf-8') as f:
            for line in f:
                if self.to_lower_case:
                    line = line.lower()
                words = line.strip().split()
                fm_target_input.append(self.sentence2indices(words, with_sos=True))
                fm_target_output.append(self.sentence2indices(words, with_eos=True))
        return inf_target_input, inf_target_output, fm_target_input, fm_target_output

    def gen_index2word_dict(self):
        i2d = []
        tmp = sorted(self.vocab_hash.items(), key=lambda d: d[1])
        for item in tmp:
            i2d.append(item[0])
        return i2d

    def add_term_to_embedding(self, term, vector):
        self.vocab_hash[term] = len(self.vocab_hash)
        self.embedding.append(vector)
        return self.vocab_hash[term]

    def load_fasttext_embedding(self):
        vectors = []
        vocab_hash = {}
        with open(self.embedding_file, 'r', encoding='utf-8') as f:
            first_line = True
            for line in f:
                if first_line:
                    first_line = False
                    continue
                strs = line.strip().split(' ')
                vocab_hash[strs[0]] = len(vectors)
                vectors.append([float(s) for s in strs[1:]])
        return vectors, vocab_hash

    def indices2sentence(self, idxs, join=True):
        if join:
            return " ".join([self.index2word[idx] for idx in idxs])
        else:
            return [self.index2word[idx] for idx in idxs]

    def sentence2indices(self, words, with_sos=False, with_eos=False):
        idxs = []
        if with_sos:
            idxs.append(self.SOS_token)
        idxs += [self.vocab_hash.get(token, self.UNK_token) for token in words]  # default to <unk>
        if with_eos:
            idxs.append(self.EOS_token)
        return idxs


def preprocess():
    train_data = prepare_data(informal_train_path,
                              formal_train_path,
                              embedding_file=embedding_path,
                              to_lower_case=False)
    pickle.dump(train_data, open(train_pkl_path, 'wb'), protocol=True)
    val_data = prepare_data(informal_val_path,
                            formal_val_path,
                            embedding_file=embedding_path,
                            to_lower_case=False)
    pickle.dump(val_data, open(val_pkl_path, 'wb'), protocol=True)
    test_data = prepare_data(informal_test_path,
                             formal_test_path,
                             embedding_file=embedding_path,
                             to_lower_case=False)
    pickle.dump(test_data, open(test_pkl_path, 'wb'), protocol=True)


def gen_val_data_for_bleu():
    inf2fm_bleu_fr = prepare_data(informal_file=informal_val_path,
                                    formal_file=formal_val_path,
                                    embedding_file=embedding_path,
                                    ref_list=informal_val_refs,
                                    to_lower_case=False)
    pickle.dump(inf2fm_bleu_fr, open(inf2fm_bleu_val_pkl_path, 'wb'), protocol=True)
    fm2inf_bleu_data=prepare_data(informal_file=fm2inf_bleu_inf_src,
                             formal_file=fm2inf_bleu_fm_src,
                             embedding_file=embedding_path,
                             ref_list=formal_val_refs,
                             to_lower_case=False)
    pickle.dump(fm2inf_bleu_data,open(fm2inf_bleu_val_pkl_path,'wb'),protocol=True)




def padding_batch(input_list):
    in_len = [len(i) for i in input_list]
    new_in = pad_sequences(input_list, padding='post')
    return new_in, in_len


def copy_list(list):
    new_list = []
    for l in list:
        if type(l) == type([0]) or type(l) == type(np.array([0])):
            new_list.append(copy_list(l))
        else:
            new_list.append(l)
    return new_list


def generate_copy(copy=True,save_bpe_result=True,para=None):
    if para is None:
        para = generate_parameters()
    test_data = pickle.load(open(para.input_path, 'rb'))
    embedding, vocab_hash = load_word_embedding(embedding_path)
    nmt = seq2seq(vocab_size=len(test_data.embedding),
                  sos_token=test_data.SOS_token, eos_token=test_data.EOS_token,
                  unk_token=test_data.UNK_token, struct_para=para,
                  vocab_hash=vocab_hash)
    with nmt.graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=nmt.graph)
        saver = tf.train.Saver(max_to_keep=3)
        saver.restore(sess, para.model_path)
        output = []
        low_pointer = 0
        if para.src_is_inf:
            data_num = len(test_data.informal_input)
        else:
            data_num = len(test_data.formal_input)
        i = 0
        while low_pointer < data_num:
            n_samples = min([para.batch_size, data_num - low_pointer])
            if para.src_is_inf:
                query, query_len = padding_batch(
                    copy_list(test_data.informal_input[low_pointer:low_pointer + n_samples]))
                res = [nmt.fm_id_copy, nmt.fm_attn_copy]
                pred_id, attn = nmt.generate_informal2formal(sess, np.array(query), np.array(query_len),
                                                             batch_size=n_samples, res=res)
                indexs = nmt.parse_output(pred_id[:, 0, 1:], with_unk=copy)
            else:
                query, query_len = padding_batch(copy_list(test_data.formal_input[low_pointer:low_pointer + n_samples]))
                res = [nmt.inf_id_copy, nmt.inf_attn_copy]
                pred_id, attn = nmt.generate_formal2informal(sess, np.array(query), np.array(query_len),
                                                             batch_size=n_samples, res=res)
                indexs = nmt.parse_output(pred_id[:, 0, 1:], with_unk=copy)
            i += 1
            if copy:
                for j in range(0, len(indexs)):
                    copy_result = []
                    copy_indices = []
                    for i in range(0, len(indexs[j])):
                        if indexs[j][i] == nmt.UNK_token:
                            weights = attn[j, 0, i, :][:query_len[j]]
                            src_indice = np.argmax(weights, axis=0)
                            copy_indices.append((i, src_indice))
                            if query[j][src_indice] != nmt.UNK_token:
                                copy_result.append(nmt.UNK_token)
                            else:
                                copy_result.append(nmt.UNK_token)
                        else:
                            copy_result.append(indexs[j][i])
                    words = test_data.indices2sentence(copy_result, join=False)
                    if '<UNK>' in words:
                        print("src:", ' '.join(test_data.inf_src_tok[low_pointer + j]))
                        print('trans:', ' '.join(words))
                    replace_list = []
                    for indice in copy_indices:
                        if words[indice[0]] == '<UNK>':
                            if para.src_is_inf:
                                words[indice[0]] = test_data.inf_src_tok[low_pointer + j][indice[1]]
                                replace_list.append(test_data.inf_src_tok[low_pointer + j][indice[1]])
                            else:
                                words[indice[0]] = test_data.fm_src_tok[low_pointer + j][indice[1]]
                                replace_list.append(test_data.fm_src_tok[low_pointer + j][indice[1]])
                    if len(replace_list) > 0:
                        print("replace list:", ' '.join(replace_list))
                        print("final trans:", ' '.join(words))
                    output.append(' '.join([w for w in words if w != '<SOS>']))
            else:
                for one_sen_index in indexs:
                    output.append(test_data.indices2sentence(one_sen_index))
            low_pointer += n_samples
    if save_bpe_result:
        with open(para.output_path+'.bpe', 'w', encoding='utf-8') as fw:
            for s in output:
                fw.write(s + '\n')
    with open(para.output_path, 'w', encoding='utf-8') as fw:
        for s in output:
            fw.write(s.replace('@@ ','') + '\n')


def get_loss(loss_dict, target_losses):
    r = 0
    for l in target_losses:
        r += loss_dict[l]
    return r


def train_onehotkey_with_multi_datasets(parameters=None):
    if parameters is None:
        parameters = train_parameters()
    train_data = [pickle.load(open(path, 'rb')) for path in parameters.train_pkl_path]
    # val_data=pickle.load(open(parameters.val_pkl_path, 'rb'))
    inf2fm_bleu_data = pickle.load(open(parameters.inf2fm_bleu_val_path, 'rb'))
    fm2inf_bleu_data = pickle.load(open(parameters.fm2inf_bleu_val_path, 'rb'))
    print('build graph')
    embedding,vocab_hash=load_word_embedding(embedding_path)
    nmt = seq2seq(vocab_size=len(train_data[0].embedding),
                  sos_token=train_data[0].SOS_token, eos_token=train_data[0].EOS_token,
                  unk_token=train_data[0].UNK_token, struct_para=parameters,
                  vocab_hash=vocab_hash)
    best_step=0
    with nmt.graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=nmt.graph)
        saver = tf.train.Saver(max_to_keep=parameters.max_to_keep)
        print('init graph')
        total_batch = 0
        best_loss = 1000000
        last_improved_step = 0
        if parameters.continue_train:
            best_model_path = None
            with open(parameters.previous_save_dir + 'checkpoint', 'r', encoding='utf-8') as f:
                for line in f:
                    strs = line.strip().split(': ')
                    best_model_path = strs[1][1:-1]
                    total_batch = int(best_model_path.split('.')[1])
                    break
            saver.restore(sess, parameters.previous_save_dir + best_model_path)
            last_improved_step = total_batch
            assert 'inf2fm_bleu' in parameters.compared_loss or 'fm2inf_bleu' in parameters.compared_loss, 'compared loss is wrong'
            assert not (
                    'inf2fm_bleu' in parameters.compared_loss and 'fm2inf_bleu' in parameters.compared_loss), 'compared loss is wrong'
            if 'inf2fm_bleu' in parameters.compared_loss:
                val_data = inf2fm_bleu_data
                val_loss = nmt.evaluate(sess,
                                        evalaute_step=[item for item in parameters.eval_step if item != 'fm2inf_bleu'],
                                        data_num=len(val_data.informal_input),
                                        all_inf_input=val_data.informal_input,
                                        all_fm_input=val_data.formal_input,
                                        all_rm_input=val_data.all_sentences,
                                        all_tgt_inf_input=val_data.tgt_informal_input,
                                        all_tgt_inf_output=val_data.tgt_informal_output,
                                        all_tgt_fm_input=val_data.tgt_formal_input,
                                        all_tgt_fm_output=val_data.tgt_formal_output,
                                        inf2fm_val_data=inf2fm_bleu_data,
                                        fm2inf_val_data=fm2inf_bleu_data,
                                        batch_size=parameters.batch_size)
                print(val_loss)
            else:  # if 'fm2inf_bleu' in parameters.compared_loss:
                val_data = fm2inf_bleu_data
                val_loss = nmt.evaluate(sess,
                                        evalaute_step=[item for item in parameters.eval_step if item != 'inf2fm_bleu'],
                                        data_num=len(val_data.informal_input),
                                        all_inf_input=val_data.informal_input,
                                        all_fm_input=val_data.formal_input,
                                        all_rm_input=val_data.all_sentences,
                                        all_tgt_inf_input=val_data.tgt_informal_input,
                                        all_tgt_inf_output=val_data.tgt_informal_output,
                                        all_tgt_fm_input=val_data.tgt_formal_input,
                                        all_tgt_fm_output=val_data.tgt_formal_output,
                                        inf2fm_val_data=inf2fm_bleu_data,
                                        fm2inf_val_data=fm2inf_bleu_data,
                                        batch_size=parameters.batch_size)
                print(val_loss)
            best_loss = get_loss(val_loss, parameters.compared_loss)
            print(best_loss)
        else:
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(nmt.init_embeddings, feed_dict={nmt.embedding_source_pl: train_data[0].embedding})
        low_pointer = [0] * len(parameters.train_pkl_path)
        train_loss = [0] * len(parameters.train_pkl_path)
        train_data_num = [len(one_set.informal_input) for one_set in train_data]
        epoch = [1] * len(parameters.train_pkl_path)
        print('start train')
        log = open(parameters.save_dir + 'log.txt', 'w', encoding='utf-8')
        # todo:temp solution,set inf2fm_bleu_data as val_data
        val_data = inf2fm_bleu_data
        def suffle_data(data,data_num):
            print('suffle data')
            indices = np.random.permutation(np.arange(data_num))
            data.informal_input = [data.informal_input[idx] for idx in indices]
            data.formal_input = [data.formal_input[idx] for idx in indices]
            data.inf_src_tok = [data.inf_src_tok[idx] for idx in indices]
            data.fm_src_tok = [data.fm_src_tok[idx] for idx in indices]
            data.tgt_informal_input = [data.tgt_informal_input[idx] for idx in indices]
            data.tgt_formal_input = [data.tgt_formal_input[idx] for idx in indices]
            data.tgt_informal_output = [data.tgt_informal_output[idx] for idx in indices]
            data.tgt_formal_output = [data.tgt_formal_output[idx] for idx in indices]
        for d,num in zip(train_data,train_data_num):
            #suffle_data(d,num)
            pass
        while total_batch < parameters.max_step:
            for i in range(0, len(train_data)):
                n_samples = min(
                    [parameters.batch_size * parameters.train_upweight[i], train_data_num[i] - low_pointer[i]])
                inf_input, inf_input_len = padding_batch(
                    copy_list(train_data[i].informal_input[low_pointer[i]:low_pointer[i] + n_samples]))
                tgt_fm_input, tgt_fm_len = padding_batch(
                    copy_list(train_data[i].tgt_formal_input[low_pointer[i]:low_pointer[i] + n_samples]))
                tgt_fm_output, _ = padding_batch(
                    copy_list(train_data[i].tgt_formal_output[low_pointer[i]:low_pointer[i] + n_samples]))
                fm_input, fm_input_len = padding_batch(
                    copy_list(train_data[i].formal_input[low_pointer[i]:low_pointer[i] + n_samples]))
                tgt_inf_input, tgt_inf_len = padding_batch(
                    copy_list(train_data[i].tgt_informal_input[low_pointer[i]:low_pointer[i] + n_samples]))
                tgt_inf_output, _ = padding_batch(
                    copy_list(train_data[i].tgt_informal_output[low_pointer[i]:low_pointer[i] + n_samples]))
                idx = [random.randint(0, len(train_data[i].all_sentences) - 1) for j in range(0, n_samples)]
                rm_input = [train_data[i].all_sentences[j] for j in idx]
                rm_input, rm_input_len = padding_batch(copy_list(rm_input))
                train_loss[i] = nmt.train(sess, train_step=parameters.train_step[i],
                                          inf_input=inf_input, inf_input_len=inf_input_len,
                                          fm_input=fm_input, fm_input_len=fm_input_len,
                                          rm_input=rm_input, rm_input_len=rm_input_len,
                                          tgt_inf_input=tgt_inf_input, tgt_inf_output=tgt_inf_output,
                                          tgt_inf_len=tgt_inf_len,
                                          tgt_fm_input=tgt_fm_input, tgt_fm_output=tgt_fm_output,
                                          tgt_fm_len=tgt_fm_len,
                                          batch_size=n_samples,
                                          apply_sample_weight=parameters.apply_sample_weight[i],
                                          dataset_weight=parameters.dataset_lr_weight[i])
                low_pointer[i] += n_samples
                if low_pointer[i] >= train_data_num[i]:
                    low_pointer[i] = 0
                    # shuffle:
                    print('dataset ' + str(i) + ': epoch' + str(epoch[i]) + '  ended')
                    epoch[i] += 1
                    #suffle_data(train_data[i],train_data_num[i])
            total_batch += 1
            if total_batch % parameters.batches_per_evaluation == 0:
                val_loss = nmt.evaluate(sess, evalaute_step=parameters.eval_step,
                                        data_num=len(val_data.informal_input),
                                        all_inf_input=val_data.informal_input,
                                        all_fm_input=val_data.formal_input,
                                        all_rm_input=val_data.all_sentences,
                                        all_tgt_inf_input=val_data.tgt_informal_input,
                                        all_tgt_inf_output=val_data.tgt_informal_output,
                                        all_tgt_fm_input=val_data.tgt_formal_input,
                                        all_tgt_fm_output=val_data.tgt_formal_output,
                                        inf2fm_val_data=inf2fm_bleu_data,
                                        fm2inf_val_data=fm2inf_bleu_data,
                                        batch_size=parameters.batch_size)
                print("batch_num: {0}".format(total_batch))
                print(train_loss)
                print(val_loss)
                log.write("batch_num: {0}".format(total_batch) + '\n')
                log.write(str(train_loss) + '\n')
                log.write(str(val_loss) + '\n')
                if get_loss(val_loss, parameters.compared_loss) < best_loss:
                    best_loss = get_loss(val_loss, parameters.compared_loss)
                    saver.save(sess, parameters.save_dir + 'best.{0}.model'.format(total_batch))
                    last_improved_step = total_batch
                    best_step=total_batch
                else:  # learning_rate_decay
                    if total_batch - last_improved_step > parameters.early_stop_num * parameters.batches_per_evaluation:
                        print('early stop at', total_batch)
                        log.write('early stop at {0}'.format(total_batch) + '\n')
                        break
                    if total_batch - last_improved_step >= parameters.batches_per_evaluation * parameters.lr_decay_freq:
                        if (total_batch - last_improved_step) % (
                                parameters.batches_per_evaluation * parameters.lr_decay_freq) == 0:
                            log.write('lr decay {0}'.format(nmt.lr) + '\n')
                            print('lr decay {0}'.format(nmt.lr))
                            nmt.learning_rate_decay(sess=sess)
                            last_improved_step = total_batch
        sess.close()
        log.close()
        return best_step, best_loss
