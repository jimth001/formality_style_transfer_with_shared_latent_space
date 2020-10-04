# coding=utf-8

import tensorflow as tf
import nltk
import pickle
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.contrib.seq2seq import sequence_loss
import random
from nltk.translate.bleu_score import corpus_bleu
from sls_settings_v2_EM import *
from gpt_components.model import Decoder
import gpt_components.beamsearch as beamsearch
import gpt_components.bpe_encoder as bpe_encoder
from gpt_components.model import Encoder,default_hparams
import json
import tensorflow.contrib.slim as slim
tf.logging.set_verbosity (tf.logging.INFO)

bpe_config_path='./models/117M'

class prepare_data():
    def __init__(self, informal_file, formal_file, to_lower_case=False, ref_list=None, bpe_config_path=bpe_config_path):
        self.informal_file = informal_file
        self.formal_file = formal_file
        self.to_lower_case = to_lower_case
        self.text_enc = bpe_encoder.get_encoder(bpe_config_path)
        self.sos_id = self.text_enc.encode('\r')[0]
        self.eos_id = self.text_enc.encode('\n')[0]
        self.UNK_token = self.text_enc.encode('\r')[0]
        self.sep_id = self.text_enc.encode('\t')[0]
        self.informal_input, self.formal_input = self.process_source_file()
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
                gen.append(nltk.word_tokenize(line.strip()))
            weight = [1.0 / ngrams] * ngrams
            return corpus_bleu(self.refs, gen, weights=weight)
        else:
            raise ValueError('this dataset has no reference')

    def load_origin_ref_data(self, ref_path):
        data = []
        with open(ref_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(nltk.word_tokenize(line.strip()))
        return data

    def process_source_file(self):
        inf_source = []
        with open(self.informal_file, 'r', encoding='utf-8') as f:
            for line in f:
                if self.to_lower_case:
                    line = line.lower()
                inf_source.append(self.text_enc.encode(" ".join(nltk.word_tokenize(line.strip())))+[self.sep_id])
        fm_source = []
        with open(self.formal_file, 'r', encoding='utf-8') as f:
            for line in f:
                if self.to_lower_case:
                    line = line.lower()
                fm_source.append(self.text_enc.encode(" ".join(nltk.word_tokenize(line.strip())))+[self.sep_id])
        return inf_source, fm_source

    def process_target_file(self):
        inf_target_input=[]
        inf_target_output=[]
        with open(self.informal_file, 'r', encoding='utf-8') as f:
            for line in f:
                if self.to_lower_case:
                    line = line.lower()
                indexs=self.text_enc.encode(" ".join(nltk.word_tokenize(line.strip())))
                inf_target_input.append([self.sos_id]+copy_list(indexs))
                inf_target_output.append(indexs+[self.eos_id])
        fm_target_input=[]
        fm_target_output=[]
        with open(self.formal_file, 'r', encoding='utf-8') as f:
            for line in f:
                if self.to_lower_case:
                    line = line.lower()
                indexs = self.text_enc.encode(" ".join(nltk.word_tokenize(line.strip())))
                fm_target_input.append([self.sos_id]+copy_list(indexs))
                fm_target_output.append(indexs+[self.eos_id])
        return inf_target_input,inf_target_output,fm_target_input,fm_target_output

class uni_sls:
    def __init__(self, struct_para, config_path):
        self.lr = struct_para.learning_rate
        self.max_length = struct_para.max_length
        self.config_path=config_path
        self.hparam = default_hparams()
        self.struct_para=struct_para
        with open(os.path.join(self.config_path, 'hparams.json')) as f:
            self.hparam.override_from_dict(json.load(f))
        self.text_enc = bpe_encoder.get_encoder(self.config_path)
        self.sos_id = self.text_enc.encode('\r')[0]
        self.eos_id = self.text_enc.encode('\n')[0]
        self.UNK_token = self.text_enc.encode('\r')[0]
        self.sep_id = self.text_enc.encode('\t')[0]
        self.max_decode_length=60
        self.beam_size = struct_para.beam_size
        self.keep_prob_value = struct_para.keep_prob
        self.lr_decay_rate = struct_para.lr_decay_rate
        self.denoising_swap_rate = struct_para.denoising_swap_rate
        self.max_swap_times = struct_para.max_swap_times
        self.max_lm_masked_num = struct_para.max_lm_masked_num
        self.max_lm_masked_rate = struct_para.max_lm_masked_rate
        self.graph=tf.Graph()
        self.vars_for_infer = []
        self.vars_for_train = []
        self.mini_batch=32

    def average_gradients(self,tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = [tf.expand_dims(g, 0) for g, _ in grad_and_vars]
            grads = tf.concat(grads, 0)
            grad = tf.reduce_mean(grads, 0)
            grad_and_var = (grad, grad_and_vars[0][1])
            # [(grad0, var0),(grad1, var1),...]
            average_grads.append(grad_and_var)
        return average_grads

    def grad_accum(self,learning_rate,loss):
        opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        tower_grads = []
        grads = opt.compute_gradients(loss)
        tvs = tf.trainable_variables()
        accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False)
                           for tv in
                           tvs]
        zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars]
        accum_grad_ops = [accum_vars[j].assign_add(gv[0]) for j, gv in
                               enumerate(grads) if gv[0] is not None]
        tower_grads.append([(accum_vars[j], gv[1]) for j, gv in enumerate(grads)])
        grads = self.average_gradients(tower_grads)
        with tf.device('/gpu:0'):
            accum_steps = tf.placeholder(tf.float32, [], name='accum_stpes')
            train_step = opt.apply_gradients([(g / accum_steps, v) for g, v in grads])
        return zero_ops,accum_grad_ops,train_step,accum_steps

    def build_model_fn(self):
        struct_para=self.struct_para
        with tf.variable_scope('placeholder_and_embedding'):
            self.src = tf.placeholder(name='gpt_components', shape=(None, None), dtype=tf.int32)
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
            self.sample_weights = tf.placeholder(name='sample_weights', dtype=tf.float32, shape=(None,))
            # src_max_len = tf.reduce_max(self.src_len)
            self.var_lr = tf.Variable(float(struct_para.learning_rate), trainable=False, dtype=tf.float32)
            self.lr_pl = tf.placeholder(name='lr_pl', shape=(), dtype=tf.float32)
            self.learning_rate_decay_op = self.var_lr.assign(self.lr_pl)
            self.match_lr = struct_para.match_lr
        with tf.variable_scope("generate"):
            # def enc and dec
            self.encoder = Encoder(scope="encoder", hparam=self.hparam)
            self.informal_decoder = Decoder(scope='inf_dec', hparam=self.hparam)
            self.formal_decoder = Decoder(scope='fm_dec', hparam=self.hparam)
            # encoding:
            src_en_out, src_en_state = self.encoder.encode_which_outputs_all_layer_h(self.src, self.src_len)
            mch1_en_out, mch1_en_state = self.encoder.encode_which_outputs_all_layer_h(self.mch1, self.mch1_len)
            mch2_en_out, mch2_en_state = self.encoder.encode_which_outputs_all_layer_h(self.mch2, self.mch2_len)
            rm_en_out, rm_en_state = self.encoder.encode_which_outputs_all_layer_h(self.random_selected_sen,
                                                                                   self.random_selected_sen_len)
            #mch1_en_state = tf.concat(mch1_en_state, axis=1)
            #mch2_en_state = tf.concat(mch2_en_state, axis=1)
            #rm_en_state = tf.concat(rm_en_state, axis=1)
            mch1_en_state = mch1_en_state[-1]
            mch2_en_state = mch2_en_state[-1]
            rm_en_state = rm_en_state[-1]
            # src2fm dec train
            all_logits = self.formal_decoder.decode_all(self.hparam, tokens=self.tgt_formal_input, past=src_en_out)[
                'logits']
            batch_max_seq_len = tf.shape(self.tgt_formal_input)[1]
            target_mask = tf.sequence_mask(self.tgt_formal_len, maxlen=batch_max_seq_len, dtype=tf.float32)
            self.src2formal_cost = sequence_loss(logits=all_logits, targets=self.tgt_formal_target,
                                                 weights=target_mask)
            if struct_para.optimizer == 'sgd':
                self.src2fm_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.var_lr).minimize(
                    self.src2formal_cost)
            else:
                self.src2fm_optimizer = tf.train.AdamOptimizer(learning_rate=self.var_lr).minimize(
                    self.src2formal_cost)
            # src2inf dec train
            all_logits = \
                self.informal_decoder.decode_all(self.hparam, tokens=self.tgt_informal_input, past=src_en_out)[
                    'logits']
            batch_max_seq_len = tf.shape(self.tgt_informal_input)[1]
            target_mask = tf.sequence_mask(self.tgt_informal_len, maxlen=batch_max_seq_len, dtype=tf.float32)
            self.src2informal_cost = sequence_loss(logits=all_logits, targets=self.tgt_informal_target,
                                                   weights=target_mask)
            if struct_para.optimizer == 'sgd':
                self.src2inf_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.var_lr).minimize(
                    self.src2informal_cost)
            else:
                self.src2inf_optimizer = tf.train.AdamOptimizer(learning_rate=self.var_lr).minimize(
                    self.src2informal_cost)
            # src2fm gen
            init_seq = tf.fill(dims=(self.batch_size, 1), value=self.sos_id)
            self.fm_seqs, self.fm_scores = beamsearch.create_inference_graph(init_seqs=init_seq, state=src_en_out,
                                                                             step_fn=self.formal_decoder.decode_one_step,
                                                                             hparams=self.hparam,
                                                                             decode_length=self.max_decode_length,
                                                                             batch_size=self.batch_size,
                                                                             beam_size=self.beam_size,
                                                                             decode_alpha=struct_para.decode_alpha,
                                                                             eos_id=self.eos_id,
                                                                             ensemble=False, concat_state_dim=None)
            # src2inf gen
            init_seq = tf.fill(dims=(self.batch_size, 1), value=self.sos_id)
            self.inf_seqs, self.inf_scores = beamsearch.create_inference_graph(init_seqs=init_seq, state=src_en_out,
                                                                               step_fn=self.informal_decoder.decode_one_step,
                                                                               hparams=self.hparam,
                                                                               decode_length=self.max_decode_length,
                                                                               batch_size=self.batch_size,
                                                                               beam_size=self.beam_size,
                                                                               decode_alpha=struct_para.decode_alpha,
                                                                               eos_id=self.eos_id,
                                                                               ensemble=False,
                                                                               concat_state_dim=None)
        with tf.variable_scope('matching'):
            q1_state = tf.concat((mch1_en_state, mch2_en_state, rm_en_state), axis=0)
            q2_state = tf.concat((mch2_en_state, rm_en_state, mch1_en_state), axis=0)
            #mch_w=tf.get_variable(name='matching_w', shape=(q1_state.shape[1], 16, q1_state.shape[1]),dtype=tf.float32)
            #mch_fea=tf.tensordot(q1_state, mch_w, axes=[[1], [0]])
            #mch_vector=tf.matmul(mch_fea,tf.expand_dims(q2_state,axis=2))[:,:,-1]
            match_fea1 = tf.abs(tf.subtract(q1_state, q2_state))
            match_fea2 = tf.matmul(tf.expand_dims(q1_state, axis=1),
                                   tf.expand_dims(q2_state, axis=2))[:, :, -1]
            match_vector = tf.concat((match_fea1, match_fea2), axis=1)
            hidden_layer1 = tf.layers.dense(match_vector, units=24, activation=tf.nn.relu)
            hidden_layer2 = tf.layers.dense(match_vector, units=24, activation=tf.nn.sigmoid)
            mch_vector=tf.concat((hidden_layer1, hidden_layer2), axis=1)
            logits = tf.layers.dense(mch_vector, units=2)
            self.loss_match = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.match_label, logits=logits))
            self.loss_mse = tf.reduce_mean(tf.square(tf.subtract(q1_state, q2_state)),
                                           axis=1)
            float_label = tf.cast(self.match_label, dtype=tf.float32)
            self.loss_mse = tf.reduce_sum(tf.multiply(self.loss_mse, float_label)) / tf.cast(self.batch_size,
                                                                                             dtype=tf.float32)
            self.loss_vector_length = tf.reduce_mean(tf.norm(q1_state, axis=1)) + tf.reduce_mean(
                tf.norm(q2_state, axis=1))
            self.total_align_loss = struct_para.match_mse_lam * self.loss_mse + struct_para.match_cls_lam * self.loss_match
            if struct_para.optimizer == 'sgd':
                self.matching_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.match_lr).minimize(
                    self.total_align_loss)
            else:
                self.matching_optimizer = tf.train.AdamOptimizer(learning_rate=self.match_lr).minimize(
                    self.total_align_loss)

    def learning_rate_decay(self, sess):
        self.lr *= self.lr_decay_rate
        sess.run([self.learning_rate_decay_op], feed_dict={self.lr_pl: self.lr})

    def mini_batch_processing_helper(self,sess,train_step,
                                 losses_to_show,batch_feed_dict,global_feed_dict,batch_size,
                                 is_train,run_options=None):
        feed_dict = {}
        for key in global_feed_dict.keys():
            feed_dict[key] = global_feed_dict[key]
        for key in batch_feed_dict.keys():
            feed_dict[key] = batch_feed_dict[key]
        if is_train:
            result = sess.run([train_step] + losses_to_show, feed_dict=feed_dict, options=run_options)
            losses = result[1:]
        else:
            losses = sess.run(losses_to_show, feed_dict=feed_dict, options=run_options)
        return losses

    def mini_batch_generating_helper(self,sess,res,batch_feed_dict,global_feed_dict,
                                     batch_size,run_options=None):
        turns = int(batch_size / self.mini_batch)
        if batch_size % self.mini_batch != 0:
            turns += 1
        all_results=[]
        for t in range(0, turns):
            mini_feed_dict = {}
            for key in global_feed_dict.keys():
                mini_feed_dict[key] = global_feed_dict[key]
            for key in batch_feed_dict.keys():
                mini_feed_dict[key] = batch_feed_dict[key][t * self.mini_batch:(t + 1) * self.mini_batch]
            gen=sess.run(res,feed_dict=mini_feed_dict,options=run_options)[0]
            gen = self.parse_output(gen[:, 0, 1:])
            for s in gen:
                all_results.append(s)
        return all_results

    def train(self, sess, train_step, inf_input, inf_input_len, fm_input, fm_input_len, rm_input, rm_input_len,
              tgt_inf_input, tgt_inf_output, tgt_inf_len, tgt_fm_input, tgt_fm_output, tgt_fm_len, batch_size,
              apply_sample_weight=0, dataset_weight=1):
        feed_dict = {}
        # print(batch_size)
        losses = {'total_aligned_loss': 0.0, 'loss_match': 0.0, 'loss_mse': 0.0, 'loss_vector_length': 0.0,
                  'inf2fm_loss': 0.0, 'fm2inf_loss': 0.0, 'fm2fm_loss': 0.0, 'inf2inf_loss': 0.0,}
        basic_sample_weights = [1.0] * batch_size
        inf_sample_weights = basic_sample_weights
        fm_sample_weights = basic_sample_weights
        if apply_sample_weight != 0:
            pass
        inf_sample_weights = [i * dataset_weight for i in inf_sample_weights]
        fm_sample_weights = [i * dataset_weight for i in fm_sample_weights]
        global_fd = {self.batch_size:batch_size,self.keep_prob:self.keep_prob_value}
        if 'match' in train_step:
            feed_dict.clear()
            feed_dict[self.match_label] = [int(1)] * batch_size + [int(0)] * (batch_size * 2)
            feed_dict[self.mch1] = inf_input
            feed_dict[self.mch1_len] = inf_input_len
            feed_dict[self.mch2] = fm_input
            feed_dict[self.mch2_len] = fm_input_len
            feed_dict[self.random_selected_sen] = rm_input
            feed_dict[self.random_selected_sen_len] = rm_input_len
            losses_to_show = [self.total_align_loss, self.loss_match, self.loss_mse, self.loss_vector_length]
            returned_losses=self.mini_batch_processing_helper(sess=sess,train_step=self.matching_optimizer,
                                              losses_to_show=losses_to_show,batch_feed_dict=feed_dict,
                                              global_feed_dict=global_fd,batch_size=batch_size,is_train=True)
            t_l, mch_l, mse_l, len_l = returned_losses
            losses['total_aligned_loss'] += t_l
            losses['loss_match'] += mch_l
            losses['loss_mse'] += mse_l
            losses['loss_vector_length'] += len_l
        if 'fm2fm' in train_step:
            feed_dict.clear()
            feed_dict[self.src] = fm_input
            feed_dict[self.src_len] = fm_input_len
            feed_dict[self.tgt_formal_input] = tgt_fm_input
            feed_dict[self.tgt_formal_target] = tgt_fm_output
            feed_dict[self.tgt_formal_len] = tgt_fm_len
            feed_dict[self.sample_weights] = fm_sample_weights
            losses_to_show=[self.src2formal_cost]
            returned_losses = self.mini_batch_processing_helper(sess=sess,
                                                                train_step=self.src2fm_optimizer,
                                                                losses_to_show=losses_to_show,
                                                                batch_feed_dict=feed_dict,
                                                                global_feed_dict=global_fd, batch_size=batch_size,
                                                                is_train=True)
            l = returned_losses[0]
            losses['fm2fm_loss'] = l
        if 'inf2inf' in train_step:
            feed_dict.clear()
            feed_dict[self.src] = inf_input
            feed_dict[self.src_len] = inf_input_len
            feed_dict[self.tgt_informal_input] = tgt_inf_input
            feed_dict[self.tgt_informal_target] = tgt_inf_output
            feed_dict[self.tgt_informal_len] = tgt_inf_len
            feed_dict[self.sample_weights] = inf_sample_weights
            losses_to_show=[self.src2informal_cost]
            returned_losses = self.mini_batch_processing_helper(sess=sess, train_step=self.src2inf_optimizer,
                                                                losses_to_show=losses_to_show,
                                                                batch_feed_dict=feed_dict,
                                                                global_feed_dict=global_fd, batch_size=batch_size,
                                                                is_train=True)
            l = returned_losses[0]
            losses['inf2inf_loss'] = l
        if 'inf2fm' in train_step:
            feed_dict.clear()
            feed_dict[self.src] = inf_input
            feed_dict[self.src_len] = inf_input_len
            feed_dict[self.tgt_formal_input] = tgt_fm_input
            feed_dict[self.tgt_formal_target] = tgt_fm_output
            feed_dict[self.tgt_formal_len] = tgt_fm_len
            feed_dict[self.sample_weights] = fm_sample_weights
            losses_to_show=[self.src2formal_cost]
            returned_losses = self.mini_batch_processing_helper(sess=sess, train_step=self.src2fm_optimizer,
                                                                losses_to_show=losses_to_show,
                                                                batch_feed_dict=feed_dict,
                                                                global_feed_dict=global_fd, batch_size=batch_size,
                                                                is_train=True)
            l = returned_losses[0]
            losses['inf2fm_loss'] = l
        if 'fm2inf' in train_step:
            feed_dict.clear()
            feed_dict[self.src] = fm_input
            feed_dict[self.src_len] = fm_input_len
            feed_dict[self.tgt_informal_input] = tgt_inf_input
            feed_dict[self.tgt_informal_target] = tgt_inf_output
            feed_dict[self.tgt_informal_len] = tgt_inf_len
            feed_dict[self.sample_weights] = inf_sample_weights
            losses_to_show = [self.src2informal_cost]
            returned_losses = self.mini_batch_processing_helper(sess=sess, train_step=self.src2inf_optimizer,
                                                                losses_to_show=losses_to_show,
                                                                batch_feed_dict=feed_dict,
                                                                global_feed_dict=global_fd, batch_size=batch_size,
                                                                is_train=True)
            l = returned_losses[0]
            losses['fm2inf_loss'] = l
        return losses

    def evaluate(self, sess, evalaute_step, data_num, all_inf_input, all_fm_input, all_rm_input,
                 all_tgt_inf_input, all_tgt_inf_output,
                 all_tgt_fm_input, all_tgt_fm_output,
                 inf2fm_val_data:prepare_data=None, fm2inf_val_data:prepare_data=None,
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
            global_fd = {self.batch_size: n_samples, self.keep_prob: 1.0}
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
                feed_dict.clear()
                feed_dict[self.match_label] = [int(1)] * n_samples + [int(0)] * (n_samples * 2)
                feed_dict[self.mch1] = inf_input
                feed_dict[self.mch1_len] = inf_input_len
                feed_dict[self.mch2] = fm_input
                feed_dict[self.mch2_len] = fm_input_len
                feed_dict[self.random_selected_sen] = rm_input
                feed_dict[self.random_selected_sen_len] = rm_input_len
                losses_to_show = [self.total_align_loss, self.loss_match, self.loss_mse, self.loss_vector_length]
                returned_losses = self.mini_batch_processing_helper(sess=sess, train_step=None,
                                                                    losses_to_show=losses_to_show,
                                                                    batch_feed_dict=feed_dict,
                                                                    global_feed_dict=global_fd, batch_size=n_samples,
                                                                    is_train=False)
                t_l, mch_l, mse_l, len_l = returned_losses
                losses['total_aligned_loss'] += t_l * n_samples
                losses['loss_match'] += mch_l * n_samples
                losses['loss_mse'] += mse_l * n_samples
                losses['loss_vector_length'] += len_l * n_samples
            if 'fm2fm' in evalaute_step:
                feed_dict.clear()
                feed_dict[self.src] = fm_input
                feed_dict[self.src_len] = fm_input_len
                feed_dict[self.tgt_formal_input] = tgt_fm_input
                feed_dict[self.tgt_formal_target] = tgt_fm_output
                feed_dict[self.tgt_formal_len] = tgt_fm_len
                feed_dict[self.sample_weights] = basic_sample_weights
                losses_to_show = [self.src2formal_cost]
                returned_losses = self.mini_batch_processing_helper(sess=sess, train_step=None,
                                                                    losses_to_show=losses_to_show,
                                                                    batch_feed_dict=feed_dict,
                                                                    global_feed_dict=global_fd, batch_size=n_samples,
                                                                    is_train=False)
                l = returned_losses[0]
                losses['fm2fm_loss'] += l * n_samples
            if 'inf2inf' in evalaute_step:
                feed_dict.clear()
                feed_dict[self.src] = inf_input
                feed_dict[self.src_len] = inf_input_len
                feed_dict[self.tgt_informal_input] = tgt_inf_input
                feed_dict[self.tgt_informal_target] = tgt_inf_output
                feed_dict[self.tgt_informal_len] = tgt_inf_len
                feed_dict[self.sample_weights] = basic_sample_weights
                losses_to_show = [self.src2informal_cost]
                returned_losses = self.mini_batch_processing_helper(sess=sess, train_step=None,
                                                                    losses_to_show=losses_to_show,
                                                                    batch_feed_dict=feed_dict,
                                                                    global_feed_dict=global_fd, batch_size=n_samples,
                                                                    is_train=False)
                l = returned_losses[0]
                losses['inf2inf_loss'] += l * n_samples
            if 'inf2fm' in evalaute_step:
                feed_dict.clear()
                feed_dict[self.src] = inf_input
                feed_dict[self.src_len] = inf_input_len
                feed_dict[self.tgt_formal_input] = tgt_fm_input
                feed_dict[self.tgt_formal_target] = tgt_fm_output
                feed_dict[self.tgt_formal_len] = tgt_fm_len
                feed_dict[self.sample_weights] = basic_sample_weights
                losses_to_show = [self.src2formal_cost]
                returned_losses = self.mini_batch_processing_helper(sess=sess, train_step=None,
                                                                    losses_to_show=losses_to_show,
                                                                    batch_feed_dict=feed_dict,
                                                                    global_feed_dict=global_fd, batch_size=n_samples,
                                                                    is_train=False)
                l = returned_losses[0]
                losses['inf2fm_loss'] += l * n_samples
            if 'fm2inf' in evalaute_step:
                feed_dict.clear()
                feed_dict[self.src] = fm_input
                feed_dict[self.src_len] = fm_input_len
                feed_dict[self.tgt_informal_input] = tgt_inf_input
                feed_dict[self.tgt_informal_target] = tgt_inf_output
                feed_dict[self.tgt_informal_len] = tgt_inf_len
                feed_dict[self.sample_weights] = basic_sample_weights
                losses_to_show = [self.src2informal_cost]
                returned_losses = self.mini_batch_processing_helper(sess=sess, train_step=None,
                                                                    losses_to_show=losses_to_show,
                                                                    batch_feed_dict=feed_dict,
                                                                    global_feed_dict=global_fd, batch_size=n_samples,
                                                                    is_train=False)
                l = returned_losses[0]
                losses['fm2inf_loss'] += l * n_samples
            if 'inf2fm_bleu' in evalaute_step:
                res.clear()
                res.append(self.fm_seqs)
                feed_dict.clear()
                feed_dict[self.src] = inf_input
                feed_dict[self.src_len] = inf_input_len
                fm_gen=self.mini_batch_generating_helper(sess=sess,res=res,batch_feed_dict=feed_dict,
                                                  global_feed_dict=global_fd,batch_size=n_samples)
                for one_sen_index in fm_gen:
                    inf2fm_gen.append(inf2fm_val_data.text_enc.decode(one_sen_index))
            if 'fm2inf_bleu' in evalaute_step:
                res.clear()
                res.append(self.inf_seqs)
                feed_dict.clear()
                feed_dict[self.src] = fm_input
                feed_dict[self.src_len] = fm_input_len
                inf_gen = self.mini_batch_generating_helper(sess=sess, res=res, batch_feed_dict=feed_dict,
                                                           global_feed_dict=global_fd, batch_size=n_samples)
                for one_sen_index in inf_gen:
                    fm2inf_gen.append(fm2inf_val_data.text_enc.decode(one_sen_index))
            low += n_samples
        losses['total_aligned_loss'] /= (data_num)
        losses['loss_match'] /= data_num
        losses['loss_mse'] /= data_num
        losses['loss_vector_length'] /= data_num
        losses['fm2inf_loss'] /= data_num
        losses['inf2fm_loss'] /= data_num
        losses['fm2fm_loss'] /= data_num
        losses['inf2inf_loss'] /= data_num
        if 'inf2fm_bleu' in evalaute_step:
            losses['inf2fm_bleu'] = -inf2fm_val_data.test_bleu(inf2fm_gen)
        if 'fm2inf_bleu' in evalaute_step:
            losses['fm2inf_bleu'] = -fm2inf_val_data.test_bleu(fm2inf_gen)
        return losses

    def generate_informal2formal(self, sess, encoder_inputs, encoder_length, batch_size):
        """
        feed data to generate.
        :param sess: session
        :param encoder_inputs: encoder inputs
        :param encoder_length: encoder inputs sequence length, (batch=1, )
        :return:
        """
        #if encoder_inputs.ndim == 1:
        #    encoder_inputs = encoder_inputs.reshape((1, -1))
        #    encoder_length = encoder_length.reshape((1,))
        res=[self.fm_seqs]
        feed_dict={}
        feed_dict[self.src] = encoder_inputs
        feed_dict[self.src_len] = encoder_length
        global_fd = {self.batch_size: batch_size, self.keep_prob: 1.0}
        fm_gen = self.mini_batch_generating_helper(sess=sess, res=res, batch_feed_dict=feed_dict,
                                                   global_feed_dict=global_fd, batch_size=batch_size)
        inf2fm_gen=[]
        for one_sen_index in fm_gen:
            inf2fm_gen.append(self.text_enc.decode(one_sen_index))
        return inf2fm_gen

    def generate_formal2informal(self, sess, encoder_inputs, encoder_length, batch_size):
        """
        feed data to generate.
        :param sess: session
        :param encoder_inputs: encoder inputs
        :param encoder_length: encoder inputs sequence length, (batch=1, )
        :return:
        """
        #if encoder_inputs.ndim == 1:
        #    encoder_inputs = encoder_inputs.reshape((1, -1))
        #    encoder_length = encoder_length.reshape((1,))
        res = [self.inf_seqs]
        feed_dict = {}
        feed_dict[self.src] = encoder_inputs
        feed_dict[self.src_len] = encoder_length
        global_fd = {self.batch_size: batch_size, self.keep_prob: 1.0}
        inf_gen = self.mini_batch_generating_helper(sess=sess, res=res, batch_feed_dict=feed_dict,
                                                   global_feed_dict=global_fd, batch_size=batch_size)
        fm2inf_gen = []
        for one_sen_index in inf_gen:
            fm2inf_gen.append(self.text_enc.decode(one_sen_index))
        return fm2inf_gen

    def parse_output(self, token_indices, with_unk=False, is_infer=False, with_eos=False):
        res = []
        unk_counter = 0
        for one_sen in token_indices:
            sen = []
            for token in one_sen:
                if token != self.eos_id:  # end
                    sen.append(token)
                else:
                    break
            if with_eos:
                sen.append(self.eos_id)
            res.append(sen)
        if is_infer:
            print('unk num:', unk_counter)
        return res

    def create_session_init_and_print_all_trainable_vars(self, max_to_save, ori_gpt_model_path=None):
        # Print parameters
        with self.graph.as_default():
            all_weights = {v.name: v for v in tf.trainable_variables()}
            print(len(all_weights))
            total_size = 0
            for v_name in sorted(list(all_weights)):
                v = all_weights[v_name]
                tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                                str(v.shape).ljust(20))
                v_size = np.prod(np.array(v.shape.as_list())).tolist()
                total_size += v_size
            tf.logging.info("Total trainable variables size: %d", total_size)
            all_var_list = slim.get_variables_to_restore()
            for v in all_var_list:
                #print(v.name)
                if 'Adam' in v.name:
                    self.vars_for_train.append(v)
                elif 'beta' in v.name:
                    self.vars_for_train.append(v)
                elif v.name.startswith('parallel'):
                    pass
                else:
                    self.vars_for_infer.append(v)
            #print('-----------------------')
            if len(self.vars_for_infer) > 0:
                self.saver_infer = tf.train.Saver(self.vars_for_infer, max_to_keep=max_to_save)
            if len(self.vars_for_train) > 0:
                self.saver_train = tf.train.Saver(self.vars_for_train, max_to_keep=max_to_save)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(graph=self.graph, config=config)
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            restore_ops = []
            if ori_gpt_model_path is not None:
                ckpt = tf.train.latest_checkpoint(ori_gpt_model_path)
                tf.logging.info("Loading %s" % ckpt)
                var_list = tf.train.list_variables(ckpt)
                values = {}
                reader = tf.train.load_checkpoint(ckpt)
                for (name, shape) in var_list:
                    if not name.startswith('model/'):  # ignore global_step
                        continue
                    tensor = reader.get_tensor(name)
                    values[name] = tensor
                for v in self.vars_for_infer:
                    # print(v.name)
                    if v.name.startswith('matching'):
                        continue
                    tmp = '/'.join(v.name.split('/')[2:])
                    v_name = tmp.split(':')[0]
                    if v_name=="":
                        print(v.name)
                        continue
                    op = tf.assign(v, values[v_name])
                    restore_ops.append(op)
                sess.run(restore_ops)
            return sess

    def restore_model_and_init(self, sess, ckpt_for_infer, ckpt_for_train):
        with self.graph.as_default():
            if ckpt_for_infer is not None:
                ckpt = tf.train.latest_checkpoint(ckpt_for_infer)
                if ckpt is not None:
                    self.saver_infer.restore(sess, ckpt)
                    tf.logging.info('restored inferring params from %s',ckpt)
            if ckpt_for_train is not None:
                ckpt = tf.train.latest_checkpoint(ckpt_for_train)
                if ckpt is not None:
                    self.saver_train.restore(sess, ckpt)
                    tf.logging.info('restored training params from %s', ckpt)

    def save_model(self, sess, infer_ckpt_path, train_ckpt_path, step):
        with self.graph.as_default():
            if infer_ckpt_path is not None and len(self.vars_for_infer) > 0:
                self.saver_infer.save(sess, os.path.join(infer_ckpt_path,'model'), global_step=step)
            if train_ckpt_path is not None and len(self.vars_for_train) > 0:
                self.saver_train.save(sess, os.path.join(train_ckpt_path,'model'), global_step=step)

def preprocess():
    train_data = prepare_data(informal_train_path,
                              formal_train_path,
                              to_lower_case=False)
    pickle.dump(train_data, open(train_pkl_path, 'wb'), protocol=True)
    val_data = prepare_data(informal_val_path,
                            formal_val_path,
                            to_lower_case=False)
    pickle.dump(val_data, open(val_pkl_path, 'wb'), protocol=True)
    test_data = prepare_data(informal_test_path,
                             formal_test_path,
                             to_lower_case=False)
    pickle.dump(test_data, open(test_pkl_path, 'wb'), protocol=True)

def gen_val_data_for_bleu():
    inf2fm_bleu_fr = prepare_data(informal_file=informal_val_path,
                                    formal_file=formal_val_path,
                                    ref_list=informal_val_refs,
                                    to_lower_case=False)
    pickle.dump(inf2fm_bleu_fr, open(inf2fm_bleu_val_pkl_path, 'wb'), protocol=True)
    fm2inf_bleu_data=prepare_data(informal_file=fm2inf_bleu_inf_src,
                             formal_file=fm2inf_bleu_fm_src,
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

def generate_copy(para=None):
    if para is None:
        para = generate_parameters()
    test_data = pickle.load(open(para.input_path, 'rb'))
    nmt = uni_sls(struct_para=para, config_path=bpe_config_path)
    with nmt.graph.as_default():
        nmt.build_model_fn()
        sess=nmt.create_session_init_and_print_all_trainable_vars(max_to_save=2)
        nmt.restore_model_and_init(sess=sess,ckpt_for_infer=para.ckpt_for_infer,ckpt_for_train=None)
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
                gen = nmt.generate_informal2formal(sess, np.array(query), np.array(query_len),
                                                             batch_size=n_samples)
            else:
                query, query_len = padding_batch(copy_list(test_data.formal_input[low_pointer:low_pointer + n_samples]))
                gen = nmt.generate_formal2informal(sess, np.array(query), np.array(query_len),
                                                             batch_size=n_samples)
            i += 1
            for one_sen in gen:
                output.append(one_sen)
            low_pointer += n_samples
    with open(para.output_path, 'w', encoding='utf-8') as fw:
        for s in output:
            fw.write(s + '\n')

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
    nmt = uni_sls(struct_para=parameters, config_path=bpe_config_path)
    best_step=0
    with nmt.graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        nmt.build_model_fn()
        print('initing graph')
        if parameters.continue_train:
            sess = nmt.create_session_init_and_print_all_trainable_vars(max_to_save=2, ori_gpt_model_path=None)
        else:
            sess = nmt.create_session_init_and_print_all_trainable_vars(max_to_save=2, ori_gpt_model_path=bpe_config_path)
        print('init graph finished')
        total_batch = 0
        best_loss = 1000000
        last_improved_step = 0
        if parameters.continue_train:
            nmt.restore_model_and_init(sess,ckpt_for_infer=parameters.last_ckpt_for_infer,ckpt_for_train=parameters.last_ckpt_for_train)
        low_pointer = [0] * len(parameters.train_pkl_path)
        train_loss = [0] * len(parameters.train_pkl_path)
        train_data_num = [len(one_set.informal_input) for one_set in train_data]
        epoch = [1] * len(parameters.train_pkl_path)
        print('start train')
        log = open(parameters.ckpt_for_infer + 'log.txt', 'w', encoding='utf-8')
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
                #print("batch size:",n_samples)
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
                    print('dataset ' + str(i) + ': epoch' + str(epoch[i]) + '  ended')
                    epoch[i] += 1
                    # shuffle:
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
                    nmt.save_model(sess,parameters.ckpt_for_infer,parameters.ckpt_for_train,step=total_batch)
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
        sess.close()
        log.close()
        return best_step, best_loss

def cal_max_len(train_pkl_path):
    train_pkl = pickle.load(open(train_pkl_path, 'rb'))
    l=[]
    for x1,x2,x3,x4 in zip(train_pkl.formal_input,train_pkl.informal_input,
                                 train_pkl.tgt_formal_input,train_pkl.tgt_informal_input):
        l.append(max(len(x1),len(x2),len(x3),len(x4)))
    print(max(l))

def cut_data_by_len(train_pkl_path):
    train_pkl = pickle.load(open(train_pkl_path, 'rb'))
    new_fm_in=[]
    new_inf_in=[]
    new_tgt_fm_in=[]
    new_tgt_fm_out=[]
    new_tgt_inf_in=[]
    new_tgt_inf_out=[]
    record=[]
    for i in range(0,len(train_pkl.formal_input)):
        l=max(len(train_pkl.formal_input[i]),len(train_pkl.informal_input[i]),
              len(train_pkl.tgt_formal_input[i]),len(train_pkl.tgt_informal_input[i]))
        if l>60:
            record.append(l)
        else:
            new_fm_in.append(train_pkl.formal_input[i])
            new_inf_in.append(train_pkl.informal_input[i])
            new_tgt_fm_in.append(train_pkl.tgt_formal_input[i])
            new_tgt_fm_out.append(train_pkl.tgt_formal_output[i])
            new_tgt_inf_in.append(train_pkl.tgt_informal_input[i])
            new_tgt_inf_out.append(train_pkl.tgt_informal_output[i])
    new_all_sen=[s for s in train_pkl.all_sentences if len(s)<=60]
    train_pkl.formal_input=new_fm_in
    train_pkl.informal_input=new_inf_in
    train_pkl.tgt_formal_input=new_tgt_fm_in
    train_pkl.tgt_formal_output=new_tgt_fm_out
    train_pkl.tgt_informal_input=new_tgt_inf_in
    train_pkl.tgt_informal_output=new_tgt_inf_out
    train_pkl.all_sentences=new_all_sen
    pickle.dump(train_pkl, open(train_pkl_path+'.new', 'wb'), protocol=True)
    print(record)

if __name__ == '__main__':
    #cal_max_len(train_pkl_path)
    cut_data_by_len(train_pkl_path)
    print('all work has finished')
