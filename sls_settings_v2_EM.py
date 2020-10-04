import numpy as np
import os

MAX_LENGTH = np.int32(150)
exp_main_path='./new_exp_em/'
embedding_path='./new_exp_em/embedding/embedding.bpe.big.txt'
informal_train_path='./new_exp_em/informal.train.rule'
informal_val_path='./new_exp_em/informal.val.rule'
informal_test_path='./new_exp_em/informal.test.rule'
formal_train_path='./new_exp_em/formal.train.rule'
formal_val_path='./new_exp_em/formal.val.rule'
formal_test_path='./new_exp_em/formal.test.rule'
informal_val_refs=['./data/Entertainment_Music/tune/formal.ref0',
                  './data/Entertainment_Music/tune/formal.ref1',
                  './data/Entertainment_Music/tune/formal.ref2',
                  './data/Entertainment_Music/tune/formal.ref3']
formal_val_refs=['./data/Entertainment_Music/tune/informal.ref0',
                './data/Entertainment_Music/tune/informal.ref1',
                './data/Entertainment_Music/tune/informal.ref2',
                './data/Entertainment_Music/tune/informal.ref3']
fm2inf_bleu_inf_src='./data/Entertainment_Music/tune/informal.ref0'
fm2inf_bleu_fm_src='./data/Entertainment_Music/tune/formal'
add_inf_data='./new_exp_em/add_data/informal.add.bpe.bpe_len_filtered.score.filtered_by_lm'
add_fm_data='./new_exp_em/add_data/formal.add.bpe.bpe_len_filtered.score.filtered_by_lm'
inf_lm_score_file='./new_exp_em/add_data/informal.add.bpe.bpe_len_filtered.score'
fm_lm_score_file='./new_exp_em/add_data/formal.add.bpe.bpe_len_filtered.score'
#output:
suffix='_3'
train_pkl_path='./new_exp_em/informal2formal.nmt.train.pkl'+suffix
val_pkl_path='./new_exp_em/informal2formal.nmt.val.pkl'+suffix
test_pkl_path='./new_exp_em/informal2formal.nmt.test.pkl'+suffix
inf2fm_bleu_val_pkl_path='./new_exp_em/inf2fm_val.pkl'+suffix
fm2inf_bleu_val_pkl_path='./new_exp_em/fm2inf_val.pkl'+suffix
add_data_big_pkl_path='./new_exp_em/add_data/add_data_big.pkl'+suffix

class parameters():
    def __init__(self):
        # structure
        self.keep_prob = 0.55
        self.encoder_cell_type = 'gru'
        self.decoder_cell_type = 'gru'
        self.encoder_is_bidirectional = True
        self.layers_num=1
        self.learning_rate = 1e-3
        self.match_lr = 1e-3
        self.hidden_size = 300
        self.max_length = MAX_LENGTH
        self.embedding_size = 300
        self.lr_decay_rate = 0.95
        self.lr_decay_freq = 10
        self.optimizer = 'adam'
        self.denoising_swap_rate = 0.0
        self.max_swap_times = 0
        self.max_lm_masked_num = 3
        self.max_lm_masked_rate = 0.25
        self.share_embedding = True
        self.match_cls_lam = 1
        self.match_mse_lam = 10
        self.match_veclen_lam = 0.0
        self.init_stddev = 0.1
        # new_dropout detail
        self.embedding_keep_prob = self.keep_prob
        self.encoder_rnn_in_keep_prob = self.keep_prob
        self.encoder_rnn_out_keep_prob = self.keep_prob
        self.decoder_rnn_in_keep_prob = self.keep_prob
        self.decoder_rnn_out_keep_prob = self.keep_prob
        self.encoder_layer = 2
        self.decoder_layer = 1
        self.use_residual=False
        self.alpha = 1
        self.decode_alpha=0.6


class train_parameters(parameters):
    def __init__(self,arch_type='rnn'):
        super(train_parameters, self).__init__()
        self.n_epochs = 12
        self.batch_size = 256
        self.continue_train =False
        self.ckpt_for_infer = './new_exp_em/s2s_sls/infer'
        self.ckpt_for_train = './new_exp_em/s2s_sls/train'
        self.last_ckpt_for_infer = './new_exp_em/s2s_sls/infer'
        self.last_ckpt_for_train = './new_exp_em/s2s_sls/train'
        self.train_pkl_path = [train_pkl_path]#'./new_exp_fr/family_relationships.addori.pkl']
        # './new_exp_fr/add_data/p_inf_and_fm.add.smt.without_rule.origincase.pkl',
        # './new_exp_fr/add_data/inf_and_p_fm.add.smt.without_rule.origincase.pkl']
        self.val_pkl_path = val_pkl_path
        self.batches_per_evaluation = 100
        self.eval_step = ['match', 'fm2fm', 'inf2inf', 'inf2fm', 'fm2inf', 'fm_bt', 'inf_bt', 'inf2fm_bleu']
        self.train_step = [['match','inf2fm','fm2inf'],['match','inf2fm','fm2inf']]#,'fm2fm_swap','inf2inf_swap','masked_fm_lm','masked_inf_lm']]
        self.train_upweight = [1]
        self.apply_sample_weight = [True]
        self.compared_loss = ['inf2fm_bleu']
        self.inf2fm_bleu_val_path = inf2fm_bleu_val_pkl_path
        self.fm2inf_bleu_val_path = fm2inf_bleu_val_pkl_path
        self.max_to_keep = 3
        self.beam_size = 4
        self.max_step = 30000
        self.early_stop_num = 30000
        if arch_type=='gpt':
            self.batch_size = 16
            self.max_step = 15000
            self.batches_per_evaluation = 100
            self.early_stop_num = 12
            self.keep_prob = 1.00
            self.decode_alpha = 0.6
            self.beam_size = 4
            self.learning_rate = 1e-4
            self.match_lr = 1e-4
            self.match_cls_lam = 1
            self.match_mse_lam = 1
            self.match_veclen_lam = 0.0
        elif arch_type=='rnn_combined':
            self.continue_train = False
            self.batch_size = 128
            self.max_step = 5000
            self.keep_prob = 0.55
            self.decode_alpha = 0.8
            self.init_stddev = 0.1
            self.beam_size = 4
            self.train_pkl_path = [train_pkl_path, add_data_big_pkl_path]
            self.train_step = [['match', 'inf2fm', 'fm2inf'], ['match', 'inf2fm', 'fm2inf']]
            self.save_dir = './new_exp_em/model_domain_combined/'
            self.train_upweight = [1, 2]
            self.apply_sample_weight = [1, 1]
            self.dataset_lr_weight = [1, 1]
        else:
            self.continue_train = False
            self.batch_size = 128
            self.max_step = 5000
            self.keep_prob = 0.55
            self.decode_alpha = 0.8
            self.init_stddev = 0.1
            self.beam_size = 4
            self.train_pkl_path = [train_pkl_path]
            self.train_step = [['match', 'inf2inf', 'fm2fm', 'inf2fm', 'fm2inf']]
            self.save_dir = './new_exp_em/model_domain_combined/'
            self.train_upweight = [1]
            self.apply_sample_weight = [1]
            self.dataset_lr_weight = [1]
        self.check()

    def check(self):
        if 'inf2fm_bleu' in self.eval_step:
            if not os.path.exists(self.inf2fm_bleu_val_path):
                raise ValueError(self.inf2fm_bleu_val_path + 'not exist')
        if 'fm2inf_bleu' in self.eval_step:
            if not os.path.exists(self.fm2inf_bleu_val_path):
                raise ValueError(self.fm2inf_bleu_val_path + 'not exist')
        if (not 'inf2fm_bleu' in self.eval_step) and (not 'fm2inf_bleu' in self.eval_step):
            raise ValueError('inf2fm_bleu or fm2inf_bleu')

class generate_parameters(parameters):
    def __init__(self,arch_type='rnn'):
        super(generate_parameters, self).__init__()
        self.output_path = './new_exp_em/informal.semd.result'
        self.model_path = './new_exp_em/model_2019_lm0203_ori3loss_big3loss/best.4300.model'
        self.ckpt_for_infer = './new_exp_fr/s2s_sls/infer'
        self.ckpt_for_train = './new_exp_fr/s2s_sls/train'
        self.batch_size = 32
        self.src_is_inf = True
        self.input_path = test_pkl_path
        self.beam_size = 4
        if arch_type=='gpt':
            self.decode_alpha = 0.8
            self.beam_size = 8
            self.batch_size = 16
        elif arch_type=='rnn_combined':
            self.decode_alpha=0.8
            self.beam_size=30