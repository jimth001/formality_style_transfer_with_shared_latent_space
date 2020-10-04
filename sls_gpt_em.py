from sls_gpt import train_onehotkey_with_multi_datasets,generate_copy
from sls_settings_v2_EM import train_parameters,generate_parameters,train_pkl_path
import os
import time
from evaluate_all import evaluate

def get_timestamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

def train():
    para = train_parameters(arch_type='gpt')
    para.ckpt_for_infer='./new_exp_em/infer/'
    para.ckpt_for_train='./new_exp_em/train/'
    if not os.path.exists(para.ckpt_for_infer):
        os.makedirs(para.ckpt_for_infer)
    if not os.path.exists(para.ckpt_for_train):
        os.makedirs(para.ckpt_for_train)
    para.train_pkl_path=[train_pkl_path]
    para.match_cls_lam = 1
    para.match_mse_lam = 1
    para.match_veclen_lam = 0.0
    para.train_step = [['match','fm2inf','inf2fm']]
    para.train_upweight=[1]
    para.apply_sample_weight=[0]
    para.dataset_lr_weight=[1]
    para.compared_loss=['inf2fm_bleu']
    best_step_num, best_val_bleu = train_onehotkey_with_multi_datasets(parameters=para)
    print(str(best_step_num) + ',' + str(best_val_bleu))

def test():
    gen_para = generate_parameters(arch_type='gpt')
    gen_para.ckpt_for_infer='./new_exp_em/infer/'
    gen_para.output_path=gen_para.ckpt_for_infer+'inf2fm_2.txt'
    generate_copy(para=gen_para)
    test_bleu = evaluate(gen_file_path=gen_para.output_path,refs_prefix='./data/Entertainment_Music/test/formal.ref')
    print(str(test_bleu))

if __name__=='__main__':
    train()
    test()



