from sls_gpt import train_onehotkey_with_multi_datasets,generate_copy
from sls_settings_v2_FR import train_parameters,generate_parameters,train_pkl_path
import os
import time
from evaluate_all import evaluate

def get_timestamp():
    return time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())

def train():
    para = train_parameters(arch_type="gpt")
    para.ckpt_for_infer='./new_exp_fr/infer/'
    para.ckpt_for_train='./new_exp_fr/train/'
    if not os.path.exists(para.ckpt_for_infer):
        os.makedirs(para.ckpt_for_infer)
    if not os.path.exists(para.ckpt_for_train):
        os.makedirs(para.ckpt_for_train)
    para.train_pkl_path=[train_pkl_path]
    para.train_step = [['match','fm2inf','inf2fm']]
    para.train_upweight=[1]
    para.apply_sample_weight=[0]
    para.dataset_lr_weight=[1]
    para.compared_loss=['inf2fm_bleu']
    best_step_num, best_val_bleu = train_onehotkey_with_multi_datasets(parameters=para)
    print(str(best_step_num) + ',' + str(best_val_bleu))

def test():
    gen_para = generate_parameters()
    gen_para.decode_alpha = 0.8
    gen_para.beam_size = 8
    gen_para.batch_size=16
    gen_para.ckpt_for_infer='./new_exp_fr/infer/'
    gen_para.output_path=gen_para.ckpt_for_infer+'inf2fm.txt'
    generate_copy(para=gen_para)
    test_bleu = evaluate(gen_file_path=gen_para.output_path)
    print(str(test_bleu))

if __name__=='__main__':
    train()
    test()





