#import os
from evaluate_.nltk_bleu import bleu
from evaluate_.evaluate import get_ref_src_list

def evaluate(gen_file_path=None,refs_prefix=None):
    if gen_file_path is None:
        gen_file_path = './new_exp_fr/informal.semd.result'
    if refs_prefix is None:
        refs_prefix='./data/Family_Relationships/test/formal.ref'
    # gen_file_path='./new_exp_fr/informal.test.rule.1'
    b=bleu(reference_files_src_list=get_ref_src_list(refs_prefix),
         gen_file_src=gen_file_path, ngrams=4, ignore_case=False)
    #bleu(reference_files_src_list=get_ref_src_list('./data/Family_Relationships/test/formal.ref'),
    #     gen_file_src='./data/Family_Relationships/model_outputs/formal.nmt_combined',
    #     ngrams=4, ignore_case=False)
    #formality=evaluate_one_formality(gen_file_path + '.bpe', is_inf=False)
    return b