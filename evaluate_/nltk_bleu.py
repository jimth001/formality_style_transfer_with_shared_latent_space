from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
import nltk
from utils import tools

def bleu(reference_files_src_list,gen_file_src,ngrams=4,ignore_case=False):
    all_reference=[]
    for src in reference_files_src_list:
        with open(src,'r',encoding='utf-8') as f:
            one_reference=[]
            for line in f:
                #one_reference.append(tools.tokenizer(line.strip(),only_split=False))
                if not ignore_case:
                    one_reference.append(nltk.word_tokenize(line.strip()))
                else:
                    one_reference.append(nltk.word_tokenize(line.strip().lower()))
            all_reference.append(one_reference)
    all_reference=[[all_reference[i][j] for i in range(0,len(all_reference))] for j in range(0,len(all_reference[0]))]
    gen=[]
    with open(gen_file_src,'r',encoding='utf-8') as f:
        for line in f:
            #gen.append(tools.tokenizer(line.strip(),only_split=False))
            if not ignore_case:
                gen.append(nltk.word_tokenize(line.strip()))
            else:
                gen.append(nltk.word_tokenize(line.strip().lower()))
    weight=[1.0/ngrams]*ngrams
    print(weight)
    #a=[]
    '''sf=SmoothingFunction()
    for sens,refs in zip(gen,all_reference):
        a.append(sentence_bleu(refs,sens, weights=weight,smoothing_function=sf.method1))
        #a.append(corpus_bleu([refs],[sens], weights=weight,smoothing_function=sf.method7))
    print(sum(a)/len(a))'''
    b=corpus_bleu(all_reference,gen,weights=weight)
    print(b)
    return b#sum(a)/len(a)#a

def get_ref_src_list(path_prefix,ref_num=4):
    src_list=[]
    for i in range(0,ref_num):
        src_list.append(path_prefix+str(i))
    return src_list
