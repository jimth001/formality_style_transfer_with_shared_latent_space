import math
import os
from nltk import word_tokenize
from nltk.util import ngrams
import sys
import string

def get_file_src_list(parent_path, file_type='.txt'):
    files = os.listdir(parent_path)
    src_list = []
    for file in files:
        absolute_path = os.path.join(parent_path, file)
        if os.path.isdir(absolute_path):
            src_list += get_file_src_list(absolute_path)
        elif file_type is None or file.endswith(file_type):
            src_list.append(absolute_path)
    return src_list

def create_dict_for_blue(parent_path,output_file_path,file_type=None):
    src_list=get_file_src_list(parent_path=parent_path,file_type=file_type)
    word_dict={}
    for src in src_list:
        with open(src,'r',encoding='utf-8') as f:
            for line in f:
                words=word_tokenize(line.strip())
                for w in words:
                    if w not in word_dict:
                        word_dict[w]=1
    with open(output_file_path,'w',encoding='utf-8') as fw:
        i=0
        for word in word_dict.keys():
            fw.write(word+'\t'+str(i)+'\n')
            i+=1

def load_dict(file_name):
    word_dict = {}
    f = open(file_name, 'r',encoding='utf-8')
    for line in f:
        lines = line.strip().split('\t')
        if (len(lines) == 2):
            word_dict[lines[0]] = int(lines[1])
    return word_dict


def sen_to_array(sen, word_dict, sen1):
    sens = word_tokenize(sen.strip())
    sen1s = word_tokenize(sen1.strip())
    words = []
    for i in sens:
        '''
        if(i not in sen1s):
            words.append(len(word_dict)+1)
            continue
        '''
        if (word_dict.get(i) != None):
            words.append(word_dict.get(i))
    return words


def compute(candidate, references, weights):
    # candidate = [c.lower() for c in candidate]
    # references = [[r.lower() for r in reference] for reference in references]
    p_ns = (modified_precision(candidate, references, i) for i, _ in enumerate(weights, start=1))
    s = math.fsum(w * math.log(p_n) for w, p_n in zip(weights, p_ns) if p_n)
    bp = brevity_penalty(candidate, references)
    return bp * math.exp(s)


def modified_precision(candidate, references, n):
    counts = counter_gram(candidate, n)
    # print counts
    if not counts:
        return 0
    max_counts = {}
    for reference in references:
        reference_counts = counter_gram(reference, n)
        for ngram in counts.keys():
            if (reference_counts.get(ngram) != None):
                max_counts[ngram] = max(max_counts.get(ngram, 0), reference_counts[ngram])
            else:
                max_counts[ngram] = 0.000000001
    clipped_counts = dict((ngram, min(counts[ngram], max_counts[ngram])) for ngram in counts.keys())
    # print counts
    # print clipped_counts
    result=sum(clipped_counts.values()) / sum(counts.values())
    return result


def counter_gram(word_array, n):
    ngram_words = {}
    for i in range(0, len(word_array) - n + 1):
        tmp_i = ''
        for j in range(0, n):
            tmp_i += str(word_array[i + j])
            tmp_i += ' '
        if (ngram_words.get(tmp_i) == None):
            ngram_words[tmp_i] = 1
        else:
            ngram_words[tmp_i] += 1
    return ngram_words


def brevity_penalty(candidate, references):
    c = len(candidate)
    r = min(abs(len(r) - c) for r in references)
    if c == 0:
        return 0
    if c > r:
        return 1
    else:
        return math.exp(1 - r / c)

def get_bleu(predict_file_path,ground_truth_file_path,word_dict_path,ngram_num=1,predict_encoding='utf-8'):
    word_dict = load_dict(word_dict_path)  # dict_file
    can = []
    query = ''
    answer = []
    weight=[]
    for i in range(ngram_num):
        weight.append(1.0/ngram_num)
    print(predict_file_path)
    f = open(predict_file_path, 'r',encoding=predict_encoding)  # generate_file
    for line in f:
        lines = line.strip().lower().split('\t')
        if (len(lines) == 3):
            can.append(sen_to_array(lines[1].strip(), word_dict, lines[0]))
            '''
            if(lines[1]=='1'):
                if(query!='' and answer!=[]):
                    can[query]=sen_to_array(answer[0].strip(),word_dict)
                query=lines[0]
                answer=[]
            else:
                answer.append(lines[0].replace('result: <END> ',''))
            '''
    f.close()
    print('total pre lines:',len(can))
    ref = []
    f = open(ground_truth_file_path, 'r',encoding='utf-8')  # orgin_file
    for line in f:
        lines = line.strip().lower().split('\t')
        if (len(lines) == 3):
            #lines[0] = lines[0] #a=a? why do this?
            tmp = []
            tmp.append(sen_to_array(lines[1].strip(), word_dict, lines[0]))
            ref.append(tmp)
    f.close()
    print('total ref lines:',len(ref))
    # print len(ref)
    # print len(can.keys())
    # print ref.keys()
    # print len(ref.keys())
    bleu_array = []
    bleu_total = 0
    for a,b in zip(can,ref):
        bleu_score = compute(a,b, weights=weight)
        bleu_total += bleu_score
        bleu_array.append(bleu_score)
    print(bleu_total / len(bleu_array))

def get_avg_bleu_for_refs(predict_file_path,ground_truth_file_path_list,word_dict_path,ngram_num=1,predict_encoding='gb18030'):
    bleu=0
    for src in ground_truth_file_path_list:
        bleu+=get_bleu(predict_file_path,src,word_dict_path,ngram_num=ngram_num,predict_encoding=predict_encoding)
    print(bleu/len(ground_truth_file_path_list))
