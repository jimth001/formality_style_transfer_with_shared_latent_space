import re
import utils.tools as tool
import random
import nltk
import multiprocessing


dom_tag='<("[^"]*"|\'[^\']*\'|[^\'">])*>'
email_tag='[\w!#$%&\'*+/=?^_`{|}~-]+(?:\.[\w!#$%&\'*+/=?^_`{|}~-]+)*@(?:[\w](?:[\w-]*[\w])?\.)+[\w](?:[\w-]*[\w])?'
url_tag='[a-zA-z]+://[^\s]*'
com_tag='.*\.com.*'
re_dom=re.compile(dom_tag)
re_email=re.compile(email_tag)
re_url=re.compile(url_tag)
re_com=re.compile(com_tag)


def cut_sent(para):
    para = re.sub('([.!\?])([^”])',r"\1\n\2",para)
    para = re.sub('(\.{6})([^”])',r"\1\n\2",para)
    para = re.sub('(\…{2})([^”])',r"\1\n\2",para)
    para = re.sub('(”)','”\n',para)
    para = para.rstrip()
    return para.split("\n")

def cut_sentences(para_list):
    print('break sentences')
    print('old lines:',len(para_list))
    sentences=[]
    for p in para_list:
        for s in tool.break_sentence(p, punctuations = ['?', '？', '!', '.']):
            sentences.append(s)
    print('new lines:',len(sentences))
    return sentences

def single_process_word_tokenizer(sentence_list):
    new_sens=[' '.join(nltk.word_tokenize(s)) for s in sentence_list]
    return new_sens

def multi_process_word_tokenizer(sentence_list):
    num_thres = multiprocessing.cpu_count()-1



def filter_by_rule(ori_list):
    print('filter by rule')
    new_list=[x for x in ori_list if re_dom.match(x) is None and re_email.match(x) is None and re_url.match(x) is None
              and re_com.match(x) is None]
    print('old lines:',len(ori_list))
    print('new lines:',len(new_list))
    return new_list


def filter_by_length(ori_list,min_len,max_len):
    print('filter by length')
    new_list=[x for x in ori_list if len(x)>min_len and len(x)<max_len]
    print('old lines:', len(ori_list))
    print('new lines:', len(new_list))
    return new_list


def split_train_dev_test(ori_list,dev_num,test_num):
    random.shuffle(ori_list)
    dev=ori_list[:dev_num]
    test=ori_list[dev_num:dev_num+test_num]
    train=ori_list[dev_num+test_num:]
    return train,dev,test


def load_file_list(f_list):
    lines=[]
    for fp in f_list:
        with open(fp,'r',encoding='utf-8') as f:
            for line in f:
                lines.append(line.strip())
    return lines


def save_lines(data,path):
    with open(path,'w',encoding='utf-8') as fw:
        for d in data:
            fw.write(' '.join(nltk.word_tokenize(d))+'\n')


def Main():
    ori_path_list=['../data/family_relationships.data','../new_exp_fr/informal.train.rule','../new_exp_fr/formal.train.rule']
    data=load_file_list(ori_path_list)
    data = filter_by_rule(data)
    data=cut_sentences(data)
    data=filter_by_length(data,6,300)
    train,dev,test=split_train_dev_test(data,1000,1000)
    out_dir='../new_exp_fr/LM/'
    save_lines(train,out_dir+'train.txt')
    save_lines(dev, out_dir + 'dev.txt')
    save_lines(test, out_dir + 'test.txt')


if __name__=='__main__':
    Main()
    #print(cut_sentences(['Limewire.com You can dowload any thing their. why... e...']))
    #print(cut_sent('Limewire.com You can dowload any thing their. why... e...'))
    print('all work has finished')