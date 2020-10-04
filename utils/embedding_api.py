import nltk

def load_word_embedding(path,tool='fasttext'):
    if tool=='fasttext':
        vectors=[]
        vocab_hash={}
        with open(path,'r',encoding='utf-8') as f:
            first_line=True
            for line in f:
                if first_line:
                    first_line=False
                    continue
                strs=line.strip().split(' ')
                vocab_hash[strs[0]]=len(vectors)
                vectors.append([float(s) for s in strs[1:]])
        return vectors,vocab_hash
    else:
        return (None,None)

def seg_and_pos_tagging(sentence,debug=False):
    words=nltk.word_tokenize(sentence)
    word_tag=nltk.pos_tag(words)
    if debug:
        print(words)
        print(word_tag)
    return word_tag

def generate_corpus_for_pretrain_embedding(src_list,output_path,input_encoding='utf-8',output_encoding='utf-8'):
    corpus=[]
    for src in src_list:
        with open(src,'r',encoding=input_encoding) as f:
            for line in f:
                corpus.append(' '.join(nltk.word_tokenize(line.strip())))
    with open(output_path,'w',encoding=output_encoding) as fw:
        for s in corpus:
            fw.write(s+'\n')












