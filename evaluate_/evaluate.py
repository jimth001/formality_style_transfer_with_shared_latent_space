from evaluate_.nltk_bleu import bleu

def transfer_origin_input_as_output(input_path,output_path,suffix='0'):
    f=open(input_path,'r',encoding='utf-8')
    fw=open(output_path,'w',encoding='utf-8')
    for line in f:
        w=line.strip('\r\n').strip('\n')
        fw.write(w+'\t'+w+'\t'+suffix+'\n')
    f.close()
    fw.close()


def transfer_baselines_as_output(origin_file,baseline_file,output_path,suffix='0'):
    f=open(origin_file,'r',encoding='utf-8')
    f2 = open(baseline_file, 'r', encoding='utf-8')
    fw=open(output_path,'w',encoding='utf-8')
    l1=[]
    for line in f:
        l1.append(line)
    l2=[]
    for line in f2:
        l2.append(line)
    for s1,s2 in zip(l1,l2):
        w1=s1.strip('\r\n').strip('\n')
        w2=s2.strip('\r\n').strip('\n')
        fw.write(w1+'\t'+w2+'\t'+suffix+'\n')
    f.close()
    f2.close()
    fw.close()


def change_rate(predict_file_path,predict_encoding='utf-8'):
    f=open(predict_file_path,'r',encoding=predict_encoding)
    change_num=0.0
    line_num=0.0
    print(predict_file_path)
    for line in f:
        strs=line.split('\t')
        if strs[0]!=strs[1]:
            change_num+=1
        line_num+=1
    print(change_num/line_num)


def get_ref_src_list(path_prefix,ref_num=4):
    src_list=[]
    for i in range(0,ref_num):
        src_list.append(path_prefix+str(i))
    return src_list
