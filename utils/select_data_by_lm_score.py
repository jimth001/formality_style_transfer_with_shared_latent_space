def load_file_list(f_list):
    lines=[]
    for fp in f_list:
        with open(fp,'r',encoding='utf-8') as f:
            for line in f:
                strs=line.strip().split('\t')
                lines.append([float(strs[0]),len(lines),strs[1]])
    return lines


def build_id_dict(list):
    dic={}
    for l in list:
        dic[len(dic)]=l
    return dic


def select_data_by_lm_score(inf_path,fm_path,top_rate=0.1,bottom_rate=0.1,suffix='.filtered_by_lm'):
    inf_ori=load_file_list([inf_path])
    fm_ori=load_file_list([fm_path])
    inf_dic=build_id_dict(inf_ori)
    fm_dic=build_id_dict(fm_ori)
    inf_sorted=sorted(inf_ori,key=lambda x:x[0],reverse=True)
    fm_sorted = sorted(fm_ori, key=lambda x: x[0], reverse=True)
    data_num=len(inf_ori)
    start_id=int(top_rate*data_num)
    end_id=int((1-bottom_rate)*data_num)
    fm_top_score=fm_sorted[start_id][0]
    fm_bottom_score=fm_sorted[end_id-1][0]
    selected_inf=inf_sorted[start_id:end_id]
    fw_inf=open(inf_path+suffix,'w',encoding='utf-8')
    fw_fm=open(fm_path+suffix,'w',encoding='utf-8')
    for item in selected_inf:
        fm_item=fm_dic[item[1]]
        if fm_item[0]>=fm_bottom_score and fm_item[0]<=fm_top_score:
            fw_inf.write(item[2]+'\n')
            fw_fm.write(fm_item[2]+'\n')
    fw_inf.close()
    fw_fm.close()


if __name__=='__main__':
    select_data_by_lm_score('../new_exp_fr/add_data/informal.add.rule.bpe.bpe_len_filtered.score',
                            '../new_exp_fr/add_data/formal.add.rule.bpe.bpe_len_filtered.score')
    print('all work has finished')

