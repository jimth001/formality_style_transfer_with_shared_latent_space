# formality style transfer with shared latent space

## 1. model outputs

The outputs of our methods is under the "**model_outputs**" directory. The "**EM_Out**" means the result for "**Entertainment&Music**". The "**FR_Out**" means the result for "**Family&Relationships**".

"**formal.sls_rnn**" is the result of "**S2S-SLS(RNN)**"

"**formal.sls_rnn_cmb**" is the result of "**S2S-SLS(RNN)-Combined**"

"**formal.sls_gpt**" is the result of "**S2S-SLS(GPT)**"

"**formal.seq2seq_gpt**" is the result of "**GPT-Finetune**"

"**formal.seq2seq_rnn_cmb**" is the result of "**NMT-Combined\***".

"**formal.our_pbmt_cmb**" is the result of "**PBMT-Combined\***".

We also provide the outputs of ablation test.

## 2. model scripts

We implement the Transformer-based S2S-SLS model in **sls_gpt.py** and implement the RNN-based S2S-SLS model in **sls_rnn.py**.
We release four python files for running different neural architectures on different domains:

**sls_gpt_em.py** includes the APIs for training and testing the Transformer-based S2S-SLS model on Entertainment&Muisc.

**sls_gpt_fr.py** includes the APIs for training and testing the Transformer-based S2S-SLS model on Family&Relationship.

**sls_rnn_em.py** includes the APIs for training and testing the RNN-based S2S-SLS model on Entertainment&Muisc. 
It is for both data-limited and data-augmentation scenarios.

**sls_rnn_fr.py** includes the APIs for training and testing the RNN-based S2S-SLS model on Family&Relationship.
It is for both data-limited and data-augmentation scenarios.


## 3. training data<div id="contact"></div>

The training data includes original GYAFC dataset, the outputs of a simple rule based system, and the psesudo-parallel data(only for s2s-sls_rnn_combined). To obtain our training data, you should first get the access to [GYAFC dataset](https://github.com/raosudha89/GYAFC-corpus). Once you have gained the access to GYAFC dataset, please forward the acknowledgment to wangyunli@buaa.edu.cn or rmwangyl@qq.com, then we will provide access to our training data for reproducing our method. 

## 4. run

Our TensorFlow version is 1.12.0. We suggest to use Pycharm to run this project.

### 4.1 Applied to Other Datasets

Suppose a task for generating tgt from src. The train set consists of two files 'src_train.txt' and 'tgt_train.txt', the validation set (with two references) consists of 'src_val.txt' , 'tgt_val_ref1.txt' and 'tgt_val_ref2.txt', the test set (with two references) consists of 'src_test.txt' , 'tgt_test_ref1.txt' and 'tgt_test_ref2.txt'.

Each file consists of the texts arranged line by line. E.g. the texts with the same row number in 'src_train.txt' and 'tgt_train.txt' form a training sample.

There is an example for running the train and test stage of s2s_sls_gpt:

a. prepare data:

train_pkl=sls_gpt.prepare_data(src_train,tgt_train)

val_pkl=sls_gpt.prepare_data(src_val,tgt_val_ref1.txt,\[tgt_val_ref1.txt,tgt_val_ref2.txt\])

test_pkl=sls_gpt.prepare_data(src_test,tgt_test_ref1.txt,\[tgt_test_ref1.txt,tgt_test_ref2.txt\])

b. running train stage:

Put the [pre_trained gpt models](https://github.com/openai/gpt-2) and bpe files under 'sls_gpt.bpe_config_path'. If you want to train from scratch, you can only put the bpe files of GPT-2 under 'sls_gpt.bpe_config_path'. You can also train your bpe model by [learn_bpe.py](https://github.com/jimth001/my-tf-framework-for-nlp-tasks/blob/Tensorflow-1.x/TextPreprocessing/learn_bpe.py).

Modify the 'train_pkl_path' and 'val_pkl_path' in sls_settings_v2_FR.py to adapt to your project.

Running sls_gpt_fr.train(). 

Note that the 'tgt_val_ref1.txt' will be used to calculate the cross_entropy and the 'tgt_test_ref1.txt,tgt_test_ref2.txt' will be used to calculate the bleu score during training. 

c. running test stage:

Modify the paths of 'sls_settings_v2_FR.sls_settings_v2_FR' if you need, and run 'sls_gpt_fr.test()' to generate the results.



