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

