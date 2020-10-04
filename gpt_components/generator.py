import tensorflow as tf
import numpy as np
import os
import tensorflow.contrib.slim as slim

class beam_search_generator():
    def __init__(self,model_fn,beam_size,model_directory,max_dec_len=40,dec_alpha=0.6):
        self.model_fn=model_fn
        self.graph=tf.Graph()
        self.beam_size=beam_size
        self.max_decode_len=max_dec_len
        self.decode_alpha=dec_alpha
        self.model_path=model_directory

    def build_graph_and_restore(self,eos_id):
        with self.graph.as_default():
            #self.context = tf.placeholder(tf.int32, [1, None])
            self.seqs, _ =self.model_fn.build_beam_search_graph(self.beam_size,1,self.max_decode_len,decode_alpha=self.decode_alpha)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(self.model_path)
            sess = tf.Session(graph=self.graph, config=config)
            saver.restore(sess, ckpt)
            return sess

    def print_all_trainable_vars(self):
        # Print parameters
        with self.graph.as_default():
            all_weights = {v.name: v for v in tf.trainable_variables()}
            total_size = 0
            for v_name in sorted(list(all_weights)):
                v = all_weights[v_name]
                tf.logging.info("%s\tshape    %s", v.name[:-2].ljust(80),
                                str(v.shape).ljust(20))
                v_size = np.prod(np.array(v.shape.as_list())).tolist()
                total_size += v_size
            tf.logging.info("Total trainable variables size: %d", total_size)


    def generate(self,sess,raw_text,append_flag='\t',multi_pls=False):
        if multi_pls:
            strs=raw_text.split(append_flag)
            assert len(strs)==len(self.model_fn.inputs)
            feed_dict={}
            for i in range(0,len(strs)):
                tokens=self.model_fn.text_enc.encode(strs[i])+self.model_fn.text_enc.encode('\t')
                l=len(tokens)
                feed_dict[self.model_fn.inputs[i]]=[tokens]
                feed_dict[self.model_fn.input_lens[i]]=[l]
            seqs = sess.run(self.seqs, feed_dict=feed_dict)
            seqs = seqs[:, 0, :]
            text = self.model_fn.text_enc.decode(seqs[0])
        else:
            context_tokens = self.model_fn.text_enc.encode(raw_text.strip() + append_flag)
            seqs = sess.run(self.seqs, feed_dict={
                self.model_fn.inputs: [context_tokens]
            })
            seqs = seqs[:, 0, :]
            text = self.model_fn.text_enc.decode(seqs[0])
        return text