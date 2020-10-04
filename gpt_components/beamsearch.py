"""
Beam Search
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
#import tensorflow as tf
from collections import namedtuple
from tensorflow.python.util import nest
from utils.common import *

class BeamSearchState(namedtuple("BeamSearchState",
                                 ("inputs", "state", "finish"))):
    pass

def _get_inference_fn(model_fns, features):
    def inference_fn(inputs, state):
        local_features = {
            "source1": features["source1"],
            "source1_length": features["source1_length"],
            "source2": features["source2"],
            "source2_length": features["source2_length"],
            "target": tf.pad(inputs[:, 1:], [[0, 0], [0, 1]]),
            "target_length": tf.fill([tf.shape(inputs)[0]],
                                     tf.shape(inputs)[1])
        }
        outputs = []
        next_state = []
        for (model_fn, model_state) in zip(model_fns, state):
            if model_state:
                output, new_state = model_fn(local_features, model_state)
                outputs.append(output)
                next_state.append(new_state)
            else:
                output = model_fn(local_features)
                outputs.append(output)
                next_state.append({})
        # Ensemble
        log_prob = tf.add_n(outputs) / float(len(outputs))
        return log_prob, next_state
    return inference_fn

def ensemble_model_fn_wrapper_gpt(model_fn,input,states,hparams,scopes_for_ensemble):
    new_state_list=[]
    step_log_probs_list=[]
    for i in range(0, len(scopes_for_ensemble)):
        next_outputs = model_fn(hparams, input, past=states[i], scope=scopes_for_ensemble[i])
        next_state = tf.concat([states[i], next_outputs['presents']], axis=-2)
        new_state_list.append(next_state)
        step_log_probs = next_outputs['logits'][:, 0, :]
        step_log_probs_list.append(step_log_probs)
    step_log_probs_ensemble=tf.reduce_mean(tf.stack(step_log_probs_list,axis=1),axis=1)
    step_log_probs_ensemble = tf.log(tf.nn.softmax(step_log_probs_ensemble))
    return new_state_list, step_log_probs_ensemble


def _beam_search_step(time, func, state, batch_size, beam_size, alpha, eos_id, hparams, scopes_for_ensemble, ensemble=False, concat_state_dim=None):
    # Compute log probabilities
    seqs, log_probs = state.inputs[:2]
    flat_seqs = merge_first_two_dims(seqs)
    if ensemble:
        flat_state = nest.map_structure(lambda x: merge_first_two_dims(x),
                                        state.state)
        next_state,step_log_probs=ensemble_model_fn_wrapper_gpt(func,tf.expand_dims(flat_seqs[:, -1], axis=1),flat_state,hparams,scopes_for_ensemble=scopes_for_ensemble)
    else:
        flat_state = nest.map_structure(lambda x: merge_first_two_dims(x),
                                        state.state)
        next_outputs = func(hparams, tf.expand_dims(flat_seqs[:, -1], axis=1), flat_state)
        if concat_state_dim is not None:#none or -2
            next_state = nest.map_structure(lambda x:tf.concat([x,next_outputs['presents']],axis=concat_state_dim),flat_state)
        else:
            next_state = next_outputs['presents']
        step_log_probs = next_outputs['logits'][:, 0, :]
        step_log_probs = tf.log(tf.nn.softmax(step_log_probs))
    step_log_probs = split_first_two_dims(step_log_probs, batch_size,
                                          beam_size)
    next_state = nest.map_structure(
        lambda x: split_first_two_dims(x, batch_size, beam_size),
        next_state)
    curr_log_probs = tf.expand_dims(log_probs, 2) + step_log_probs
    # Apply length penalty
    length_penalty = tf.pow((5.0 + tf.to_float(time + 1)) / 6.0, alpha)
    curr_scores = curr_log_probs / length_penalty
    vocab_size = curr_scores.shape[-1].value or tf.shape(curr_scores)[-1]
    # Select top-k candidates
    # [batch_size, beam_size * vocab_size]
    curr_scores = tf.reshape(curr_scores, [-1, beam_size * vocab_size])
    # [batch_size, 2 * beam_size]
    top_scores, top_indices = tf.nn.top_k(curr_scores, k=2 * beam_size)
    # Shape: [batch_size, 2 * beam_size]
    beam_indices = top_indices // vocab_size
    symbol_indices = top_indices % vocab_size
    # Expand sequences
    # [batch_size, 2 * beam_size, time]
    candidate_seqs = gather_2d(seqs, beam_indices)
    candidate_seqs = tf.concat([candidate_seqs,
                                tf.expand_dims(symbol_indices, 2)], 2)
    # Expand sequences
    # Suppress finished sequences
    flags = tf.equal(symbol_indices, eos_id)
    # [batch, 2 * beam_size]
    alive_scores = top_scores + tf.to_float(flags) * tf.float32.min
    # [batch, beam_size]
    alive_scores, alive_indices = tf.nn.top_k(alive_scores, beam_size)
    alive_symbols = gather_2d(symbol_indices, alive_indices)
    alive_indices = gather_2d(beam_indices, alive_indices)
    alive_seqs = gather_2d(seqs, alive_indices)
    # [batch_size, beam_size, time + 1]
    alive_seqs = tf.concat([alive_seqs, tf.expand_dims(alive_symbols, 2)], 2)
    alive_state = nest.map_structure(
        lambda x: gather_2d(x, alive_indices),
        next_state)
    alive_log_probs = alive_scores * length_penalty
    # Select finished sequences
    prev_fin_flags, prev_fin_seqs, prev_fin_scores = state.finish
    # [batch, 2 * beam_size]
    step_fin_scores = top_scores + (1.0 - tf.to_float(flags)) * tf.float32.min
    # [batch, 3 * beam_size]
    fin_flags = tf.concat([prev_fin_flags, flags], axis=1)
    fin_scores = tf.concat([prev_fin_scores, step_fin_scores], axis=1)
    # [batch, beam_size]
    fin_scores, fin_indices = tf.nn.top_k(fin_scores, beam_size)
    fin_flags = gather_2d(fin_flags, fin_indices)
    pad_seqs = tf.fill([batch_size, beam_size, 1],
                       tf.constant(eos_id, tf.int32))
    prev_fin_seqs = tf.concat([prev_fin_seqs, pad_seqs], axis=2)
    fin_seqs = tf.concat([prev_fin_seqs, candidate_seqs], axis=1)
    fin_seqs = gather_2d(fin_seqs, fin_indices)
    new_state = BeamSearchState(
        inputs=(alive_seqs, alive_log_probs, alive_scores),
        state=alive_state,
        finish=(fin_flags, fin_seqs, fin_scores)
    )
    return time + 1, new_state


def beam_search(func, state, batch_size, beam_size, max_length, alpha,
                init_seqs, eos_id, hparams, scopes_for_ensemble, ensemble=False, concat_state_dim=None):
    init_log_probs = tf.constant([[0.] + [tf.float32.min] * (beam_size - 1)])
    init_log_probs = tf.tile(init_log_probs, [batch_size, 1])
    init_scores = tf.zeros_like(init_log_probs)
    fin_seqs = tf.zeros([batch_size, beam_size, 1], tf.int32)
    fin_scores = tf.fill([batch_size, beam_size], tf.float32.min)
    fin_flags = tf.zeros([batch_size, beam_size], tf.bool)
    state = BeamSearchState(
        inputs=(init_seqs, init_log_probs, init_scores),
        state=state,
        finish=(fin_flags, fin_seqs, fin_scores)
    )
    max_step = tf.reduce_max(max_length)

    def _is_finished(t, s):
        log_probs = s.inputs[1]
        finished_flags = s.finish[0]
        finished_scores = s.finish[2]
        max_lp = tf.pow(((5.0 + tf.to_float(max_step)) / 6.0), alpha)
        best_alive_score = log_probs[:, 0] / max_lp
        worst_finished_score = tf.reduce_min(
            finished_scores * tf.to_float(finished_flags), axis=1)
        add_mask = 1.0 - tf.to_float(tf.reduce_any(finished_flags, 1))
        worst_finished_score += tf.float32.min * add_mask
        bound_is_met = tf.reduce_all(tf.greater(worst_finished_score,
                                                best_alive_score))
        cond = tf.logical_and(tf.less(t, max_step),
                              tf.logical_not(bound_is_met))
        return cond

    def _loop_fn(t, s):
        outs = _beam_search_step(t, func, s, batch_size, beam_size, alpha, eos_id, hparams=hparams, scopes_for_ensemble=scopes_for_ensemble,
                                 ensemble=ensemble, concat_state_dim=concat_state_dim)
        return outs

    if type(state.state)==list:
        tmp=state.state
        state_shape_invariants=[]
        for item in tmp:
            state_shape_invariants.append(item.shape)
    else:
        state_shape_invariants=state.state.shape
    time = tf.constant(0, name="time")
    shape_invariants = BeamSearchState(
        inputs=(tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None])),
        state=state_shape_invariants,
        finish=(tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None]))
    )
    outputs = tf.while_loop(_is_finished, _loop_fn, [time, state],
                            shape_invariants=[tf.TensorShape([]),
                                              shape_invariants],
                            parallel_iterations=1,
                            back_prop=False)

    final_state = outputs[1]
    alive_seqs = final_state.inputs[0]
    alive_scores = final_state.inputs[2]
    final_flags = final_state.finish[0]
    final_seqs = final_state.finish[1]
    final_scores = final_state.finish[2]

    alive_seqs.set_shape([None, beam_size, None])
    final_seqs.set_shape([None, beam_size, None])

    final_seqs = tf.where(tf.reduce_any(final_flags, 1), final_seqs,
                          alive_seqs)
    final_scores = tf.where(tf.reduce_any(final_flags, 1), final_scores,
                            alive_scores)

    return final_seqs, final_scores


def create_inference_graph(init_seqs, state, step_fn, hparams, decode_length, batch_size, beam_size, decode_alpha, eos_id,
                           ensemble, concat_state_dim, scopes_for_ensemble=None):
    tiled_context_state = nest.map_structure(
        lambda x:tile_to_beam_size(x,beam_size),
        state
    )
    tiled_init_seq=nest.map_structure(
        lambda x:tile_to_beam_size(x,beam_size),
        init_seqs
    )
    seqs, scores = beam_search(step_fn, tiled_context_state, batch_size, beam_size,
                               decode_length, decode_alpha, tiled_init_seq, eos_id, hparams=hparams, scopes_for_ensemble=scopes_for_ensemble,
                               ensemble=ensemble, concat_state_dim=concat_state_dim)
    # return seqs[:, :top_beams, 1:], scores[:, :top_beams]
    return seqs, scores
