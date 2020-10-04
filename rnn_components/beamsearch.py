"""
Beam Search
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

from collections import namedtuple
from tensorflow.python.util import nest
from utils.common import *

class BeamSearchState(namedtuple("BeamSearchState",
                                 ("inputs", "state", "finish"))):
    pass


def _beam_search_step(time, func, state, batch_size, beam_size, alpha, eos_id, output_layer, embedding, padding_attn, memory, memory_len):
    # Compute log probabilities
    seqs, log_probs = state.inputs[:2]
    flat_seqs = merge_first_two_dims(seqs)
    flat_state = nest.map_structure(lambda x: merge_first_two_dims(x),
                                    state.state)
    step_input=tf.gather(embedding,flat_seqs[:,-1])
    step_out, next_state, attn = func(step_input, flat_state, memory=memory, memory_len=memory_len, return_attn=True)
    step_log_probs = output_layer(step_out)
    step_log_probs = tf.log(tf.nn.softmax(step_log_probs))
    step_log_probs = split_first_two_dims(step_log_probs, batch_size,
                                          beam_size)
    next_state = nest.map_structure(
        lambda x: split_first_two_dims(x, batch_size, beam_size),
        next_state)
    attn = split_first_two_dims(attn, batch_size, beam_size)
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
    candidate_attn = gather_2d(tf.concat([state.inputs[-1],attn],2),beam_indices)
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

    alive_attn = gather_2d(candidate_attn, alive_indices)

    # record history for alive:
    # tf.concat(state.inputs[4],alive_state

    # Select finished sequences
    prev_fin_flags, prev_fin_seqs, prev_fin_scores, prev_attn = state.finish
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

    prev_attn=tf.concat([prev_attn,padding_attn],axis=2)
    fin_attn=tf.concat([prev_attn,candidate_attn],axis=1)
    fin_attn=gather_2d(fin_attn,fin_indices)

    new_state = BeamSearchState(
        inputs=(alive_seqs, alive_log_probs, alive_scores, alive_attn),
        state=alive_state,
        finish=(fin_flags, fin_seqs, fin_scores, fin_attn),
    )

    return time + 1, new_state


def beam_search(func, state, memory, memory_len, batch_size, beam_size, max_length, alpha,
                bos_id, eos_id, embedding, output_layer, source_max_length):
    init_seqs = tf.fill([batch_size, beam_size, 1], bos_id)
    init_log_probs = tf.constant([[0.] + [tf.float32.min] * (beam_size - 1)])
    init_log_probs = tf.tile(init_log_probs, [batch_size, 1])
    init_scores = tf.zeros_like(init_log_probs)
    fin_seqs = tf.zeros([batch_size, beam_size, 1], tf.int32)
    fin_scores = tf.fill([batch_size, beam_size], tf.float32.min)
    fin_flags = tf.zeros([batch_size, beam_size], tf.bool)
    init_attn= tf.zeros([batch_size, beam_size, 0, source_max_length], tf.float32)
    padding_attn = tf.zeros([batch_size, beam_size, 1, source_max_length], tf.float32)
    state = BeamSearchState(
        inputs=(init_seqs, init_log_probs, init_scores, init_attn),
        state=state,
        finish=(fin_flags, fin_seqs, fin_scores, init_attn),
    )

    max_step = tf.reduce_max(max_length)

    def _is_finished(t, s):  # time,state ;the state is a BeamSearchState
        log_probs = s.inputs[1]  # init_log_probs
        finished_flags = s.finish[0]  # fin_flags
        finished_scores = s.finish[2]  # fin_scores
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
        outs = _beam_search_step(t, func, s, batch_size, beam_size, alpha, eos_id, output_layer, embedding, padding_attn, memory, memory_len)
        return outs

    time = tf.constant(0, name="time")
    shape_invariants = BeamSearchState(
        inputs=(tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None, None])),
        state=nest.map_structure(infer_shape_invariants, state.state),
        finish=(tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None, None]))
    )
    # shape_invariants：The shape invariants for the loop variables.
    outputs = tf.while_loop(_is_finished, _loop_fn, [time, state],
                            shape_invariants=[tf.TensorShape([]),
                                              shape_invariants],
                            parallel_iterations=1,
                            back_prop=False)

    final_state = outputs[1]
    alive_seqs = final_state.inputs[0]
    alive_scores = final_state.inputs[2]
    alive_attns = final_state.inputs[-1]
    final_flags = final_state.finish[0]
    final_seqs = final_state.finish[1]
    final_scores = final_state.finish[2]
    final_attns = final_state.finish[-1]

    alive_seqs.set_shape([None, beam_size, None])
    final_seqs.set_shape([None, beam_size, None])

    final_seqs = tf.where(tf.reduce_any(final_flags, 1), final_seqs,
                          alive_seqs)
    final_scores = tf.where(tf.reduce_any(final_flags, 1), final_scores,
                            alive_scores)
    final_attns = tf.where(tf.reduce_any(final_flags, 1), final_attns,
                            alive_attns)

    return final_seqs, final_scores, final_attns


def create_inference_graph(decoding_fn, states, memory, memory_len, decode_length, batch_size, beam_size, decode_alpha, bos_id, eos_id, embedding, output_layer, src_max_len):
    states = nest.map_structure(
        lambda x: tile_to_beam_size(x, beam_size),
        states)
    memory = nest.map_structure(
        lambda x: tile_to_beam_size(x, beam_size),
        memory)
    memory_len = nest.map_structure(
        lambda x: tile_to_beam_size(x, beam_size),
        memory_len)
    memory = nest.map_structure(lambda x: merge_first_two_dims(x),
                                    memory)
    memory_len = nest.map_structure(lambda x: merge_first_two_dims(x),
                                memory_len)
    seqs, scores, attns = beam_search(decoding_fn, states, memory, memory_len, batch_size, beam_size,
                               decode_length, decode_alpha, bos_id, eos_id, embedding, output_layer, src_max_len)
    # return seqs[:, :top_beams, 1:], scores[:, :top_beams]
    return seqs, scores, attns