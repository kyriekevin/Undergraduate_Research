# -*- coding: utf-8 -*-
# @Author: zyz
# @Time: 2021/8/4 15:51
# @File: model_baseline.py
# @Software: PyCharm


import torch
import torch.nn as nn
from transformers import BertModel

import pykp.utils.io as io
from inference.beam import Beam, GNMTGlobalScorer
from pykp.decoder.transformer import TransformerSeq2SeqDecoder
from pykp.encoder.transformer import TransformerSeq2SeqEncoder
from pykp.modules.position_embed import get_sinusoid_encoding_table

EPS = 1e-8


class Seq2SeqModel(nn.Module):
    """Container module with encoder, decoder, embeddings."""

    def __init__(self, opt):
        super(Seq2SeqModel, self).__init__()
        """Initialize model."""
        embed = nn.Embedding(opt.vocab_size, opt.word_vec_size, opt.vocab["word2idx"][io.PAD_WORD])
        self.init_emb(embed)
        self.cuda = opt.gpuid > -1
        self.global_scorer = GNMTGlobalScorer(0., 0., 'none', 'none')
        self.word2idx = opt.vocab["word2idx"]
        self.exclusion_tokens = set([self.word2idx[t] for t in ['<sep>']])
        self.copy_attn = opt.copy_attention

        pos_embed = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(3000, opt.word_vec_size, padding_idx=opt.vocab["word2idx"][io.PAD_WORD]),
            freeze=True)
        self.encoder = TransformerSeq2SeqEncoder.from_opt(opt, embed, pos_embed)
        self.decoder = TransformerSeq2SeqDecoder.from_opt(opt, embed, pos_embed)

    def init_emb(self, embed):
        """Initialize weights."""
        init_range = 0.1
        embed.weight.data.uniform_(-init_range, init_range)

    def forward(self, src, src_lens, src_oov, max_num_oov, src_mask):
        """
        :param src: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], with oov words replaced by unk idx
        :param src_lens: a list containing the length of src sequences for each batch, with len=batch, with oov words replaced by unk idx
        :param trg: a LongTensor containing the word indices of target sentences, [batch, trg_seq_len]
        :param src_oov: a LongTensor containing the word indices of source sentences, [batch, src_seq_len], contains the index of oov words (used by copy)
        :param max_num_oov: int, max number of oov for each batch
        :param src_mask: a FloatTensor, [batch, src_seq_len]
        :return:
        """
        with torch.no_grad():
            batch_size = src.size(0)
            beam_size = 5
            n_best = 1

            # Encoding
            memory_bank = self.encoder(src, src_lens, src_mask)

            # expand memory_bank, src_mask
            memory_bank = memory_bank.repeat(beam_size, 1, 1)  # [batch * beam_size, max_src_len, memory_bank_size]
            src_mask = src_mask.repeat(beam_size, 1)  # [batch * beam_size, src_seq_len]
            src_oov = src_oov.repeat(beam_size, 1)  # [batch * beam_size, src_seq_len]

            state = self.decoder.init_state(memory_bank, src_mask)

            beam_list = [Beam(beam_size, n_best=n_best, cuda=self.cuda, global_scorer=self.global_scorer,
                              pad=self.word2idx[io.PAD_WORD], eos=self.word2idx[io.EOS_WORD],
                              bos=self.word2idx[io.BOS_WORD], block_ngram_repeat=0,
                              exclusion_tokens=self.exclusion_tokens)
                         for _ in range(batch_size)]

            # Help functions for working with beams and batches
            def var(a):
                return torch.tensor(a, requires_grad=False)

            for t in range(1, len(src_lens) + 1):
                if all((b.done() for b in beam_list)):
                    break
                # Construct batch x beam_size nxt words.
                # Get all the pending current beam words and arrange for forward.
                # b.get_current_tokens(): [beam_size]
                # torch.stack([ [beam of batch 1], [beam of batch 2], ... ]) -> [batch, beam]
                # after transpose -> [beam, batch]
                # After flatten, it becomes
                # [batch_1_beam_1, batch_2_beam_1,..., batch_N_beam_1, batch_1_beam_2, ..., batch_N_beam_2, ...]
                # this match the dimension of hidden state
                decoder_input = var(torch.stack([b.get_current_tokens() for b in beam_list])
                                    .t().contiguous().view(-1, 1))
                # decoder_input: [batch_size * beam_size, 1]

                # Turn any copied words to UNKS
                if self.copy_attn:
                    decoder_input = decoder_input.masked_fill(
                        decoder_input.gt(self.decoder.vocab_size - 1), self.word2idx[io.UNK_WORD])

                # run one step of decoding
                if t > 1:
                    decoder_inputs = torch.cat([decoder_inputs, decoder_input], -1)
                else:
                    decoder_inputs = decoder_input

                decoder_dist, attn_dist = self.decoder(decoder_inputs, state,
                                                       src_oov, max_num_oov)
                decoder_dist = decoder_dist.squeeze(1)
                attn_dist = attn_dist.squeeze(1)

                log_decoder_dist = torch.log(decoder_dist + EPS)

                # Compute a vector of batch x beam word scores
                log_decoder_dist = log_decoder_dist.view(beam_size, batch_size,
                                                         -1)  # [beam_size, batch_size, vocab_size]
                attn_dist = attn_dist.view(beam_size, batch_size, -1)  # [beam_size, batch_size, src_seq_len]

                # Advance each beam
                for batch_idx, beam in enumerate(beam_list):
                    beam.advance(log_decoder_dist[:, batch_idx], attn_dist[:, batch_idx, :src_lens[batch_idx]])

            result_token = self._from_beam(beam_list)

            return result_token

    def _from_beam(self, beam_list):
        ret = []
        for b in beam_list:
            n_best = 1
            scores, ks = b.sort_finished(minimum=n_best)
            hyps, attn = [], []
            # Collect all the decoded sentences in to hyps (list of list of idx) and attn (list of tensor)
            for i, (times, k) in enumerate(ks[:n_best]):
                # Get the corresponding decoded sentence, and also the attn dist [seq_len, memory_bank_size].
                hyp, att = b.get_hyp(times, k)
                hyps.append(hyp)
                attn.append(att)
            ret.append(hyps)
        return ret


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased").to(opt.device)
        self.dropout = nn.Dropout(p=0.1)
        # self.linear = nn.Linear(self.bert.config.hidden_size, 12)

    def forward(self, src, src_mask, kg):
        # src
        # src_last_hidden_state [batch_size, src_seq_len, bert_dense]
        # src_pooled_output [batch_size, bert_dense]
        src_last_hidden_state, _ = self.bert(
            input_ids=src,
            attention_mask=src_mask,
            return_dict=False)
        src_mask_expanded = src_mask.unsqueeze(-1).expand(
            src_last_hidden_state.size()).float()  # [batch_size, src_seq_len, bert_dense]
        src_mask_embeddings = src_last_hidden_state * src_mask_expanded  # [batch_size, src_seq_len, bert_dense]
        src_sum_embeddings = torch.sum(src_mask_embeddings, 1)  # [batch_size, bert_dense]
        src_sum_mask = torch.clamp(src_mask_expanded.sum(1), min=1e-9)  # [batch_size, bert_dense]
        src_mean_pooling = src_sum_embeddings / src_sum_mask  # [batch_size, bert_dense]

        # kg
        kg = torch.Tensor(kg).to(src.device)
        kg = kg.reshape(kg.size(0), -1).long()
        kg_mask = torch.ones_like(kg).long()
        kg_last_hidden_state, _ = self.bert(
            input_ids=kg,
            attention_mask=kg_mask,
            return_dict=False)
        kg_mask_expanded = kg_mask.unsqueeze(-1).expand(
            kg_last_hidden_state.size()).float()  # [batch_size, kg_seq_len, bert_dense]
        kg_mask_embeddings = kg_last_hidden_state * kg_mask_expanded  # [batch_size, kg_seq_len, bert_dense]
        kg_sum_embeddings = torch.sum(kg_mask_embeddings, 1)  # [batch_size, bert_dense]
        kg_sum_mask = torch.clamp(kg_mask_expanded.sum(1), min=1e-9)  # [batch_size, bert_dense]
        kg_mean_pooling = kg_sum_embeddings / kg_sum_mask  # [batch_size, bert_dense]

        similarity = torch.mm(kg_mean_pooling, src_mean_pooling.T)
        reward = []
        for i in range(similarity.size(0)):
            reward.append(similarity[i, i] + similarity[i, i] / similarity.sum(axis=0))
        reward = torch.Tensor(reward)
        return reward, similarity
