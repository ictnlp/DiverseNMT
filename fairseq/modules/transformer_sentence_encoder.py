# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from fairseq import utils
from fairseq.modules import (
    MultiheadAttention, LearnedPositionalEmbedding, TransformerSentenceEncoderLayer, LayerNorm
)


def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
    if isinstance(module, MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)


def PositionalEmbedding(
    num_embeddings: int,
    embedding_dim: int,
    padding_idx: int,
)-> nn.Embedding:
    m = LearnedPositionalEmbedding(
        num_embeddings + padding_idx + 1, embedding_dim, padding_idx,
    )
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


class TransformerSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape B x T x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        encoder_normalize_before: bool = False,
        use_bert_layer_norm: bool = False,
        use_gelu: bool = True,
        apply_bert_init: bool = False,
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.ffn_embedding_dim = ffn_embedding_dim

        self.normalize_before = True

        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx
        )

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, self.padding_idx)
            if self.num_segments > 0
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                self.padding_idx,
            )
            if self.use_position_embeddings
            else None
        )

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    encoder_normalize_before=encoder_normalize_before,
                    use_bert_layer_norm=use_bert_layer_norm,
                    use_gelu=use_gelu,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        self.encoder_attn = MultiheadAttention(
                self.embedding_dim, num_attention_heads,
                dropout=attention_dropout,
                )

        self.encoder_attn_layer_norm = LayerNorm(self.embedding_dim)
        
        self.final_layer_norm = LayerNorm(self.embedding_dim)
        self.activation_dropout = 0.1
        self.fc1 = Linear(self.embedding_dim, self.ffn_embedding_dim)
        self.fc2 = Linear(self.ffn_embedding_dim, self.embedding_dim)

        self.linear1 = Linear(self.embedding_dim*2, self.embedding_dim)
        self.linear2 = Linear(self.embedding_dim*3, self.embedding_dim)

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)
        
    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor,
        encoder_out,
        decoder_out,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        attn_mask = self.buffered_future_mask_base(tokens)
        segment_labels = [[0] * tokens.size(1)] * tokens.size(0)
        x_forword, sen_rep, padding_mask = self.lm_features(tokens, attn_mask, segment_labels, None)
        x = x_forword[-1]#T*B*C
        
        segment_labels_rev = [[1] * tokens.size(1)] * tokens.size(0)
        x_rev, sen_rep_rev, _ = self.lm_features(tokens, attn_mask, segment_labels_rev, padding_mask, isreverse=True)
        x_rev = x_rev[-1][:-2]#T-2*B*C
        x_rev = torch.cat((reverse_3d(x_rev.transpose(0,1)), torch.zeros(x.size(1), 2, x_rev.size(2)).cuda()), 1)
        x_rev = x_rev.transpose(0,1)#T*B*C

        x_all2 = torch.cat((x, x_rev), -1)
        x = self.linear1(x_all2)

        residual = x
        x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
        x, attn = self.encoder_attn(
                query=x,
                key=encoder_out['encoder_out'],
                value=encoder_out['encoder_out'],
                key_padding_mask=encoder_out['encoder_padding_mask'],
                static_kv=True,
                need_weights=True,
                )
        x = F.dropout(x, p=self.dropout)
        x = residual + x 
        x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)

        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        x_all3 = torch.cat((x_all2, x), -1)
        x_all = self.linear2(x_all3).transpose(0,1)

        #Using decoder output for LM
        tokens_decoder = torch.argmax(decoder_out[0], dim=2).transpose(0,1)[:-1]
        tokens_decoder = torch.cat((2*torch.ones([1, tokens_decoder.size(1)]).cuda(), tokens_decoder.float()), 0).transpose(0,1).long()
        x_dec, sen_rep_dec, _ = self.lm_features(tokens_decoder, attn_mask, segment_labels, padding_mask, hadpadding=False)
        x_dec = x_dec[-1]
        x_mixed2 = torch.cat((x_dec, x_rev), -1)
        x = self.linear1(x_mixed2)

        residual = x
        x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, before=True)
        x, attn = self.encoder_attn(
                query=x,
                key=encoder_out['encoder_out'],
                value=encoder_out['encoder_out'],
                key_padding_mask=encoder_out['encoder_padding_mask'],
                static_kv=True,
                need_weights=True,
                )
        x = F.dropout(x, p=self.dropout)
        x = residual + x
        x = self.maybe_layer_norm(self.encoder_attn_layer_norm, x, after=True)
        residual = x
        x = self.maybe_layer_norm(self.final_layer_norm, x, before=True)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout)
        x = residual + x
        x = self.maybe_layer_norm(self.final_layer_norm, x, after=True)

        x_mixed3 = torch.cat((x_mixed2, x), -1)
        x_mixed = self.linear2(x_mixed3)

        return x_all, x_mixed


    def lm_features(self, tokens, attn_mask, segment_labels, padding_mask, isreverse=False, hadpadding=True):
        """
        Similar to *forword* but change the attn_mask for computing each word of a sentence

        Returns:
            tuple:
                - the encoder's features of shape `(batch, tgt_len, embed_dim)`
                - the sentence representation of shape `(batch, embed_dim)`
        """
        # compute padding mask. This is needed for multi-head attention
        if hadpadding:
            padding_mask = tokens.eq(self.padding_idx)
            if not padding_mask.any():
                padding_mask = None

        # embed positions
        positions = (
            self.embed_positions(tokens)
            if self.embed_positions is not None else None
        )
        #print(positions)

        # embed segments
        segments = (
            self.segment_embeddings(segment_labels)
            if self.segment_embeddings is not None
            else None
        )

        #print(segments)
        if isreverse:
            x = reverse_3d(self.embed_tokens(tokens))
        else:
            x = self.embed_tokens(tokens)
        
        if positions is not None:
            x += positions
        if segments is not None:
            x += segments
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        inner_states = [x]
        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_mask=attn_mask,
                self_attn_padding_mask=padding_mask,
            )
            inner_states.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        sentence_rep = x[:, 0, :]

        return inner_states, sentence_rep, padding_mask

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x


    def buffered_future_mask(self, tensor):
        #mask for 5-gram
        dim = tensor.size(0)
        #if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            #self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        self._future_mask = utils.fill_with_neg_inf(tensor.new(dim, dim))
        #self._future
        for i in range(dim):
            self._future_mask[i][i+1] = 0
            self._future_mask[i+1][i] = 0
            if (i > dim-3):
                break
            self._future_mask[i][i+2] = 0
            self._future_mask[i+2][i] = 0

        return self._future_mask[:dim, :dim]
    def buffered_future_mask_short(self, tensor, line):
        dim = tensor.size(1)
        self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        for i in range(line, dim):
            self._future_mask[i] = [float('-inf')] * dim
        return self._future_mask
    
    def buffered_future_mask_base(self, tensor):
        dim = tensor.size(1)
        self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1).float()
        return self._future_mask[:dim, :dim]

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias).cuda()
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m

def reverse_3d(inputs):
    assert len(inputs.size()) == 3
    res = reversed(inputs.contiguous().view(-1, inputs.size(-1))).view(inputs.size())
    res = reversed(res)
    #res = torch.randn(inputs.size())
    #for i in range(inputs.size(0)):
    #    res[i] = reversed(inputs[i])
    return res.cuda()
