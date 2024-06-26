import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq2seq import utils


class TransformerEncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.encoder_embed_dim
        self.self_attn = MultiHeadAttention(
            self.embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout

        self.fc1 = generate_linear(self.embed_dim, args.encoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.encoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, state, encoder_padding_mask):
        """Forward pass of a single Transformer Encoder Layer"""
        residual = state.clone()

        """
        ___QUESTION-6-DESCRIBE-D-START___
        1.  What is the purpose of encoder_padding_mask?
            Transformers work with fixed-size input, so we need to pad the input
            to the same length. encoder_padding_mask is used to disregard the
            padding when calculating the attention weights.
        """
        state, _ = self.self_attn(
            query=state, key=state, value=state, key_padding_mask=encoder_padding_mask
        )
        """
        ___QUESTION-6-DESCRIBE-D-END___
        """

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state


class TransformerDecoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout = args.attention_dropout
        self.activation_dropout = args.activation_dropout
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)

        self.self_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
        )

        self.encoder_attn = MultiHeadAttention(
            embed_dim=self.embed_dim,
            num_attn_heads=args.decoder_attention_heads,
            kdim=args.encoder_embed_dim,
            vdim=args.encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
        )

        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = generate_linear(self.embed_dim, args.decoder_ffn_embed_dim)
        self.fc2 = generate_linear(args.decoder_ffn_embed_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.need_attn = True

    def forward(
        self,
        state,
        encoder_out=None,
        encoder_padding_mask=None,
        incremental_state=None,
        prev_self_attn_state=None,
        self_attn_mask=None,
        self_attn_padding_mask=None,
        need_attn=False,
        need_head_weights=False,
    ):
        """Forward pass of a single Transformer Decoder Layer"""

        # need_attn must be True if need_head_weights
        need_attn = True if need_head_weights else need_attn

        residual = state.clone()
        state, _ = self.self_attn(
            query=state,
            key=state,
            value=state,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.self_attn_layer_norm(state)

        residual = state.clone()
        """
        ___QUESTION-6-DESCRIBE-E-START___
        1.  How does encoder attention differ from self attention?
            Encoder attention performs attention between the encoder's input and target sequence to create hidden
            representations. It can be used to 'align' the input and target sequences and has access to the entire
            encoder output.
            Self attention is used to create hidden representations of the input sequence which can be used to assign
            'importance' to different parts of the input. It aims to capture the dependencies between different words
            in the input and only has access up to the current step in the sequence.

        2.  What is the difference between key_padding_mask and attn_mask?
            attn_mask is used to prevent the model from seeing future tokens when training.
            key_padding_mask is used to stop the model from attending to padded tokens.

        3.  If you understand this difference, then why don't we need to give attn_mask here?
            attn_mask is not needed because there are no problems with seeing the entire encoder output. The encoder
            attention can have access to the entire encoder output while it's self attention that is only allowed
            to see up to the current step in the sequence.


        """
        state, attn = self.encoder_attn(
            query=state,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=encoder_padding_mask,
            need_weights=need_attn or (not self.training and self.need_attn),
        )
        """
        ___QUESTION-6-DESCRIBE-E-END___
        """

        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.encoder_attn_layer_norm(state)

        residual = state.clone()
        state = F.relu(self.fc1(state))
        state = F.dropout(state, p=self.activation_dropout, training=self.training)
        state = self.fc2(state)
        state = F.dropout(state, p=self.dropout, training=self.training)
        state += residual
        state = self.final_layer_norm(state)

        return state, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention"""

    def __init__(
        self,
        embed_dim,
        num_attn_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        self_attention=False,
        encoder_decoder_attention=False,
    ):
        """
        ___QUESTION-7-MULTIHEAD-ATTENTION-NOTE
        You shouldn't need to change the __init__ of this class for your attention implementation
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.k_embed_size = kdim if kdim else embed_dim
        self.v_embed_size = vdim if vdim else embed_dim

        self.num_heads = num_attn_heads
        self.attention_dropout = dropout
        self.head_embed_size = embed_dim // num_attn_heads  # this is d_k in the paper
        self.head_scaling = math.sqrt(self.head_embed_size)

        self.self_attention = self_attention
        self.enc_dec_attention = encoder_decoder_attention

        kv_same_dim = self.k_embed_size == embed_dim and self.v_embed_size == embed_dim
        assert (
            self.head_embed_size * self.num_heads == self.embed_dim
        ), "Embed dim must be divisible by num_heads!"
        assert (
            not self.self_attention or kv_same_dim
        ), "Self-attn requires query, key and value of equal size!"
        assert (
            self.enc_dec_attention ^ self.self_attention
        ), "One of self- or encoder- attention must be specified!"

        self.k_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.v_proj = nn.Linear(self.v_embed_size, embed_dim, bias=True)
        self.q_proj = nn.Linear(self.k_embed_size, embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        # Xavier initialisation
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        attn_mask=None,
        need_weights=True,
    ):

        # Get size features
        tgt_time_steps, batch_size, embed_dim = query.size()
        assert self.embed_dim == embed_dim
        """
        ___QUESTION-7-MULTIHEAD-ATTENTION-START
        Implement Multi-Head attention  according to Section 3.2.2 of https://arxiv.org/pdf/1706.03762.pdf.
        Note that you will have to handle edge cases for best model performance. Consider what behaviour should
        be expected if attn_mask or key_padding_mask are given?
        """
        # attn is the output of MultiHead(Q,K,V) in Vaswani et al. 2017
        # attn must be size [tgt_time_steps, batch_size, embed_dim]
        # attn_weights is the combined output of h parallel heads of Attention(Q,K,V) in Vaswani et al. 2017
        # attn_weights must be size [num_heads, batch_size, tgt_time_steps, key.size(0)]

        def __reshape_qkv_for_bmm(query, key, value):
            # Project the queries, keys and values to the multi-head space.
            q = self.q_proj(query)
            # q.shape = [tgt_time_steps, batch_size, embed_dim]
            k = self.k_proj(key)
            # k.shape = [key.size(0), batch_size, embed_dim]
            v = self.v_proj(value)
            # v.shape = [key.size(0), batch_size, embed_dim]

            # Reshape q so that it can be split into the different heads, one for each batch.
            q = q.view(
                tgt_time_steps, batch_size * self.num_heads, self.head_embed_size
            )
            # q.shape = [tgt_time_steps, batch_size * num_heads, head_embed_size]
            q = q.transpose(0, 1)  # for bmm
            # q.shape = [batch_size * num_heads, tgt_time_steps, head_embed_size]

            # Reshape k, same reasoning as for q.
            k = k.view(-1, batch_size * self.num_heads, self.head_embed_size)
            # k.shape = [key.size(0), batch_size * num_heads, head_embed_size]
            k = k.transpose(0, 1)
            # k.shape = [batch_size * num_heads, key.size(0), head_embed_size]
            k = k.transpose(1, 2)
            # k.shape = [batch_size * num_heads, head_embed_size, key.size(0)]

            # Reshape v, same reasoning.
            v = v.view(-1, batch_size * self.num_heads, self.head_embed_size)
            # v.shape = [key.size(0), batch_size * num_heads, head_embed_size]
            v = v.transpose(0, 1)
            # v.shape = [batch_size * num_heads, key.size(0), head_embed_size]
            return q, k, v

        def __calculate_scaled_attn_weights(q, k):
            # QK^T
            attn_weights_unscaled = torch.bmm(q, k)
            # attn_weights_unscaled.shape = [batch_size * num_heads, tgt_time_steps, key.size(0)]

            # QK^T / sqrt(d_k)
            attn_weights = attn_weights_unscaled / self.head_scaling
            # attn_weights.shape = [batch_size * num_heads, tgt_time_steps, key.size(0)]
            return attn_weights

        def __apply_key_padding_mask(key_padding_mask, attn_weights):
            if key_padding_mask is None:
                return attn_weights
                # attn_weights.shape = [batch_size * num_heads, tgt_time_steps, key.size(0)]

            # Reshape the key_padding_mask to match the attn_weights
            attn_weights = attn_weights.view(
                batch_size, self.num_heads, tgt_time_steps, key.size(0)
            )
            # attn_weights.shape = [batch_size, num_heads, tgt_time_steps, key.size(0)]

            # Expand the key_padding_mask to match the attn_weights
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            # key_padding_mask.shape = [batch_size, 1, 1, key.size(0)]
            # Apply the key_padding_mask to the attn_weights
            attn_weights = attn_weights.masked_fill(key_padding_mask, float("-inf"))
            # attn_weights.shape = [batch_size, num_heads, tgt_time_steps, key.size(0)]
            # Reshape the attn_weights back to the original shape
            attn_weights = attn_weights.view(
                batch_size * self.num_heads, tgt_time_steps, key.size(0)
            )
            # attn_weights.shape = [batch_size * num_heads, tgt_time_steps, key.size(0)]
            return attn_weights

        q, k, v = __reshape_qkv_for_bmm(query, key, value)
        # q.shape = [batch_size * num_heads, tgt_time_steps, head_embed_size]
        # k.shape = [batch_size * num_heads, head_embed_size, key.size(0)]
        # v.shape = [batch_size * num_heads, key.size(0), head_embed_size]

        attn_weights = __calculate_scaled_attn_weights(q, k)
        #attn_weights.shape = [batch_size * num_heads, tgt_time_steps, key.size(0)]

        attn_weights = __apply_key_padding_mask(key_padding_mask, attn_weights)
        # attn_weights.shape = [batch_size * num_heads, tgt_time_steps, key.size(0)]

        # Apply the attn_mask
        if attn_mask is not None:
            attn_weights += attn_mask

        # Apply the softmax function to the attn_weights
        attn_weights = F.softmax(attn_weights, dim=2)
        # attn_weights.shape = [batch_size * num_heads, tgt_time_steps, key.size(0)]

        # Calculate attn by performing bmm between the attn_weights and the values
        attn = torch.bmm(attn_weights, v)
        # attn.shape = [batch_size * num_heads, tgt_time_steps, head_embed_size]

        # Reshape attn to original shape
        attn = attn.transpose(0, 1)
        # attn.shape = [tgt_time_steps, batch_size * num_heads, head_embed_size]
        attn = attn.contiguous()
        attn = attn.view(tgt_time_steps, batch_size, embed_dim)
        # attn.shape = [tgt_time_steps, batch_size, embed_dim]

        # Reshape attn_weights acc. to num of heads
        attn_weights = attn_weights.view(
            batch_size, self.num_heads, tgt_time_steps, key.size(0)
        ) if need_weights else None
        # attn_weights.shape = [batch_size, num_heads, tgt_time_steps, key.size(0)]
        """
        ___QUESTION-7-MULTIHEAD-ATTENTION-END
        """
        return attn, attn_weights


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embed_dim = embed_dim
        self.padding_idx = padding_idx
        self.weights = PositionalEmbedding.get_embedding(
            init_size, embed_dim, padding_idx
        )
        self.register_buffer("_float_tensor", torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embed_dim, padding_idx=None):
        half_dim = embed_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embed_dim % 2 == 1:
            # Zero pad in specific mismatch case
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0.0
        return emb

    def forward(self, inputs, incremental_state=None, timestep=None):
        batch_size, seq_len = inputs.size()
        max_pos = self.padding_idx + 1 + seq_len

        if self.weights is None or max_pos > self.weights.size(0):
            # Expand embeddings if required
            self.weights = PositionalEmbedding.get_embedding(
                max_pos, self.embed_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        if incremental_state is not None:
            #   Positional embed is identical for all tokens during single step decoding
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return (
                self.weights.index_select(index=self.padding_idx + pos, dim=0)
                .unsqueeze(1)
                .repeat(batch_size, 1, 1)
            )

        # Replace non-padding symbols with position numbers from padding_idx+1 onwards.
        mask = inputs.ne(self.padding_idx).int()
        positions = (
            torch.cumsum(mask, dim=1).type_as(inputs) * mask
        ).long() + self.padding_idx

        # Lookup positional embeddings for each position and return in shape of input tensor w/o gradient
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(batch_size, seq_len, -1)
            .detach()
        )


def LayerNorm(normal_shape, eps=1e-5):
    return torch.nn.LayerNorm(
        normalized_shape=normal_shape, eps=eps, elementwise_affine=True
    )


def fill_with_neg_inf(t):
    return t.float().fill_(float("-inf")).type_as(t)


def generate_embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim**-0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def generate_linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
