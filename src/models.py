from typing import Any, Optional, Tuple
from numpy.core.numeric import zeros_like
import pdb
from pdb import Pdb, set_trace
from sklearn import model_selection
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch._C import device, dtype
from transformers import BertConfig, BertModel, BertPreTrainedModel
from transformers.modeling_utils import PreTrainedModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertOutput, BertLayer, BertOnlyMLMHead
from transformers.models.roberta.modeling_roberta import RobertaConfig, RobertaEmbeddings, RobertaPreTrainedModel, RobertaLMHead
from transformers.activations import ACT2FN


class EntityAwareBertConfig(BertConfig):
    def __init__(self, entity_queries_num = None, entity_emb_size = None, mask_ent2tok = True, mask_tok2ent = False, mask_ent2ent = False, mask_entself = False,
                 entity_aware_attention = False, entity_aware_intermediate = False, entity_aware_selfout = False, entity_aware_output = True,
                 use_entity_pos = True, use_entity_common_embedding = False, **kwargs):
        super(EntityAwareBertConfig, self).__init__( **kwargs)

        self.entity_queries_num = entity_queries_num
        self.mask_ent2tok = mask_ent2tok
        self.mask_tok2ent = mask_tok2ent
        self.mask_ent2ent = mask_ent2ent
        self.mask_entself = mask_entself
        self.entity_aware_attention = entity_aware_attention
        self.entity_aware_selfout = entity_aware_selfout
        self.entity_aware_intermediate = entity_aware_intermediate
        self.entity_aware_output = entity_aware_output
        self.type_vocab_size = 2

        self.use_entity_pos = use_entity_pos
        self.use_entity_common_embedding = use_entity_common_embedding

        if entity_emb_size is None:
            self.entity_emb_size = self.hidden_size
        else:
            self.entity_emb_size = entity_emb_size

class EntityEmbeddings(nn.Module):
    def __init__(self, config, is_pos_embedding = False):
        super().__init__()
        self.entity_embeddings = nn.Embedding(config.entity_queries_num, config.hidden_size)
        self.use_common_embedding = config.use_entity_common_embedding and not is_pos_embedding
        if self.use_common_embedding:
            self.entity_common_embedding = nn.Embedding(1, config.hidden_size)
            # self.entity_common_embedding = nn.Parameter(torch.rand(config.hidden_size))
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids):
        embeddings = self.entity_embeddings(input_ids)
        if self.use_common_embedding:
            embeddings = embeddings + self.entity_common_embedding(torch.tensor(0, device=embeddings.device))
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fusion_function = 'softmax'

    def forward(
            self,
            hidden_states,
            visual_hidden_state=None,
    ):
        fusion_scores = torch.matmul(hidden_states, visual_hidden_state.transpose(-1, -2))  # bsz, 128, 49
        if self.fusion_function == 'softmax':
            fusion_probs = nn.Softmax(dim=-1)(fusion_scores)
            fusion_output = torch.matmul(fusion_probs, visual_hidden_state)
        elif self.fusion_function == 'max':
            fusion_probs = fusion_scores.max(dim=-1)
        return fusion_output

class CLIPFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fusion_function = 'softmax'

    def forward(
            self,
            visual_hidden_state,
            text_hidden_states=None,
    ):
        fusion_scores = torch.matmul(visual_hidden_state, text_hidden_states.transpose(-1, -2))
        if self.fusion_function == 'softmax':
            fusion_probs = nn.Softmax(dim=-1)(fusion_scores)
            fusion_output = torch.matmul(fusion_probs, text_hidden_states)
        elif self.fusion_function == 'max':
            fusion_probs = fusion_scores.max(dim=-1)
        return fusion_output

class EntityAwareBertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.config = config


        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        if config.entity_aware_attention:
            # self.entity_w2e_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.entity_e2w_query = nn.Linear(config.hidden_size, self.all_head_size)
            self.entity_e2w_key = nn.Linear(config.hidden_size, self.all_head_size)
            self.entity_e2w_value = nn.Linear(config.hidden_size, self.all_head_size)

        # else:
        #     self.entity_e2w_query = self.query
        #     self.entity_e2w_key = self.key
        #     self.entity_e2w_value = self.value


        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.fusion = BertFusion(config)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, token_hidden_states, entity_hidden_states, attention_mask, query_pos = None, visual_hidden_state=None, output_qks=None,):
        context_size = token_hidden_states.size(1)

        pos_aware_entity_hidden_states = entity_hidden_states
        if query_pos is not None:
            pos_aware_entity_hidden_states = (entity_hidden_states + query_pos)/2
            # pos_aware_entity_hidden_states = entity_hidden_states + query_pos
        # query specific
        w2w_query_layer = self.transpose_for_scores(self.query(token_hidden_states))

        if self.config.entity_aware_attention:
            e2w_query_layer = self.transpose_for_scores(self.entity_e2w_query(pos_aware_entity_hidden_states))
        else:
            e2w_query_layer = self.transpose_for_scores(self.query(pos_aware_entity_hidden_states))


        # key unified transformered
        w2w_key_layer = self.transpose_for_scores(self.key(token_hidden_states))

        if self.config.entity_aware_attention:
            e2w_key_layer = self.transpose_for_scores(self.entity_e2w_key(pos_aware_entity_hidden_states))
        else:
            e2w_key_layer = self.transpose_for_scores(self.key(pos_aware_entity_hidden_states))


        w2w_value_layer = self.transpose_for_scores(self.value(token_hidden_states))

        if self.config.entity_aware_attention:
            e2w_value_layer = self.transpose_for_scores(self.entity_e2w_value(entity_hidden_states))
        else:
            e2w_value_layer = self.transpose_for_scores(self.value(entity_hidden_states))


        # w2w_key_layer = key_layer[:, :, :context_size, :]
        # e2w_key_layer = key_layer[:, :, :context_size, :]
        # w2e_key_layer = key_layer[:, :, context_size:, :]
        # e2e_key_layer = key_layer[:, :, context_size:, :]


        w2w_attention_scores = torch.matmul(w2w_query_layer, w2w_key_layer.transpose(-1, -2))
        w2e_attention_scores = torch.matmul(w2w_query_layer, e2w_key_layer.transpose(-1, -2))
        e2w_attention_scores = torch.matmul(e2w_query_layer, w2w_key_layer.transpose(-1, -2))
        # w2e_attention_scores = torch.zeros(e2w_attention_scores.size()).transpose(-1, -2).to(e2w_attention_scores.device) - 1e30
        e2e_attention_scores = torch.matmul(e2w_query_layer, e2w_key_layer.transpose(-1, -2))

        word_attention_scores = torch.cat([w2w_attention_scores, w2e_attention_scores], dim=3)
        entity_attention_scores = torch.cat([e2w_attention_scores, e2e_attention_scores], dim=3)
        attention_scores = torch.cat([word_attention_scores, entity_attention_scores], dim=2)

        attention_scores = attention_scores / (self.attention_head_size**0.5)
        attention_scores = attention_scores + attention_mask

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        # value unified transformered
        value_layer = torch.cat([w2w_value_layer, e2w_value_layer], dim = -2)
        context_layer = torch.matmul(attention_probs, value_layer)

        key_layer = torch.cat([w2w_key_layer, e2w_key_layer], dim = -2)
        qks = (w2w_key_layer, w2w_value_layer) if output_qks else None

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        fusion_output = self.fusion(context_layer[:, :context_size, :],
                                    visual_hidden_state) if visual_hidden_state is not None else None  # add

        return context_layer[:, :context_size, :], context_layer[:, context_size:, :], fusion_output, qks


class EntityAwareBertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.entity_aware_selfout:
            self.entity_dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.entity_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # else:
        #     self.entity_dense = self.dense
        #     self.entity_LayerNorm = self.LayerNorm
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_self_output, entity_self_output, token_hidden_states, entity_hidden_states):
        # why? solved, code above also works
        token_self_output = self.dense(token_self_output)
        token_self_output = self.dropout(token_self_output)
        token_self_output = self.LayerNorm(token_self_output + token_hidden_states)

        if self.config.entity_aware_selfout:
            entity_self_output = self.entity_dense(entity_self_output)
            entity_self_output = self.dropout(entity_self_output)
            entity_self_output = self.entity_LayerNorm(entity_self_output + entity_hidden_states)
        else:
            entity_self_output = self.dense(entity_self_output)
            entity_self_output = self.dropout(entity_self_output)
            entity_self_output = self.LayerNorm(entity_self_output + entity_hidden_states)
        hidden_states = torch.cat([token_self_output, entity_self_output], dim=1)
        return hidden_states


class EntityAwareBertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = EntityAwareBertSelfAttention(config)
        self.output = EntityAwareBertSelfOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask, query_pos = None,
                visual_hidden_state=None, output_qks=None):
        word_self_output, entity_self_output, fusion_output, qks = self.self(word_hidden_states, entity_hidden_states, attention_mask, query_pos = query_pos,
                                                         visual_hidden_state=visual_hidden_state, output_qks=output_qks)
        output = self.output(word_self_output, entity_self_output, word_hidden_states, entity_hidden_states)
        return output[:, : word_hidden_states.size(1), :], output[:, word_hidden_states.size(1) :, :], fusion_output, qks

class EntityAwareBertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fusion_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if config.entity_aware_intermediate:
            self.entity_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # else:
        #     self.entity_dense = self.dense
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, token_hidden_states, entity_hidden_states, fusion_output = None):
        context_size = token_hidden_states.size(1)
        token_hidden_states = self.dense(token_hidden_states)
        if self.config.entity_aware_intermediate:
            entity_hidden_states = self.entity_dense(entity_hidden_states)
        else:
            entity_hidden_states = self.dense(entity_hidden_states)
        if fusion_output is not None:
            fusion_states = self.fusion_dense(fusion_output)
            token_hidden_states = token_hidden_states + fusion_states
            # entity_hidden_states = entity_hidden_states + fusion_states[:, context_size:, :]
        token_hidden_states = self.intermediate_act_fn(token_hidden_states)
        entity_hidden_states = self.intermediate_act_fn(entity_hidden_states)

        return token_hidden_states, entity_hidden_states

class EntityAwareBertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if self.config.entity_aware_output:
            self.entity_dense = nn.Linear(config.intermediate_size, config.hidden_size)
            self.entity_LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # else:
        #     self.entity_dense = self.dense
        #     self.entity_LayerNorm = self.LayerNorm
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, token_intermediate_output, entity_intermediate_output, word_attention_output, entity_attention_output):
        token_intermediate_output = self.dense(token_intermediate_output)
        token_intermediate_output = self.dropout(token_intermediate_output)
        token_intermediate_output = self.LayerNorm(token_intermediate_output + word_attention_output)

        if self.config.entity_aware_output:
            entity_intermediate_output = self.entity_dense(entity_intermediate_output)
            entity_intermediate_output = self.dropout(entity_intermediate_output)
            entity_intermediate_output = self.entity_LayerNorm(entity_intermediate_output + entity_attention_output)
        else:
            entity_intermediate_output = self.dense(entity_intermediate_output)
            entity_intermediate_output = self.dropout(entity_intermediate_output)
            entity_intermediate_output = self.LayerNorm(entity_intermediate_output + entity_attention_output)

        return token_intermediate_output, entity_intermediate_output


class EntityAwareBertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = EntityAwareBertAttention(config)
        self.intermediate = EntityAwareBertIntermediate(config)
        self.output = EntityAwareBertOutput(config)

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask, query_pos = None, visual_hidden_state=None, output_qks=None):
        word_attention_output, entity_attention_output, fusion_output, qks = self.attention(
            word_hidden_states, entity_hidden_states, attention_mask, query_pos = query_pos,
            visual_hidden_state=visual_hidden_state, output_qks=output_qks
        )
        token_intermediate_output, entity_intermediate_output = self.intermediate(word_attention_output, entity_attention_output, fusion_output)
        token_output, entity_output = self.output(token_intermediate_output, entity_intermediate_output, word_attention_output, entity_attention_output)
        if output_qks:
            return token_output, entity_output, qks
        return token_output, entity_output, None

class CLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.fusion_dense_layer = nn.Linear(config.hidden_size, config.intermediate_size)


    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: bool = False,
            past_key_values: torch.Tensor = None,
            text_hidden_states = None
    ):
        """
        Args:
            hidden_states (:obj:`torch.FloatTensor`): input to the layer of shape :obj:`(seq_len, batch, embed_dim)`
            attention_mask (:obj:`torch.FloatTensor`): attention mask of size
                :obj:`(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (:obj:`torch.FloatTensor`): mask for attention heads in a given layer of size
                :obj:`(config.encoder_attention_heads,)`.
            output_attentions (:obj:`bool`, `optional`):
                Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights, fusion_output = self.self_attn(
            hidden_states=hidden_states,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            text_hidden_states=text_hidden_states
        )
        hidden_states = residual + hidden_states

        if fusion_output is not None:
            # fusion_states = self.fusion_dense_layer(fusion_output)
            hidden_states = hidden_states + fusion_output

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class CLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        assert (
                self.head_dim * self.num_heads == self.embed_dim
        ), f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`: {self.num_heads})."
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.fusion = CLIPFusion(config)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            output_attentions: bool = False,
            past_key_values: torch.Tensor = None,
            text_hidden_states = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        if past_key_values is not None:
            key_states = torch.cat([past_key_values[0], key_states], dim=2)
            value_states = torch.cat([past_key_values[1], value_states], dim=2)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz)

        query_states = query_states.view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is {attn_weights.size()}"
            )
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, embed_dim)

        attn_output = self.out_proj(attn_output)

        fusion_output = self.fusion(attn_output,
                                    text_hidden_states) if text_hidden_states is not None else None

        return attn_output, attn_weights_reshaped, fusion_output

class CLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states):
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states

class EntityAwareBertEncoder(nn.Module):
    def __init__(self, text_config, vision_config):
        super().__init__()
        self.vision_config = vision_config
        self.text_config = text_config

        self.vision_layers = nn.ModuleList([CLIPEncoderLayer(vision_config) for _ in range(vision_config.num_hidden_layers)])
        self.layer = nn.ModuleList([EntityAwareBertLayer(text_config) for _ in range(text_config.num_hidden_layers)])

    def forward(self, word_hidden_states, entity_hidden_states, attention_mask, vision_embeds=None, query_pos = None,
                head_mask=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):

        assert self.vision_config.num_hidden_layers == self.text_config.num_hidden_layers

        all_vision_hidden_states = () if output_hidden_states else None
        all_text_hidden_states = () if output_hidden_states else None
        all_vision_attentions = () if output_attentions else None
        all_text_attentions = () if output_attentions else None

        vision_hidden_states = vision_embeds

        intermediate = [{"h_token": word_hidden_states, "h_entity": entity_hidden_states, "h_region": vision_hidden_states}]
        # ori_entity_hidden_states = entity_hidden_states
        # for layer_module in self.layer:
        #     word_hidden_states, entity_hidden_states = layer_module(
        #         word_hidden_states, entity_hidden_states, attention_mask, query_pos
        #     )
        #     intermediate.append({"h_token": word_hidden_states, "h_entity": entity_hidden_states})
        # TODO  query-guided fusion
        for idx in range(self.vision_config.num_hidden_layers):
            # vision
            past_key_values = qks if idx >= 8 else None
            past_text_hidden_states = word_hidden_states if idx >= 8 else None
            vision_layer_module = self.vision_layers[idx]
            vision_layer_output = vision_layer_module(
                vision_hidden_states,
                output_attentions=output_attentions,
                past_key_values=past_key_values,
                text_hidden_states=None
            )
            vision_hidden_states = vision_layer_output[0]

            # text
            last_hidden_state = vision_hidden_states if idx >= 8 else None
            output_qks = True if idx >= 7 else None
            layer_head_mask = head_mask[idx] if head_mask is not None else None
            text_layer_module = self.layer[idx]
            word_hidden_states, entity_hidden_states, qks = text_layer_module(
                word_hidden_states, entity_hidden_states, attention_mask, query_pos,
                visual_hidden_state=last_hidden_state,
                output_qks=output_qks,
            )
            intermediate.append({"h_token": word_hidden_states, "h_entity": entity_hidden_states, "h_region": vision_hidden_states})
            # text_hidden_states = text_layer_output[0]

        return word_hidden_states, entity_hidden_states, intermediate, vision_hidden_states

class EntitySelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.entity_attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, dropout=config.hidden_dropout_prob)
        self.entity_dropout = nn.Dropout(config.hidden_dropout_prob)
        self.entity_norm = nn.LayerNorm(config.hidden_size)

    def forward(self, h_entity, query_pos):
        q = k = v = h_entity
        if query_pos is not None:
            q = k = h_entity + query_pos
        tgt = self.entity_attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[0].transpose(0, 1)
        tgt = h_entity + self.entity_dropout(tgt)
        h_entity = self.entity_norm(tgt)
        return h_entity

class SelfCrossAttention(nn.Module):
    def __init__(self, text_config, vision_config, use_token_level_encoder, use_entity_attention, num_selfcrosslayer):
        super().__init__()
        self.use_token_level_encoder = use_token_level_encoder
        self.use_entity_attention = use_entity_attention
        self.num_selfcrosslayer = num_selfcrosslayer

        self.selflaters = None
        if self.use_entity_attention:
            # self.selflaters = EntitySelfAttention(config)
            self.selflaters = nn.ModuleList([EntitySelfAttention(text_config) for _ in range(num_selfcrosslayer)])

        self.crosslayers = None
        if self.use_token_level_encoder:
            self.crosslayers = nn.ModuleList([EntityAwareBertLayer(text_config) for _ in range(num_selfcrosslayer)])
            self.vision_layers = nn.ModuleList([CLIPEncoderLayer(vision_config) for _ in range(num_selfcrosslayer)])

    def forward(self, h_token, h_entity, token_entity_attention_mask, query_pos = None , visual_hidden_state=None, output_qks=None):
        # intermediate = [{"h_token":h_token, "h_entity":h_entity}]
        intermediate = []
        for i in range(self.num_selfcrosslayer):

            if self.use_token_level_encoder:
                h_token, h_entity, qks = self.crosslayers[i](h_token, h_entity, token_entity_attention_mask, query_pos = query_pos,
                                                           visual_hidden_state=None, output_qks=output_qks)
                if visual_hidden_state is not None:
                    vision_layer_module = self.vision_layers[i]
                    vision_layer_output = vision_layer_module(
                        visual_hidden_state,
                        output_attentions=False,
                        past_key_values=qks,
                    )
                    visual_hidden_state = vision_layer_output[0]

            if self.use_entity_attention:
                h_entity = self.selflaters[i](h_entity, query_pos)

            intermediate.append({"h_token":h_token, "h_entity":h_entity, "h_region": visual_hidden_state})

        return h_token, h_entity, visual_hidden_state, intermediate


class EntityBoundaryPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.token_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.entity_embedding_linear = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size)
        )
        self.boundary_predictor = nn.Linear(self.hidden_size, 1)

    def forward(self, token_embedding, entity_embedding, token_mask):
        # B x #ent x #token x hidden_size
        entity_token_matrix = self.token_embedding_linear(token_embedding).unsqueeze(1) + self.entity_embedding_linear(entity_embedding).unsqueeze(2)
        entity_token_cls = self.boundary_predictor(torch.relu(entity_token_matrix)).squeeze(-1)
        token_mask = token_mask.unsqueeze(1).expand(-1, entity_token_cls.size(1), -1)
        entity_token_cls[~token_mask] = -10000 #只预测有token的boundary
        # entity_token_p = entity_token_cls.softmax(dim=-1)
        entity_token_p = F.sigmoid(entity_token_cls)

        return entity_token_p

class RegionBoundaryPredictor(nn.Module):
    def __init__(self, vision_config, text_config):
        super().__init__()
        self.vision_hidden_size = vision_config.hidden_size
        self.text_hidden_size = text_config.hidden_size
        self.token_embedding_linear = nn.Sequential(
            # nn.Dropout(p=0.5),
            # nn.ReLU(),
            nn.Linear(self.vision_hidden_size * 16, self.text_hidden_size),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )
        self.token_embedding_linear_boxfea = nn.Sequential(
            # nn.Dropout(p=0.5),
            # nn.ReLU(),
            nn.Linear(self.vision_hidden_size, self.text_hidden_size)
        )
        self.entity_embedding_linear = nn.Sequential(
            # nn.Linear(cls_size, config.hidden_size),
            # nn.Dropout(p=0.5),
            # nn.ReLU(),
            nn.Linear(self.text_hidden_size, self.text_hidden_size),
        )
        self.boundary_predictor = nn.Linear(self.text_hidden_size, 1)

    def forward(self, region_embedding, entity_embedding, token_mask):
        batch_size = entity_embedding.size(0)
        path_num = token_mask.size(1)
        region_embedding = region_embedding[:, 1:, :]
        region_embedding = region_embedding.view(batch_size, path_num, -1)
        # B x #ent x #token x hidden_size
        entity_token_matrix = self.token_embedding_linear(region_embedding).unsqueeze(1) + self.entity_embedding_linear(entity_embedding).unsqueeze(2)
        entity_token_cls = self.boundary_predictor(torch.relu(entity_token_matrix)).squeeze(-1)
        token_mask = token_mask.unsqueeze(1).expand(-1, entity_token_cls.size(1), -1)
        entity_token_cls[~token_mask] = -10000 #只预测有token的boundary
        # entity_token_p = entity_token_cls.softmax(dim=-1)
        entity_token_p = F.sigmoid(entity_token_cls)

        return entity_token_p

class EntityTypePredictor(nn.Module):
    def __init__(self, config, cls_size, entity_type_count):
        super().__init__()


        self.linnear = nn.Linear(cls_size, config.hidden_size)

        self.multihead_attn = nn.MultiheadAttention(config.hidden_size, dropout=config.hidden_dropout_prob, num_heads= config.num_attention_heads)
        # self.multihead_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, dropout=config.hidden_dropout_prob)

        self.classifier = nn.Sequential(
            # nn.Linear(cls_size, config.hidden_size),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(config.hidden_size * 3, entity_type_count)
        )

    def forward(self, h_entity, h_token, p_left, p_right, token_mask):
        h_entity = self.linnear(torch.relu(h_entity))
        # p_left = p_left/token_mask.sum(dim = -1).unsqueeze(-1).unsqueeze(-1)
        # p_right = p_right/token_mask.sum(dim = -1).unsqueeze(-1).unsqueeze(-1)

        attn_output, _ = self.multihead_attn(h_entity.transpose(0, 1).clone(), h_token.transpose(0, 1), h_token.transpose(0, 1), key_padding_mask=~token_mask)
        attn_output = attn_output.transpose(0, 1)
        h_entity += attn_output

        # # B N T      B T H   ->   B N H 。
        # p_left = p_left.detach()
        # p_right = p_right.detach()

        left_token = torch.matmul(p_left, h_token)
        right_token = torch.matmul(p_right, h_token)

        h_entity = torch.cat([h_entity,left_token,right_token], dim =-1)


        entity_logits = self.classifier(h_entity)

        return entity_logits, h_entity

class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        self.patch_embedding = nn.Conv2d(
            in_channels=3, out_channels=self.embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )

        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)))

        self.aux_position_embedding = nn.Embedding(160, self.embed_dim)
        self.register_buffer("aux_position_ids", torch.arange(160).expand((1, -1)))

        self.rcnn_position_embedding = nn.Embedding(12, self.embed_dim)
        self.register_buffer("rcnn_position_ids", torch.arange(12).expand((1, -1)))

        # for box features
        self.img_proj = nn.Sequential(
            nn.Linear(2048, self.embed_dim),
            nn.Dropout(p=0.5)
        )
        self.aux_position_embedding_boxfea = nn.Embedding(19, self.embed_dim)
        self.register_buffer("aux_position_ids_boxfea", torch.arange(19).expand((1, -1)))

    def forward(self, pixel_values, aux_embeddings=None, rcnn_embeddings=None):
        if pixel_values is None:
            batch_size = aux_embeddings.shape[0]

            class_embeds = self.class_embedding.expand(batch_size, 1, -1)
            embeddings = class_embeds
            aux_embeddings = self.img_proj(aux_embeddings)
            aux_embeds = aux_embeddings + self.aux_position_embedding_boxfea(self.aux_position_ids_boxfea)
            embeddings = torch.cat((embeddings, aux_embeds), dim=1)
        else:
            batch_size = pixel_values.shape[0]

            class_embeds = self.class_embedding.expand(batch_size, 1, -1)
            embeddings = class_embeds

            if aux_embeddings is not None:
                aux_embeds = []
                for aux_embedding in aux_embeddings:
                    # 10,768,4,4
                    aux_embed = self.patch_embedding(aux_embedding)
                    # 160,768
                    aux_embed = aux_embed.flatten(2).transpose(1, 2).flatten(0, 1)  # 10*16, 768
                    aux_embeds.append(aux_embed)
                aux_embeds = torch.stack(aux_embeds)  # bsz, 160, 768
                aux_embeds = aux_embeds + self.aux_position_embedding(self.aux_position_ids)
                embeddings = torch.cat((embeddings, aux_embeds), dim=1)

            if rcnn_embeddings is not None:
                rcnn_embeds = []
                for rcnn_embedding in rcnn_embeddings:
                    rcnn_embed = self.patch_embedding(rcnn_embedding)
                    rcnn_embed = rcnn_embed.flatten(2).transpose(1, 2).flatten(0, 1)
                    rcnn_embeds.append(rcnn_embed)
                rcnn_embeds = torch.stack(rcnn_embeds)  # bsz, 12, 768
                rcnn_embeds = rcnn_embeds + self.rcnn_position_embedding(self.rcnn_position_ids)
                embeddings = torch.cat((embeddings, rcnn_embeds), dim=1)
        return embeddings
class EntityAwareBertModel(nn.Module):

    config_class = EntityAwareBertConfig
    def __init__(self, text_config, vision_config):
        super().__init__()
        # text
        self.config = text_config
        self.embeddings = BertEmbeddings(text_config)
        # vision
        self.vision_config = vision_config
        self.vision_embeddings = CLIPVisionEmbeddings(vision_config)
        self.vision_pre_layrnorm = nn.LayerNorm(vision_config.hidden_size)
        self.vision_post_layernorm = nn.LayerNorm(vision_config.hidden_size)

        self.encoder = EntityAwareBertEncoder(text_config, vision_config)

        self.entity_embeddings = EntityEmbeddings(text_config)

        if text_config.use_entity_pos:
            self.pos_entity_embeddings = EntityEmbeddings(text_config, is_pos_embedding = True)
            # self.pos_entity_embeddings.entity_embeddings = self.entity_embeddings.entity_embeddings

        # self.register_buffer("entity_ids", torch.arange(config.entity_queries_num).expand((1, -1)))

        # self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _compute_extended_attention_mask(self, word_attention_mask: torch.LongTensor, entity_attention_mask: torch.LongTensor, mask_ent2tok = None, mask_tok2ent = None, mask_ent2ent = None, mask_entself = None, seg_mask = None):
        # pdb.set_trace()
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        word_num = word_attention_mask.size(1)
        entity_num = entity_attention_mask.size(1)
        # #batch x #head x seq_len x seq_len
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, word_num+entity_num, -1).clone()

        # mask entity2token attention
        if (mask_ent2tok == None and self.config.mask_ent2tok) or mask_ent2tok == True:
            extended_attention_mask[:, :, :word_num, word_num:] = 0

        if  (mask_tok2ent == None and self.config.mask_tok2ent) or mask_tok2ent == True:
            extended_attention_mask[:, :, word_num:, :word_num] = 0

        if seg_mask != None:
            tok2ent_mask = extended_attention_mask[:, :, word_num:, :word_num]
            seg_mask = seg_mask.bool().unsqueeze(1).unsqueeze(2).expand_as(tok2ent_mask)
            extended_attention_mask[:, :, word_num:, :word_num] = seg_mask

        if  (mask_ent2ent == None and self.config.mask_ent2ent) or mask_ent2ent == True:
            entity_attention = extended_attention_mask[:, :, word_num:, word_num:]
            mask = torch.eye(entity_num, entity_num, dtype = torch.bool).expand_as(entity_attention)
            entity_attention[~mask] = 0

        if (mask_entself == None and self.config.mask_entself) or mask_entself == True:
            entity_attention = extended_attention_mask[:, :, word_num:, word_num:]
            mask = torch.eye(entity_num, entity_num, dtype = torch.bool).expand_as(entity_attention)
            entity_attention[mask] = 0

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(self, token_input_ids, token_attention_mask, entity_ids = None, entity_attention_mask = None, seg_encoding = None,
        pixel_values = None, aux_values = None, output_attentions = None, output_hidden_states = None, return_dict = None,):
        # vision embedding
        # pre vision
        # bs,61,768
        vision_embedding_output = self.vision_embeddings(pixel_values, aux_values)
        vision_embedding_output = self.vision_pre_layrnorm(vision_embedding_output)


        # text embedding
        word_embeddings = self.embeddings(token_input_ids, token_type_ids = seg_encoding) #bert embedding

        if entity_ids is None:
            entity_ids = torch.arange(self.config.entity_queries_num, device=token_input_ids.device).expand((token_input_ids.size(0), -1))
            entity_attention_mask = torch.ones(entity_ids.size(), dtype=torch.long, device=token_input_ids.device)

        entity_embeddings = self.entity_embeddings(entity_ids) #query embedding
        attention_mask = self._compute_extended_attention_mask(token_attention_mask, entity_attention_mask, seg_mask=None)

        query_pos = None
        if self.config.use_entity_pos:
            query_pos = self.pos_entity_embeddings(entity_ids)
        # entity_embeddings = query_pos
        return self.encoder(word_embeddings, entity_embeddings, attention_mask, query_pos = query_pos,
                            vision_embeds=vision_embedding_output,output_attentions=output_attentions,output_hidden_states=output_hidden_states, return_dict=return_dict,)



class RobertaEntityAwareBertModel(RobertaPreTrainedModel):
    config_class = EntityAwareBertConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.config.type_vocab_size = 1
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = EntityAwareBertEncoder(config)

        self.entity_embeddings = EntityEmbeddings(config)
        if config.use_entity_pos:
            self.pos_entity_embeddings = EntityEmbeddings(config, is_pos_embedding = True)
            # self.pos_entity_embeddings.entity_embeddings = self.entity_embeddings.entity_embeddings

        # self.register_buffer("entity_ids", torch.arange(config.entity_queries_num).expand((1, -1)))

        # self.init_weights()


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _compute_extended_attention_mask(self, word_attention_mask: torch.LongTensor, entity_attention_mask: torch.LongTensor, mask_ent2tok = None, mask_tok2ent = None, mask_ent2ent = None, mask_entself = None, seg_mask = None):
        # pdb.set_trace()
        attention_mask = word_attention_mask
        if entity_attention_mask is not None:
            attention_mask = torch.cat([attention_mask, entity_attention_mask], dim=1)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        word_num = word_attention_mask.size(1)
        entity_num = entity_attention_mask.size(1)
        # #batch x #head x seq_len x seq_len
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).expand(-1, -1, word_num+entity_num, -1).clone()

        # mask entity2token attention
        if (mask_ent2tok == None and self.config.mask_ent2tok) or mask_ent2tok == True:
            extended_attention_mask[:, :, :word_num, word_num:] = 0

        if  (mask_tok2ent == None and self.config.mask_tok2ent) or mask_tok2ent == True:
            extended_attention_mask[:, :, word_num:, :word_num] = 0

        if seg_mask != None:
            tok2ent_mask = extended_attention_mask[:, :, word_num:, :word_num]
            seg_mask = seg_mask.bool().unsqueeze(1).unsqueeze(2).expand_as(tok2ent_mask)
            extended_attention_mask[:, :, word_num:, :word_num] = seg_mask

        if  (mask_ent2ent == None and self.config.mask_ent2ent) or mask_ent2ent == True:
            entity_attention = extended_attention_mask[:, :, word_num:, word_num:]
            mask = torch.eye(entity_num, entity_num, dtype = torch.bool).expand_as(entity_attention)
            entity_attention[~mask] = 0

        if (mask_entself == None and self.config.mask_entself) or mask_entself == True:
            entity_attention = extended_attention_mask[:, :, word_num:, word_num:]
            mask = torch.eye(entity_num, entity_num, dtype = torch.bool).expand_as(entity_attention)
            entity_attention[mask] = 0

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        return extended_attention_mask

    def forward(self, token_input_ids, token_attention_mask, entity_ids = None, entity_attention_mask = None, seg_encoding = None):
        # word_embeddings = self.embeddings(token_input_ids, token_type_ids = seg_encoding)
        word_embeddings = self.embeddings(token_input_ids, token_type_ids = None)


        if entity_ids is None:
            entity_ids = torch.arange(self.config.entity_queries_num, device=token_input_ids.device).expand((token_input_ids.size(0), -1))
            entity_attention_mask = torch.ones(entity_ids.size(), dtype=torch.long, device=token_input_ids.device)

        entity_embeddings = self.entity_embeddings(entity_ids)
        attention_mask = self._compute_extended_attention_mask(token_attention_mask, entity_attention_mask, seg_mask=None)

        query_pos = None
        if self.config.use_entity_pos:
            query_pos = self.pos_entity_embeddings(entity_ids)
        # entity_embeddings = query_pos
        return self.encoder(word_embeddings, entity_embeddings, attention_mask, query_pos = query_pos)

class MQSPN(nn.Module):


    def __init__(self, model_type, text_config, embed: torch.tensor, entity_type_count: int,
                 prop_drop: float, freeze_transformer: bool, pos_size: int = 25, char_lstm_layers: int = 1,
                 char_lstm_drop: int = 0.2, char_size: int = 25, use_glove: bool = True, use_pos: bool = True,
                 use_char_lstm: bool = True, lstm_layers=3, pool_type: str = "max", word_mask_tok2ent=None,
                 word_mask_ent2tok=None, word_mask_ent2ent=None, word_mask_entself=None, share_query_pos=False,
                 use_token_level_encoder=True, num_token_entity_encoderlayer=1, use_entity_attention=False,
                 use_masked_lm=False, use_aux_loss=False, use_lstm=False, use_mask_query = False,inlcude_subword_aux_loss=False,
                 last_layer_for_loss=3, split_epoch=0, vision_config = None, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.text_config = text_config
        self.vision_config = vision_config
        self.model_type = model_type
        if model_type == "roberta":
            self.roberta = RobertaEntityAwareBertModel(text_config)
            self.model = self.roberta
            if use_masked_lm:
                self.lm_head = RobertaLMHead(text_config)
                self.mlm_head = lambda *args, **kwagrs: self.lm_head(*args, **kwagrs)
            # self.model = lambda *args, **kwagrs: self.roberta(*args, **kwagrs)

        elif model_type == "bert":
            self.bert = EntityAwareBertModel(text_config, vision_config)
            self.model = self.bert
            if use_masked_lm:
                self.cls = BertOnlyMLMHead(text_config)
                self.mlm_head = lambda *args, **kwagrs: self.cls(*args, **kwagrs)
            # self.model = lambda *args, **kwagrs: self.bert(*args, **kwagrs)
        self._keys_to_ignore_on_save = ["model." + k for k,v in self.model.named_parameters()]
        self._keys_to_ignore_on_load_unexpected = ["model." + k for k,v in self.model.named_parameters()]
        self._keys_to_ignore_on_load_missing = ["model." + k for k,v in self.model.named_parameters()]
        self.use_masked_lm = use_masked_lm
        self._entity_type_count = entity_type_count
        self.prop_drop = prop_drop
        if embed is not None:
            self.wordvec_size = embed.size(-1)
        self.pos_size = pos_size
        self.use_glove = use_glove
        self.use_pos = use_pos
        self.char_lstm_layers = char_lstm_layers
        self.char_lstm_drop = char_lstm_drop
        self.char_size = char_size
        self.use_char_lstm = use_char_lstm
        self._share_query_pos = share_query_pos
        self.use_token_level_encoder = use_token_level_encoder
        self.num_token_entity_encoderlayer = num_token_entity_encoderlayer
        self.split_epoch = split_epoch
        self.use_entity_attention = use_entity_attention
        self.use_aux_loss = use_aux_loss
        self.use_lstm = use_lstm
        self.use_mask_query = use_mask_query

        self.word_mask_tok2ent = word_mask_tok2ent
        self.word_mask_ent2tok = word_mask_ent2tok
        self.word_mask_ent2ent = word_mask_ent2ent
        self.word_mask_entself = word_mask_entself


        # lstm_input_size = 0
        lstm_input_size = text_config.hidden_size
        # assert use_glove or use_pos or use_char_lstm, "At least one be True"


        if use_glove:
            lstm_input_size += self.wordvec_size
        if use_pos:
            lstm_input_size += self.pos_size
            self.pos_embedding = nn.Embedding(100, pos_size)
        if use_char_lstm:
            lstm_input_size += self.char_size * 2
            self.char_lstm = nn.LSTM(input_size = char_size, hidden_size = char_size, num_layers = char_lstm_layers,  bidirectional = True, dropout = char_lstm_drop, batch_first = True)
            self.char_embedding = nn.Embedding(103, char_size)

        if not self.use_lstm and (use_glove or use_pos or use_char_lstm):
            self.reduce_dimension = nn.Linear(lstm_input_size, text_config.hidden_size)

        if self.use_lstm:
            self.lstm = nn.LSTM(input_size = lstm_input_size, hidden_size = text_config.hidden_size//2, num_layers = lstm_layers,  bidirectional = True, dropout = 0.5, batch_first = True)

        if self.use_lstm and inlcude_subword_aux_loss:
            self.subword_lstm = nn.LSTM(input_size = lstm_input_size, hidden_size = text_config.hidden_size//2, num_layers = lstm_layers,  bidirectional = True, dropout = 0.5, batch_first = True)

        cls_size = text_config.hidden_size

        self.pool_type = pool_type

        self.dropout = nn.Dropout(self.prop_drop)

        # self.entity_classifier = nn.Sequential(
        #     # nn.Linear(cls_size, config.hidden_size),
        #     nn.GELU(),
        #     nn.Linear(cls_size, entity_type_count)
        # )
        if use_mask_query:
            self.query_positions, self.query_embed = self._prompt_query()


        self.entity_classifier = EntityTypePredictor(text_config, cls_size, entity_type_count)

        self.left_boundary_classfier = EntityBoundaryPredictor(text_config)
        self.right_boundary_classfier = EntityBoundaryPredictor(text_config)
        self.region_pointer_classifier = RegionBoundaryPredictor(vision_config, text_config)

        if not self._share_query_pos and self.use_token_level_encoder:
            self.pos_entity_embeddings = EntityEmbeddings(text_config, is_pos_embedding = True)
            # self.pos_entity_embeddings.entity_embeddings = self.model.pos_entity_embeddings.entity_embeddings


        if self.use_token_level_encoder:
            self.selfcrossattention = SelfCrossAttention(text_config, vision_config, use_token_level_encoder, use_entity_attention, self.num_token_entity_encoderlayer)

        # self.init_weights()

        if use_glove:
            self.wordvec_embedding = nn.Embedding.from_pretrained(embed, freeze=False)

        self.freeze_transformer = freeze_transformer
        self.split_epoch = split_epoch
        self.has_changed = False
        self.inlcude_subword_aux_loss = inlcude_subword_aux_loss
        self.last_layer_for_loss = last_layer_for_loss

        if freeze_transformer or self.split_epoch > 0:
            print("Freeze transformer weights")
            # if self.model_type == "bert":
            #     model = self.bert
            # if self.model_type == "roberta":
            #     model = self.roberta
            for name, param in self.model.named_parameters():
                if "entity" not in name:
                    param.requires_grad = False

        self.register_buffer("entity_ids", torch.arange(self.text_config.entity_queries_num))
        self.register_buffer("entity_attention_mask", torch.ones(self.text_config.entity_queries_num))

    def combine(self, sub, sup_mask, pool_type = "max" ):
        sup = None
        if len(sub.shape) == len(sup_mask.shape) :
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub.unsqueeze(1).repeat(1, sup_mask.shape[1], 1, 1)
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        else:
            if pool_type == "mean":
                size = (sup_mask == 1).float().sum(-1).unsqueeze(-1) + 1e-30
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2) / size
            if pool_type == "sum":
                m = (sup_mask.unsqueeze(-1) == 1).float()
                sup = m * sub
                sup = sup.sum(dim=2)
            if pool_type == "max":
                m = (sup_mask.unsqueeze(-1) == 0).float() * (-1e30)
                sup = m + sub
                sup = sup.max(dim=2)[0]
                sup[sup==-1e30]=0
        return sup

    def _prompt_query(self):
        # prompt query objects
        query_input_ids, query_attention_mask, query_positions = self.prompt_query()
        query_attention_mask = query_attention_mask.to(self.device)
        with torch.no_grad():
            query_objects_states = self.bert(
                query_input_ids,
                query_attention_mask,
            )[0]
            _, query_mask_ids = (query_input_ids == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
            query_objects_states = query_objects_states[torch.arange(len(query_input_ids)), query_mask_ids]
        return query_positions, query_objects_states

    def prompt_query(self):
        # prepare query object inputs
        template = self._template()
        ent2type = self._entity2types()
        ent2num = self._entity2num()
        query_input_token = []
        query_positions = []
        for ent in ent2type:
            for i in range(ent2num[ent]):
                query_input_token.append(template.format(ent2type[ent]))
                query_positions.append(i)
        query_positions = torch.tensor(query_positions)
        query_inputs = self.tokenizer(query_input_token, padding=True, return_tensors='pt')
        query_input_ids, query_attention_mask = query_inputs['input_ids'], query_inputs['attention_mask']
        # get prompt entity 2num
        query_ent2span = {}
        ent_num_cums = torch.tensor(list(ent2num.values())).cumsum(dim=0)
        for i, ent in enumerate(ent2num.keys()):
            if i == 0:
                query_ent2span[ent] = (0, ent_num_cums[i].item() - 1)
            else:
                query_ent2span[ent] = (ent_num_cums[i - 1].item(), ent_num_cums[i].item() - 1)
        return query_input_ids, query_attention_mask, query_positions

    def _common_forward(self, encodings: torch.tensor, context_masks: torch.tensor, seg_encoding: torch.tensor,
                        context2token_masks:torch.tensor, token_masks:torch.tensor, pos_encoding: torch.tensor = None,
                        wordvec_encoding:torch.tensor = None, char_encoding:torch.tensor = None, token_masks_char = None, char_count:torch.tensor = None,
                        image = None, aux_imgs = None, region_mask = None):
        context_masks = context_masks.float()
        # set_trace

        # instances query
        entity_ids = self.entity_ids.expand(encodings.size(0), -1)
        entity_attention_mask = self.entity_attention_mask.expand(encodings.size(0), -1)
        # h是token encoding, h_entity是queries encoding, intermediate是 encoder中间的各层输出
        h, h_entity, intermediate_subword_entity, h_region = self.model(token_input_ids=encodings, token_attention_mask=context_masks, entity_ids = entity_ids, entity_attention_mask = entity_attention_mask, seg_encoding = seg_encoding,
                                                              pixel_values = image, aux_values = aux_imgs)
        # h_entity = h_entity + self.bert.entity_embeddings.entity_embeddings.weight

        masked_seq_logits = None
        if self.use_masked_lm and self.training:
            masked_seq_logits = self.mlm_head(h)

        batch_size = encodings.shape[0]
        token_count = token_masks.long().sum(-1,keepdim=True)
        # 实际句中token向量
        h_token = self.combine(h, context2token_masks, self.pool_type)

        intermediate_word_entity = []
        for subword_entity_dic in intermediate_subword_entity:
            entity = subword_entity_dic["h_entity"]
            token = self.combine(subword_entity_dic["h_token"], context2token_masks, self.pool_type)
            intermediate_word_entity.append({"h_token":token, "h_entity":entity})

        def add_other_embedding(h_token, char_count, token_masks_char, char_encoding):
            embeds = [h_token]

            if self.use_pos:
                pos_embed = self.pos_embedding(pos_encoding)
                pos_embed = self.dropout(pos_embed)
                embeds.append(pos_embed)
            if self.use_glove:
                word_embed = self.wordvec_embedding(wordvec_encoding)
                word_embed = self.dropout(word_embed)
                embeds.append(word_embed)
            if self.use_char_lstm:
                char_count = char_count.view(-1)
                token_masks_char = token_masks_char
                max_token_count = char_encoding.size(1)
                max_char_count = char_encoding.size(2)

                char_encoding = char_encoding.view(max_token_count*batch_size, max_char_count)

                char_encoding[char_count==0][:, 0] = 101
                char_count[char_count==0] = 1
                char_embed = self.char_embedding(char_encoding)
                char_embed = self.dropout(char_embed)
                char_embed_packed = nn.utils.rnn.pack_padded_sequence(input = char_embed, lengths = char_count.tolist(), enforce_sorted = False, batch_first = True)
                char_embed_packed_o, (_, _) = self.char_lstm(char_embed_packed)
                char_embed, _ = nn.utils.rnn.pad_packed_sequence(char_embed_packed_o, batch_first=True)
                char_embed = char_embed.view(batch_size, max_token_count, max_char_count, self.char_size * 2)
                h_token_char = self.combine(char_embed, token_masks_char, "mean")
                embeds.append(h_token_char)

            h_token = torch.cat(embeds, dim = -1)
            if len(embeds)>1 and not self.use_lstm:
                h_token = self.reduce_dimension(h_token)
            return h_token

        h_token = add_other_embedding(h_token, char_count, token_masks_char, char_encoding)

        if self.use_lstm:
            h_token = nn.utils.rnn.pack_padded_sequence(input = h_token, lengths = token_count.squeeze(-1).cpu().tolist(), enforce_sorted = False, batch_first = True)
            h_token, (_, _) = self.lstm(h_token)
            h_token, _ = nn.utils.rnn.pad_packed_sequence(h_token, batch_first=True)

        entity_attention_mask = torch.ones(h_entity.size()[:-1], dtype=torch.long, device=h_entity.device)
        # token_entity_attention_mask = self.model._compute_extended_attention_mask(token_masks, entity_attention_mask, mask_ent2tok = None, mask_tok2ent = None, mask_ent2ent = None, mask_entself = True)
        token_entity_attention_mask = self.model._compute_extended_attention_mask(token_masks, entity_attention_mask, mask_ent2tok =  self.word_mask_ent2tok, mask_tok2ent =  self.word_mask_tok2ent, mask_ent2ent =  self.word_mask_ent2ent, mask_entself = self.word_mask_entself)

        query_pos = None
        if self.text_config.use_entity_pos and self._share_query_pos and self.use_token_level_encoder:
            query_pos = self.model.pos_entity_embeddings(entity_ids)

        if not self._share_query_pos and self.use_token_level_encoder:
            query_pos = self.pos_entity_embeddings(entity_ids)

        intermediate = []
        if self.use_token_level_encoder:
            h_token, h_entity, h_region, intermediate = self.selfcrossattention(h_token, h_entity, token_entity_attention_mask, query_pos = query_pos, visual_hidden_state=h_region, output_qks=True)

        intermediate_word_entity = intermediate_word_entity[-self.last_layer_for_loss:]
        if self.inlcude_subword_aux_loss:
            for i, v in enumerate(intermediate_word_entity):
                h_token, h_entity = v["h_token"], v["h_entity"]
                h_token = add_other_embedding(h_token, char_count, token_masks_char, char_encoding)
                if self.use_lstm:
                    h_token = nn.utils.rnn.pack_padded_sequence(input = h_token, lengths = token_count.squeeze(-1).cpu().tolist(), enforce_sorted = False, batch_first = True)
                    h_token, (_, _) = self.subword_lstm(h_token)
                    h_token, _ = nn.utils.rnn.pad_packed_sequence(h_token, batch_first=True)
                    v = {"h_token": h_token, "h_entity":h_entity}
                intermediate.insert(i, v)
        output = []
        # intermediate is for auxiliary loss
        if self.use_aux_loss and len(intermediate) != 0:
            for h_dict in intermediate:
                h_token, h_entity, h_region = h_dict["h_token"], h_dict["h_entity"], h_dict["h_region"]
                p_left = self.left_boundary_classfier(h_token, h_entity, token_masks)
                p_right = self.right_boundary_classfier(h_token, h_entity, token_masks)
                entity_logits, _ = self.entity_classifier(h_entity, h_token, p_left, p_right, token_masks)
                p_region = self.region_pointer_classifier(h_region, h_entity, region_mask)
                output.append({"p_left": p_left, "p_right": p_right, "entity_logits": entity_logits, "p_region": p_region})
        else:
            p_left = self.left_boundary_classfier(h_token, h_entity, token_masks)
            p_right = self.right_boundary_classfier(h_token, h_entity, token_masks)
            entity_logits, _ = self.entity_classifier(h_entity, h_token, p_left, p_right, token_masks)
            p_region = self.region_pointer_classifier(h_region, h_entity, region_mask)
            output = [{"p_left": p_left, "p_right": p_right, "entity_logits": entity_logits, "p_region": p_region}]
        # entity_logits = self.entity_classifier(h_entity)


        return entity_logits, p_left, p_right, p_region, masked_seq_logits, output

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, seg_encoding: torch.tensor,  context2token_masks:torch.tensor,
                       token_masks:torch.tensor, epoch, pos_encoding: torch.tensor = None,  wordvec_encoding:torch.tensor = None,
                       char_encoding:torch.tensor = None, token_masks_char = None, char_count:torch.tensor = None,
                       image = None, aux_imgs = None, region_mask = None):
        if not self.has_changed and epoch >= self.split_epoch and not self.freeze_transformer:
            print("Now, update bert weights")
            for name, param in self.model.named_parameters():
                param.requires_grad = True
            self.has_changed = True

        return self._common_forward(encodings, context_masks, seg_encoding, context2token_masks, token_masks, pos_encoding,
                        wordvec_encoding, char_encoding, token_masks_char, char_count, image, aux_imgs, region_mask)

    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, seg_encoding: torch.tensor, context2token_masks:torch.tensor,
                      token_masks:torch.tensor, pos_encoding: torch.tensor = None, wordvec_encoding:torch.tensor = None, char_encoding:torch.tensor = None,
                      token_masks_char = None, char_count:torch.tensor = None,
                      image = None, aux_imgs = None, region_mask = None):
        return self._common_forward(encodings, context_masks, seg_encoding, context2token_masks, token_masks, pos_encoding,
                        wordvec_encoding, char_encoding, token_masks_char, char_count, image, aux_imgs, region_mask)

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)

    def _template(self):
        return "{} is an entity type about [MASK]."

    def _entity2types(self):
        return {'ORG':'organization', 'MISC':'other', 'PER':'person', 'LOC':'location'}

    def _entity2num(self):
        return {'ORG':0, 'MISC':1, 'PER':2, 'LOC':3}

class BertMQSPN(MQSPN):

    config_class = BertConfig
    base_model_prefix = "bert"
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, *args, **kwagrs):
        super().__init__("bert", *args, **kwagrs)

class RobertaMQSPN(MQSPN):

    config_class = RobertaConfig
    base_model_prefix = "roberta"
    authorized_missing_keys = [r"position_ids"]

    def __init__(self, *args, **kwagrs):
        super().__init__("roberta", *args, **kwagrs)

_MODELS = {
    'MQSPN': BertMQSPN,
    'roberta_MQSPN': RobertaMQSPN
}

def get_model(name):
    return _MODELS[name]
