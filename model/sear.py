import os
from copy import deepcopy
import numpy as np
from recbole.utils import FeatureType

from recbole.model.abstract_recommender import SequentialRecommender
import math
from sklearn.decomposition import PCA

from mamba_ssm import Mamba
from sentence_transformers import SentenceTransformer
from model.loss import *

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.num_heads
        tensor = tensor.view(batch_size, length, self.num_heads, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class AggregationLayer(nn.Module):
    def __init__(self, d_model, num_heads, drop_prob=0.1):
        super().__init__()
        self.agg_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_model * 4, drop_prob)

        self.dropout1 = nn.Dropout(p=drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, q_tokens, k_v_tokens, masks=None):
        # Attention for q_tokens
        q_tokens = q_tokens + self.dropout1(self.agg_attn(q_tokens, k_v_tokens, k_v_tokens, masks))
        q_tokens = self.norm1(q_tokens)

        # Feed Forward Network
        q_tokens = q_tokens + self.dropout2(self.ffn(q_tokens))
        q_tokens = self.norm2(q_tokens)

        return q_tokens


class MambaLayer(nn.Module):
    def __init__(self, d_model, d_state, d_conv, expand, drop_prob, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.mamba = Mamba(
            # This module uses roughly 3 * expand * d_model^2 parameters
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.ffn = FeedForward(d_model=d_model, inner_size=d_model * 4, drop_prob=drop_prob)

        self.dropout1 = nn.Dropout(drop_prob)
        self.dropout2 = nn.Dropout(drop_prob)
        self.LayerNorm1 = nn.LayerNorm(d_model, eps=1e-12)
        self.LayerNorm2 = nn.LayerNorm(d_model, eps=1e-12)

    def forward(self, hidden_states):
        # if self.num_layers == 1:  # one Mamba layer without residual connection
        #     hidden_states = self.dropout1(self.mamba(hidden_states))
        # else:  # stacked Mamba layers with residual connections
        hidden_states = self.dropout1(self.mamba(hidden_states)) + hidden_states
        hidden_states = self.LayerNorm1(hidden_states)

        hidden_states = hidden_states+ self.dropout2(self.ffn(hidden_states))
        hidden_states = self.LayerNorm2(hidden_states)

        return hidden_states


class FeedForward(nn.Module):
    def __init__(self, d_model, inner_size, drop_prob=0.2):
        super().__init__()
        self.w_1 = nn.Linear(d_model, inner_size)
        self.w_2 = nn.Linear(inner_size, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, input_tensor):
        hidden_states = self.w_1(input_tensor)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.w_2(hidden_states)
        return hidden_states

class SEAR(SequentialRecommender):
    def __init__(self, config, dataset):
        super(SEAR, self).__init__(config, dataset)

        self.device = torch.device(config['device'])

        self.rating_list_field = dataset.rating_list_field
        # load parameters info
        self.dropout_prob = config["dropout_prob"]

        self.other_parameter_name = ['LLM_embedding']

        self.item_sem_embedding_model_file = f"{os.path.split(config['item_sem_embedding_model_path'])[-1]}-{dataset.dataset_name}-{dataset.inter_num}-{dataset.item_num}-{dataset.user_num}.pt"

        if os.path.exists(self.item_sem_embedding_model_file):
             self.LLM_embedding = torch.load(self.item_sem_embedding_model_file, map_location=self.device)
        else:
            self.get_item_sem_embedding(config, dataset)
            torch.save(self.LLM_embedding, self.item_sem_embedding_model_file)

        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]
        self.agg_num_heads = config['agg_num_heads']

        # Hyperparameters for Mamba block
        self.d_state = config["d_state"]
        self.d_conv = config["d_conv"]
        self.expand = config["expand"]

        self.collaborative_emb = nn.Embedding(
            self.n_items, self.hidden_size, padding_idx=0
        )

        self.item_emb_LayerNorms =  nn.ModuleList([nn.LayerNorm(self.hidden_size, eps=1e-12) for _ in range(3)])
        self.item_emb_dropouts =  nn.ModuleList([nn.Dropout(self.dropout_prob)  for _ in range(3)])

        self.adapter = nn.Linear(self.LLM_embedding.shape[-1], self.hidden_size)

        self.mamba_layers = nn.ModuleList([
            nn.ModuleList([
            MambaLayer(
                d_model=self.hidden_size,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand,
                drop_prob=self.dropout_prob,
                num_layers=self.num_layers,
            ) for _ in range(self.num_layers)])  for _ in range(3)
        ])

        self.agg_layers = nn.ModuleList([ nn.ModuleList([
            AggregationLayer(
                d_model= self.hidden_size,
                num_heads = self.agg_num_heads,
                drop_prob=self.dropout_prob,
            ) for _ in range(self.num_layers)
        ]) for _ in range(3)
        ])

        self.rating_emb = nn.Linear(1,self.hidden_size)

        self.item_decoder = nn.Linear(self.hidden_size * 3, self.n_items, device=self.device, bias = False)

        self.user_item_loss_fct = KLDivergenceLoss(lambda_weight=config['lambda'])

        self.apply(self._init_weights)


        pca = PCA(n_components=self.hidden_size)
        self.collaborative_emb.weight = nn.Parameter(
                torch.tensor(pca.fit_transform(self.LLM_embedding.detach().cpu().numpy())))

        pca2 = PCA(n_components=self.hidden_size * 3)
        self.item_decoder.weight = nn.Parameter(
                torch.tensor(pca2.fit_transform(self.LLM_embedding.detach().cpu().numpy())))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


    def get_item_sem_embedding(self, config, dataset):
        self.logger.info('Modeling semantic information of item.')
        ### Item ontology features
        item_sen = "Below is the item's information: "
        for k, v in dataset.item_feat.interaction.items():
            if k == config['ITEM_ID_FIELD']:
                continue

            if dataset.field2type[k] == FeatureType.FLOAT or dataset.field2type[k] == FeatureType.FLOAT_SEQ:
                feat_sen = v.numpy().astype(str)
            else:
                field2id_token = deepcopy(dataset.field2id_token[k])
                field2id_token[0] = ''
                feat_sen = field2id_token[v]

            if dataset.field2type[k] == FeatureType.TOKEN_SEQ or dataset.field2type[k] == FeatureType.FLOAT_SEQ:
                feat_sen = np.array([" ".join(filter(None, feat_sen[i])) for i in range(feat_sen.shape[0])])

            # REPLACE THE not given feature
            if dataset.field2type[k] == FeatureType.TOKEN_SEQ or dataset.field2type[k] == FeatureType.TOKEN:
                feat_sen[feat_sen == ''] = 'N/A'
            if dataset.field2type[k] == FeatureType.FLOAT:
                feat_sen[feat_sen == 'inf'] = 'N/A'

            if isinstance(item_sen, str):
                item_sen = np.char.add(item_sen + f"*{k}*: ",
                                       feat_sen.astype('U200'))  # feat_sen.astype('U200'): Truncate long names
            else:
                item_sen = np.char.add(item_sen, f"; *{k}*: ")
                item_sen = np.char.add(item_sen, feat_sen.astype('U200'))
        # print(item_sen)

        model_file = config['item_sem_embedding_model_path']
        model = SentenceTransformer(model_file).to(self.device)
        model.eval()

        self.LLM_embedding = torch.tensor(model.encode(list(item_sen)), device=self.device)

        del model


    def forward(self, emb_list, item_seq_len):
        view_shape = emb_list[0].shape

        for j in range(self.num_layers):
            for i, embedding in enumerate(emb_list):
                emb_list[i] = embedding.view((-1,self.hidden_size))

            all_emb = torch.stack(emb_list, dim = 1)

            for i, embedding in enumerate(emb_list):
                emb_list[i] = self.agg_layers[i][j](embedding.unsqueeze(1), all_emb).squeeze().view(view_shape)


            for i,embedding in enumerate(emb_list):
                emb_list[i] = self.mamba_layers[i][j](embedding)


        item_emb = torch.concatenate(emb_list, dim=-1)

        user_embedding = self.gather_indexes(item_emb, item_seq_len - 1)

        return user_embedding

    def calculate_loss(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        target_id_list = interaction['target_id_list']
        rating_seq = interaction[self.rating_list_field]


        item_seq = item_seq[:, :item_seq_len.max()]
        rating_seq = rating_seq[:, :item_seq_len.max()]

        emb_list =[]
        emb_list.append(self.adapter(self.LLM_embedding)[item_seq])
        emb_list.append(self.collaborative_emb(item_seq))
        emb_list.append(self.rating_emb(rating_seq.unsqueeze(-1)))

        for i, embeding in enumerate(emb_list):
            emb_list[i] =  self.item_emb_LayerNorms[i](self.item_emb_dropouts[i](embeding))

        user_embedding = self.forward(emb_list, item_seq_len)

        logits = self.item_decoder(user_embedding) # [B, n_items]

        user_item_loss = self.user_item_loss_fct(logits, target_id_list)

        return user_item_loss


    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        rating_seq = interaction[self.rating_list_field]

        item_seq = item_seq[:, :item_seq_len.max()]
        rating_seq = rating_seq[:, :item_seq_len.max()]

        emb_list = []
        emb_list.append(self.adapter(self.LLM_embedding)[item_seq])
        emb_list.append(self.collaborative_emb(item_seq))
        emb_list.append(self.rating_emb(rating_seq.unsqueeze(-1)))

        for i, embeding in enumerate(emb_list):
            emb_list[i] = self.item_emb_LayerNorms[i](self.item_emb_dropouts[i](embeding))

        user_embedding = self.forward(emb_list, item_seq_len)


        scores = self.item_decoder(user_embedding) # [B, n_items]
        return scores

    def predict(self, interaction):
        raise NotImplementedError("Functon 'predict' is not implemented")
