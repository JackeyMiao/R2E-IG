
import torch
import torch.nn as nn
import torch.nn.functional as F


class TSPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = TSP_Encoder(**model_params)
        self.decoder = TSP_Decoder(**model_params)
        if self.model_params['CE_Loss']:
            self.predictor = nn.Linear(model_params['embedding_dim'], 3)

        self.encoded_nodes = None
        # shape: (batch, problem, EMBEDDING_DIM)
        # embed_loss and hidden_loss
        if model_params['embedding_dim'] != 128:
            self.W_hidden = nn.Linear(64, 128, bias=False)
            self.W_embed = nn.Linear(64, 128, bias=False)

    def pre_forward(self, reset_state, attn_type=None):
        # Expert Distribution Initialization
        if self.model_params['encoder_moe'] and self.model_params['is_moe']:
            for i in range(self.model_params['encoder_layer_num']):
                self.encoder.layers[i].feed_forward._load.zero_()
        if self.model_params['decoder_moe'] and self.model_params['is_moe']:
            self.decoder.multi_head_combine._load.zero_()


        # Encoder
        if attn_type is not None:  # layer loss
            if attn_type == 'no':  # no attn loss
                embedding, self.encoded_nodes, moe_loss = self.encoder(reset_state.problems, attn_type)
            else:  # attn loss
                embedding, self.encoded_nodes, attn, moe_loss = self.encoder(reset_state.problems, attn_type)
            hidden = self.encoded_nodes
            if self.model_params['embedding_dim'] != 128:
                embedding = self.W_embed(embedding)
                hidden = self.W_hidden(hidden)
        else:
            self.encoded_nodes, moe_loss = self.encoder(reset_state.problems)
        # shape: (batch, problem, EMBEDDING_DIM)
        self.decoder.set_kv(self.encoded_nodes)

        self.decoder.count = 0
        self.aux_loss = moe_loss

        if attn_type is not None:
            return embedding, hidden, attn

    def compute_ce_loss(self, minibatch_size, distribution, cumulate_weight):
        # CE Loss
        input_prediction = self.decoder.graph_embedding[:,0,:]
        logits_prediction = self.predictor(input_prediction)

        if distribution == 'mix_three':
            idx_uni = int(minibatch_size * cumulate_weight[0])      # [0, idx_uni) -> uniform
            idx_clu = int(minibatch_size * cumulate_weight[1])      # [idx_uni, idx_clu) -> cluster
            idx_mix = int(minibatch_size * cumulate_weight[2]) 
            uni_size = idx_uni
            clu_size = idx_clu - idx_uni
            mix_size = idx_mix - idx_clu
            total_size = uni_size + clu_size + mix_size
            assert total_size == input_prediction.size(0)

            dist_label = torch.empty(total_size, dtype=torch.long)
            dist_label[:idx_uni] = 0
            dist_label[idx_uni:idx_clu] = 1
            dist_label[idx_clu:idx_mix] = 2
        else:
            dist_label = torch.empty(total_size, dtype=torch.long)
            if distribution == 'uniform':
                dist_label[:] = 0
            elif distribution == 'cluster':
                dist_label[:] = 1
            elif distribution == 'mixed':
                dist_label[:] = 2
            else:
                raise NotImplementedError
        
        ce_loss = F.cross_entropy(logits_prediction, dist_label)
        return ce_loss
    
    def forward(self, state, route=None, return_probs=False, teacher=False,selected_count=None):
        batch_size = state.BATCH_IDX.size(0)
        pomo_size = state.BATCH_IDX.size(1)

        if state.current_node is None:
            selected = torch.arange(pomo_size)[None, :].expand(batch_size, pomo_size)
            prob = torch.ones(size=(batch_size, pomo_size))
            if return_probs:
                probs = torch.ones(size=(batch_size, pomo_size, self.encoded_nodes.size(1)))

            encoded_first_node = _get_encoding(self.encoded_nodes, selected)
            # shape: (batch, pomo, embedding)
            self.decoder.set_q1(encoded_first_node)

        else:
            encoded_last_node = _get_encoding(self.encoded_nodes, state.current_node)
            # shape: (batch, pomo, embedding)
            probs, moe_loss = self.decoder(encoded_last_node, ninf_mask=state.ninf_mask, last_node_idx=state.current_node)
            # shape: (batch, pomo, problem)
            self.aux_loss += moe_loss

            if route is None:
                if self.training or self.model_params['eval_type'] == 'softmax':
                    selected = probs.reshape(batch_size * pomo_size, -1).multinomial(1) \
                        .squeeze(dim=1).reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                    prob = probs[state.BATCH_IDX, state.POMO_IDX, selected] \
                        .reshape(batch_size, pomo_size)
                    # shape: (batch, pomo)
                else:
                    if teacher:
                        selected = probs.argmax(dim=2)
                        # shape: (batch, pomo)
                        prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)
                    else:
                        selected = probs.argmax(dim=2)
                        # shape: (batch, pomo)
                        prob = None  # value not needed. Can be anything.
            else:
                selected = route[:, :,selected_count].reshape(batch_size, pomo_size).long()
                prob = probs[state.BATCH_IDX, state.POMO_IDX, selected].reshape(batch_size, pomo_size)

        if return_probs:
            return selected, prob, probs
        return selected, prob


def _get_encoding(encoded_nodes, node_index_to_pick):
    # encoded_nodes.shape: (batch, problem, embedding)
    # node_index_to_pick.shape: (batch, pomo)

    batch_size = node_index_to_pick.size(0)
    pomo_size = node_index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)

    gathering_index = node_index_to_pick[:, :, None].expand(batch_size, pomo_size, embedding_dim)
    # shape: (batch, pomo, embedding)

    picked_nodes = encoded_nodes.gather(dim=1, index=gathering_index)
    # shape: (batch, pomo, embedding)

    return picked_nodes


########################################
# ENCODER
########################################

class TSP_Encoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        encoder_layer_num = self.model_params['encoder_layer_num']

        self.embedding = nn.Linear(2, embedding_dim)
        self.layers = nn.ModuleList([EncoderLayer(**model_params) for _ in range(encoder_layer_num)])

    def forward(self, data,attn_type =None):
        # data.shape: (batch, problem, 2)

        embedded_input = self.embedding(data)
        # shape: (batch, problem, embedding)

        out = embedded_input
        if attn_type is not None:
            embedding = out
            if attn_type != 'no':
                count = 0
        moe_loss = 0
        for layer in self.layers:
            if attn_type is None or attn_type == 'no':
                out, loss = layer(out)
            else:
                count += 1  # current loop
                if count == self.model_params['encoder_layer_num']:  # output attn
                    out, attn, loss = layer([out, count, attn_type])
                else:
                    out, loss = layer([out, count, attn_type])
            moe_loss += loss

        if attn_type is not None and attn_type == 'no':  # layer loss
            return embedding, out, moe_loss
        elif attn_type is not None and attn_type != 'no':  # attn loss
            return embedding, out, attn, moe_loss
        elif attn_type is None:
            return out, moe_loss


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.addAndNormalization1 = Add_And_Normalization_Module(**model_params)
        if self.model_params['is_moe'] and self.model_params['encoder_moe']:
            self.feed_forward = MOELayer(**model_params)
        else:
            self.feed_forward = Feed_Forward_Module(**model_params)
        self.addAndNormalization2 = Add_And_Normalization_Module(**model_params)

    def forward(self, input1):
        # input.shape: (batch, problem, EMBEDDING_DIM)
        head_num = self.model_params['head_num']

        # attn
        if isinstance(input1,list):
            input1, count, attn_type = input1
        else:
            attn_type = None

        q = reshape_by_heads(self.Wq(input1), head_num=head_num)
        k = reshape_by_heads(self.Wk(input1), head_num=head_num)
        v = reshape_by_heads(self.Wv(input1), head_num=head_num)
        # q shape: (batch, HEAD_NUM, problem, KEY_DIM)

        out_concat = multi_head_attention(q, k, v, attn_type=attn_type)
        # shape: (batch, problem, HEAD_NUM*KEY_DIM)
        if isinstance(out_concat,list):
            out_concat, attn = out_concat

        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch, problem, EMBEDDING_DIM)

        out1 = self.addAndNormalization1(input1, multi_head_out)
        out2, moe_loss = self.feed_forward(out1)
        out3 = self.addAndNormalization2(out1, out2)

        if attn_type is None or attn_type == 'no':
            return out3, moe_loss
            # shape: (batch, problem, embedding)
        else:
            # torch.save(attn,'./result/cvrp50/saved_CVRP50_model/attn_{}.pt'.format(count))
            if  count == self.model_params['encoder_layer_num']:
                return out3, attn, moe_loss
            else:
                return out3, moe_loss
        # shape: (batch, problem, EMBEDDING_DIM)


########################################
# DECODER
########################################

class TSP_Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        embedding_dim = self.model_params['embedding_dim']
        num_experts = self.model_params['num_experts']
        top_k = self.model_params['top_k']
        head_num = self.model_params['head_num']
        qkv_dim = self.model_params['qkv_dim']

        self.Wq_first = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wq_last = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        self.Wq_g = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wk_g = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)
        self.Wv_g = nn.Linear(embedding_dim, head_num * qkv_dim, bias=False)

        if self.model_params['is_moe'] and self.model_params['decoder_moe']:
            if self.model_params['router_method'] == 'instance':
                self.instance_router = Router(embedding_dim, num_experts, top_k)
            self.multi_head_combine = MOELayer(**model_params)
            
        else:
            self.multi_head_combine = nn.Linear(head_num * qkv_dim, embedding_dim)

        self.k = None  # saved key, for multi-head attention
        self.v = None  # saved value, for multi-head_attention
        self.single_head_key = None  # saved, for single-head attention
        self.q_first = None  # saved q1, for multi-head attention
        self.count = 0


    def set_kv(self, encoded_nodes):
        # encoded_nodes.shape: (batch, problem, embedding)
        head_num = self.model_params['head_num']
        q = reshape_by_heads(self.Wq_g(encoded_nodes), head_num=head_num)
        k = reshape_by_heads(self.Wk_g(encoded_nodes), head_num=head_num)
        v = reshape_by_heads(self.Wv_g(encoded_nodes), head_num=head_num)
        # qkv shape: (batch, head_num, problem, qkv_dim)

        self.graph_embedding = multi_head_attention(q, k, v).mean(dim = 1, keepdim = True).repeat((1, encoded_nodes.size(1), 1))

        if self.model_params['is_moe'] and self.model_params['router_method'] == 'instance' and self.model_params['decoder_moe']:
            self.topk_weights, self.topk_indices, self.probs = self.instance_router(self.graph_embedding)
            B, N, E = self.graph_embedding.size()
            self.histogram_expert = self.topk_indices.reshape(B, N, -1)

        self.q_graph = reshape_by_heads(self.graph_embedding, head_num=head_num)


        self.k = reshape_by_heads(self.Wk(encoded_nodes), head_num=head_num)
        self.v = reshape_by_heads(self.Wv(encoded_nodes), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)
        self.single_head_key = encoded_nodes.transpose(1, 2)
        # shape: (batch, embedding, problem)

    def set_q1(self, encoded_q1):
        # encoded_q.shape: (batch, n, embedding)  # n can be 1 or pomo
        head_num = self.model_params['head_num']

        self.q_first = reshape_by_heads(self.Wq_first(encoded_q1), head_num=head_num)
        # shape: (batch, head_num, n, qkv_dim)

    def forward(self, encoded_last_node, ninf_mask, last_node_idx=None):
        # encoded_last_node.shape: (batch, pomo, embedding)
        # ninf_mask.shape: (batch, pomo, problem)

        head_num = self.model_params['head_num']
        moe_loss = 0
        self.count += 1


        #  Multi-Head Attention
        #######################################################
        # shape = (batch, group, EMBEDDING_DIM+1)

        q_last = reshape_by_heads(self.Wq_last(encoded_last_node), head_num=head_num)
        # shape: (batch, head_num, pomo, qkv_dim)

        q = q_last


        # q = self.q_first + q_last
        # shape: (batch, head_num, pomo, qkv_dim)

        out_concat = multi_head_attention(q, self.k, self.v, rank3_ninf_mask=ninf_mask)
        # shape: (batch, pomo, head_num*qkv_dim)

        if self.model_params['is_moe']  and self.model_params['decoder_moe']:
            if self.model_params['router_method'] == 'instance':
                g = {
                        'topk_weights': self.topk_weights, 
                        'topk_indices': self.topk_indices,
                        'probs': self.probs
                    }
                mh_atten_out, moe_loss = self.multi_head_combine(out_concat, g)
            else:
                mh_atten_out, moe_loss = self.multi_head_combine(out_concat)
        else:
            mh_atten_out = self.multi_head_combine(out_concat)


        # mh_atten_out = self.multi_head_combine(out_concat)
        # shape: (batch, pomo, embedding)

        #  Single-Head Attention, for probability calculation
        #######################################################
        score = torch.matmul(mh_atten_out, self.single_head_key)
        # shape: (batch, pomo, problem)

        sqrt_embedding_dim = self.model_params['sqrt_embedding_dim']
        logit_clipping = self.model_params['logit_clipping']

        score_scaled = score / sqrt_embedding_dim
        # shape: (batch, pomo, problem)

        score_clipped = logit_clipping * torch.tanh(score_scaled)

        score_masked = score_clipped + ninf_mask

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, pomo, problem)

        return probs, moe_loss


########################################
# NN SUB CLASS / FUNCTIONS
########################################

def reshape_by_heads(qkv, head_num):
    # q.shape: (batch, n, head_num*key_dim)   : n can be either 1 or PROBLEM_SIZE

    batch_s = qkv.size(0)
    n = qkv.size(1)

    q_reshaped = qkv.reshape(batch_s, n, head_num, -1)
    # shape: (batch, n, head_num, key_dim)

    q_transposed = q_reshaped.transpose(1, 2)
    # shape: (batch, head_num, n, key_dim)

    return q_transposed


def multi_head_attention(q, k, v, rank2_ninf_mask=None, rank3_ninf_mask=None,attn_type=None):
    # q shape: (batch, head_num, n, key_dim)   : n can be either 1 or PROBLEM_SIZE
    # k,v shape: (batch, head_num, problem, key_dim)
    # rank2_ninf_mask.shape: (batch, problem)
    # rank3_ninf_mask.shape: (batch, group, problem)

    batch_s = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)

    input_s = k.size(2)

    score = torch.matmul(q, k.transpose(2, 3))
    # shape: (batch, head_num, n, problem)

    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if attn_type == 'qk_scaled':
        qk_scaled = score_scaled
    if rank2_ninf_mask is not None:
        score_scaled = score_scaled + rank2_ninf_mask[:, None, None, :].expand(batch_s, head_num, n, input_s)
    if rank3_ninf_mask is not None:
        score_scaled = score_scaled + rank3_ninf_mask[:, None, :, :].expand(batch_s, head_num, n, input_s)

    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch, head_num, n, problem)

    out = torch.matmul(weights, v)
    # shape: (batch, head_num, n, key_dim)

    out_transposed = out.transpose(1, 2)
    # shape: (batch, n, head_num, key_dim)

    out_concat = out_transposed.reshape(batch_s, n, head_num * key_dim)
    # shape: (batch, n, head_num*key_dim)

    if attn_type == 'qk_scaled':
        return [out_concat, qk_scaled]
    elif attn_type == 'add_mask':
        return [out_concat, score_scaled]
    elif attn_type is None:
        return out_concat
    else:
        assert 0, 'Attention loss type not defined!'


class Add_And_Normalization_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # input.shape: (batch, problem, embedding)

        added = input1 + input2
        # shape: (batch, problem, embedding)

        transposed = added.transpose(1, 2)
        # shape: (batch, embedding, problem)

        normalized = self.norm(transposed)
        # shape: (batch, embedding, problem)

        back_trans = normalized.transpose(1, 2)
        # shape: (batch, problem, embedding)

        return back_trans


class Feed_Forward_Module(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        embedding_dim = model_params['embedding_dim']
        ff_hidden_dim = model_params['ff_hidden_dim']

        self.W1 = nn.Linear(embedding_dim, ff_hidden_dim)
        self.W2 = nn.Linear(ff_hidden_dim, embedding_dim)

    def forward(self, input1):
        # input.shape: (batch, problem, embedding)

        return self.W2(F.relu(self.W1(input1)))



##### MOE BLOCKS #####
class ResExpert(nn.Module):
    def __init__(self, embedding_dim, intermediate_dim):
        super(ResExpert, self).__init__()
        self.fc1 = nn.Linear(embedding_dim, intermediate_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(intermediate_dim, embedding_dim)
        self.shortcut = nn.Linear(embedding_dim, embedding_dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x))) + self.shortcut(x)
    
class ExpertNetwork(nn.Module):
    def __init__(
            self,
            embedding_dim,
            intermediate_dim,
    ):
        super(ExpertNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.intermediate_dim = intermediate_dim
    
        self.linear1 = nn.Linear(embedding_dim, intermediate_dim)
        self.linear2 = nn.Linear(intermediate_dim, embedding_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        output = self.linear2(x)
        return output




class Router(nn.Module):
    def __init__(
            self,
            embedding_dim,
            num_experts,
            top_k,
    ):
        super(Router, self).__init__()
        self.embedding_dim = embedding_dim
        self.router_logits = nn.Linear(embedding_dim, num_experts)
        self.top_k = top_k

    def forward(self, x, method='top_k'):
        x = x.reshape(-1, self.embedding_dim)

        logits = self.router_logits(x)
        probs = F.softmax(logits, dim=-1)
        if method == 'top_k':
            top_k_probs, top_k_indices = torch.topk(probs, self.top_k, dim=-1, sorted=False)
            top_k_weights = top_k_probs/(top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)
        elif method == 'sampling':
            # Sampling k experts based on probabilities
            top_k_indices = torch.multinomial(probs, self.top_k, replacement=False)  # [BN, top_k]
            
            # Compute the weights for each sampled expert
            top_k_probs = torch.gather(probs, dim=-1, index=top_k_indices)  # [BN, top_k]
            top_k_weights = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)  # Normalize to get the weights
        else:
            raise NotImplementedError
        
        return top_k_weights, top_k_indices, probs


class MOELayer(nn.Module):
    def __init__(
            self,
            embedding_dim,
            intermediate_dim,
            num_experts,
            top_k,
            used_shared_expert,
            loading_balance_loss,
            expert_method,
            type_expert,
            **kwargs
    ):
        super(MOELayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.used_shared_expert = used_shared_expert
        self.loading_balance_loss = loading_balance_loss
        self.expert_method = expert_method
        self.MODULE_REGISTRY = {
            "original": ExpertNetwork,
            "Res": ResExpert,
        }

        self.experts = nn.ModuleList([self.MODULE_REGISTRY[type_expert](embedding_dim, intermediate_dim) for _ in range(num_experts)])

        if used_shared_expert:
            self.shared_expert = self.MODULE_REGISTRY[type_expert](embedding_dim, intermediate_dim)
        else:
            self.shared_expert = None
        
        self.router = Router(embedding_dim, num_experts, top_k)

        self.register_buffer('_load', torch.zeros(num_experts))  # buffer

        # self.aux_loss = None
        # self._last_aux_terms = None
    
    
    def forward(self, x, g=None, return_histogram=False):
        aux = 0
        B, N, D = x.size()
        token_num = B * N
        if g is None:
            topk_weights, topk_indices, probs = self.router(x)  # probs: [token_num, expert_nums]
        else:
            topk_weights, topk_indices, probs = g['topk_weights'], g['topk_indices'], g['probs'] 

        
        # Flatten
        x_flat = x.reshape(token_num, D)

        x_flat = x_flat.repeat_interleave(self.top_k, dim=0)
        topk_indices_flat = topk_indices.view(-1)


        output = torch.zeros_like(x_flat)
        
        for i, expert in enumerate(self.experts):
            output[topk_indices_flat == i] = expert(x_flat[topk_indices_flat == i])
        output = (output.view(*topk_weights.shape, -1) * topk_weights.unsqueeze(-1)).sum(dim=1)

        if self.used_shared_expert:
            shared_expert = self.shared_expert(x.reshape(token_num, D))
            output += shared_expert
        
        # ------- Loading Balance Loss -------
        if self.loading_balance_loss:
            importance = probs.mean(dim=0)  # [expert_nums]

            # load: Frequency of each expert (Top-k), Normalize to [0,1]
            # Construct one-hot： shape [T, K, E]
            one_hot = torch.zeros(token_num, self.top_k, self.num_experts, device=probs.device, dtype=probs.dtype)
            one_hot.scatter_(2, topk_indices.unsqueeze(-1), 1.0)
            load = one_hot.mean(dim=(0, 1))

            # Record expert distribution
            self._load += load.detach()

            # GShard/Switch aux loss
            aux = self.num_experts * torch.sum(importance * load)

            # self.aux_loss = aux
            # self._last_aux_terms = {'importance': importance.detach(), 'load': load.detach()}
            
        output = output.view(B, N, D)
        if not return_histogram:
            return output, aux
        else:
            return output, aux, topk_indices.reshape(B, N, -1)

    
