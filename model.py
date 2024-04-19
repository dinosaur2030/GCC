import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn import Linear, ReLU
from torch_geometric.nn import ARMAConv, global_mean_pool
import torch.nn.functional as F
from torch_geometric.data import Data,Batch
from torch.nn import ModuleList
from utils.get_subgraph import split_graph, relabel
from utils.mask import set_masks, clear_masks

class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hid_channels=128, feat_dim=128, num_class=2, num_layers=1):
        super(GraphEncoder, self).__init__()

        self.convs = ModuleList([
            ARMAConv(in_channels, hid_channels),
            ARMAConv(hid_channels, hid_channels, num_layers=num_layers)])
        self.head = torch.nn.Sequential(
            Linear(hid_channels, 2*hid_channels),
            ReLU(),
            Linear(2*hid_channels, feat_dim)
        )
        self.fc = Linear(hid_channels, num_class)

    def forward(self, x, edge_index, edge_attr, batch):
        node_x = self.get_node_reps(x, edge_index, edge_attr, batch)
        graph_x = global_mean_pool(node_x, batch)

        feat_c = self.head(graph_x)
        # logits = self.fc(graph_x)
        return graph_x, F.normalize(feat_c, dim=1)

        # return self.get_causal_pred(graph_x)
    
    def get_pred(self, graph_x):
        logits = self.fc(graph_x)
        return logits
    
    def get_node_reps(self, x, edge_index, edge_attr, batch):
        edge_weight = edge_attr.view(-1)
        temp = self.convs[0](x, edge_index, edge_weight)
        x = F.relu(temp)
        node_x = self.convs[1](x, edge_index, edge_weight)
        return node_x

    def reset_parameters(self):
        with torch.no_grad():
            for param in self.parameters():
                param.uniform_(-1.0, 1.0)

    def save(self, path):
        torch.save(self.state_dict(), path)


class CausalAttNet(nn.Module):
    
    def __init__(self, args):
        super(CausalAttNet, self).__init__()
        self.conv1 = ARMAConv(in_channels=args.in_channels, out_channels=args.channels)
        self.conv2 = ARMAConv(in_channels=args.channels, out_channels=args.channels)
        self.mlp = nn.Sequential(
            nn.Linear(args.channels*2, args.channels*4),
            nn.ReLU(),
            nn.Linear(args.channels*4, 1)
        )
        self.ratio = args.ratio
    
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr.view(-1)))
        x = self.conv2(x, data.edge_index, data.edge_attr.view(-1))
        
        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_score = self.mlp(edge_rep).view(-1)

        (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        (conf_edge_index, conf_edge_attr, conf_edge_weight) = split_graph(data, edge_score, self.ratio)

        causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)
        conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, data.batch)

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch),\
                edge_score


class GCCL(nn.Module):

    def __init__(self, args, base_encoder=GraphEncoder):
        super().__init__()

        self.num_class = args.num_class

        self.att_net = CausalAttNet(args)

        self.encoder_q = base_encoder(in_channels=args.channels, hid_channels=args.hid_channels, feat_dim=args.feat_dim, num_class=args.num_class)
        # momentum encoder
        self.encoder_k = base_encoder(in_channels=args.channels, hid_channels=args.hid_channels, feat_dim=args.feat_dim, num_class=args.num_class)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            # param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim))
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))        
        self.register_buffer("prototypes", torch.zeros(args.num_class,args.low_dim))
        
        # 样本权重
        self.register_buffer("queue_weight", torch.zeros(args.moco_queue))

        self.queue = F.normalize(self.queue, dim=0)

    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args):
        # gather keys before updating queue
        keys = concat_all_gather_tensor(keys)
        labels = concat_all_gather_tensor(labels)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert args.moco_queue % batch_size == 0  # for simplicity

        # 
        # self.queue_weight[ptr:ptr + batch_size] = weight

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % args.moco_queue  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.num_graphs
        x_gather = concat_all_gather_batch(x)

        batch_size_all = x_gather.num_graphs

        # batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather_tensor(x)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    # def forward(self, graphs_q, graphs_k=None, partial_Y=None, args=None, eval_only=False):
        
    def forward(self, graphs, partial_Y=None, args=None, eval_only=False):

        (graphs_q_x, graphs_q_edge_index, graphs_q_edge_attr, graphs_q_edge_weight, graphs_q_batch),\
        (graphs_k_x, graphs_k_edge_index, graphs_k_edge_attr, graphs_k_edge_weight, graphs_k_batch), edge_score = self.att_net(graphs)

        set_masks(graphs_q_edge_weight, self.encoder_q)
        graph_q, q = get_sub_graph_x(self.encoder_q, graphs_q_x, graphs_q_edge_index, graphs_q_edge_attr, graphs_q_batch)
        output_q = self.encoder_q.get_pred(graph_q)
        clear_masks(self.encoder_q)

        if eval_only:
            return output_q
        # for testing

        predicted_scores = torch.softmax(output_q, dim=1) * partial_Y
        max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)
        # using partial labels to filter out negative labels

        # compute protoypical logits
        prototypes = self.prototypes.clone().detach()
        logits_prot = torch.mm(q, prototypes.t())
        score_prot = torch.softmax(logits_prot, dim=1)

        # update momentum prototypes with pseudo labels
        for feat, label in zip(concat_all_gather_tensor(q), concat_all_gather_tensor(pseudo_labels_b)):
            self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat
        # normalize prototypes    
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        
        # compute key features 
        with torch.no_grad():  # no gradient 
            self._momentum_update_key_encoder(args)  # update the momentum encoder
            # shuffle for making use of BN
            # graphs_k, idx_unshuffle = self._batch_shuffle_ddp(graphs_k)
            # _, k = self.encoder_k(graphs_k.x, graphs_k.edge_index, graphs_k.edge_attr, graphs_k.batch)

        set_masks(graphs_k_edge_weight, self.encoder_k)
        graph_k, k = get_sub_graph_x(self.encoder_k, graphs_k_x, graphs_k_edge_index, graphs_k_edge_attr, graphs_k_batch)
        output_k = self.encoder_q.get_pred(graph_k)
        clear_masks(self.encoder_q)

        # negative samples loss
        output_k = torch.softmax(output_k, dim=1)
        k_ = k.clone().detach()

        features = torch.cat((q, k_, self.queue.clone().detach()), dim=0)
        pseudo_labels_k = torch.full((pseudo_labels_b.shape[0],), self.num_class).cuda()
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_k, self.queue_pseudo.clone().detach()), dim=0)

        self._dequeue_and_enqueue(k_, pseudo_labels_k, args)
        self._dequeue_and_enqueue(q, pseudo_labels_b, args)

        return output_q, output_k, features, pseudo_labels, score_prot

    def test(self,graphs):
        (graphs_q_x, graphs_q_edge_index, graphs_q_edge_attr, graphs_q_edge_weight, graphs_q_batch),\
        (graphs_k_x, graphs_k_edge_index, graphs_k_edge_attr, graphs_k_edge_weight, graphs_k_batch), edge_score = self.att_net(graphs)

        set_masks(graphs_q_edge_weight, self.encoder_q)
        output_q, q = get_sub_graph_x(self.encoder_q, graphs_q_x, graphs_q_edge_index, graphs_q_edge_attr, graphs_q_batch)
        clear_masks(self.encoder_q)

        set_masks(graphs_k_edge_weight, self.encoder_k)
        output_k, k = get_sub_graph_x(self.encoder_k, graphs_k_x, graphs_k_edge_index, graphs_k_edge_attr, graphs_k_batch)
        clear_masks(self.encoder_q)

        return q,k

def get_sub_graph_x(encoder, sub_graph_x, sub_graph_edge_index, sub_graph_edge_attr, sub_graph_batch):
    graph_q, q = encoder(sub_graph_x, sub_graph_edge_index, sub_graph_edge_attr, sub_graph_batch)
    return graph_q, q


# # utils
@torch.no_grad()
def concat_all_gather_tensor(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

# utils
@torch.no_grad()
def concat_all_gather_batch(dataBatch):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    dataBatch_gather = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(dataBatch_gather, dataBatch)

    return dataBatch_gather