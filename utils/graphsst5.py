import torch
from torch.utils.data import Dataset
from .utils_algo import generate_uniform_cv_candidate_labels

from datasets.graphsst2_dataset import get_dataset,get_dataloader
from torch_geometric.data import Data,Batch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset

from ..data.createLabel import createNoise

def custom_collate(batch):
    elem = batch[0]
    if isinstance(elem, Data):
        return Batch.from_data_list(batch)
    else:
        raise TypeError(f"Unsupported datatype found in batch: {type(elem)}")

def custom_collate2(batch):
    elem = batch[0]
    if isinstance(elem[0], Data):
        graph_dataBatch1 = Batch.from_data_list([item[0] for item in batch])
        # graph_dataBatch2 = Batch.from_data_list([item[1] for item in batch])
        labels = [item[1].unsqueeze(dim=0) for item in batch]
        labels = torch.cat(labels,dim=0)
        true_labels = [item[2].unsqueeze(dim=0) for item in batch]
        true_labels = torch.cat(true_labels,dim=0)
        index = [torch.tensor(item[3]).unsqueeze(dim=0) for item in batch]
        index = torch.cat(index,dim=0)
        return graph_dataBatch1, labels, true_labels, index
    else:
        raise TypeError(f"Unsupported datatype found in batch: {type(elem)}")

def load_loader(dataset_name,partial_rate, batch_size, uniform, in_channels, modelPath, degree_bias=False, data_split_ratio=[0.8, 0.1, 0.1], seed=2):
    if dataset_name == 'Graph_SST5':
        dataset = get_dataset('./data','Graph_SST5')
    elif dataset_name == 'Graph_Twitter':
        dataset = get_dataset('./data','Graph_Twitter')
    elif dataset_name == 'COLLAB':
        dataset = load_COLLAB()
    elif dataset_name == 'REDDIT_MULTI_5K':
        dataset = load_REDDIT5K()
    else:
        raise NotImplementedError("You have chosen an unsupported dataset. Please check and try again.")

    if degree_bias:
        train, test = [], []
        for g in dataset:
            if g.num_edges <= 2: continue
            degree = float(g.num_edges) / g.num_nodes
            if degree >= 1.76785714:
                train.append(g)
            elif degree <= 1.57142857:
                test.append(g)
        
        eval = train[:int(len(train) * 0.1)]
        train = train[int(len(train) * 0.1):]
        print(len(train), len(eval), len(test))
    else:
        num_train = int(data_split_ratio[0] * len(dataset))
        num_eval = int(data_split_ratio[1] * len(dataset))
        num_test = len(dataset) - num_train - num_eval

        train, eval, test = random_split(dataset, lengths=[num_train, num_eval, num_test],
                                         generator=torch.Generator().manual_seed(seed))

    val_loader = torch.utils.data.DataLoader(dataset=eval, 
        batch_size=batch_size,
        shuffle=False,
        # num_workers=4,
        sampler= torch.utils.data.distributed.DistributedSampler(eval, shuffle=False),
        collate_fn=custom_collate)
    # set val dataloader
    test_loader = torch.utils.data.DataLoader(dataset=test, 
        batch_size=batch_size, 
        shuffle=False, 
        # num_workers=4,
        sampler= torch.utils.data.distributed.DistributedSampler(test, shuffle=False),
        collate_fn=custom_collate)
    # set test dataloader

    label_lis = [g.y for g in train]
    labels = torch.cat(label_lis, dim=0)
    # get labels
    
    if uniform: partialY = generate_uniform_cv_candidate_labels(labels, partial_rate)
    else: createNoise(train, in_channels, partial_rate, modelPath, topk=3)

    # generate partial labels
    temp = torch.zeros(partialY.shape)
    temp[torch.arange(partialY.shape[0]), labels] = 1
    if torch.sum(partialY * temp) == partialY.shape[0]:
        print('partialY correctly loaded')
    else:
        print('inconsistent permutation')

    print('Average candidate num: ', partialY.sum(1).mean())
    partial_matrix_dataset = SST_Augmentention(train, partialY.float(), labels.float())
    # generate partial label dataset

    train_sampler = torch.utils.data.distributed.DistributedSampler(partial_matrix_dataset)
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        # num_workers=4,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=custom_collate2)

    return partial_matrix_train_loader,partialY,train_sampler,test_loader,val_loader


def load_COLLAB():
    dataset = TUDataset(root='data/', name='COLLAB', use_node_attr = True, use_edge_attr = True)
    updated_dataset = [] 
    for index, g in enumerate(dataset):
        g.x = torch.rand(g.num_nodes,768)
        g.edge_attr = torch.ones(g.edge_index.shape[1],1)
        updated_dataset.append(g)
    return updated_dataset

def load_REDDIT5K():
    dataset = TUDataset(root='data/', name='REDDIT-MULTI-5K', use_node_attr = True, use_edge_attr = True)
    updated_dataset = [] 
    for index, g in enumerate(dataset):
        g.x = torch.rand(g.num_nodes,32)
        g.edge_attr = torch.ones(g.edge_index.shape[1],1)
        updated_dataset.append(g)
    return updated_dataset

class SST_Augmentention(Dataset):
    def __init__(self, graphs, given_label_matrix, true_labels):
        self.graphs = graphs
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels

    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        graph = self.graphs[index]
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return graph, each_label, each_true_label, index

