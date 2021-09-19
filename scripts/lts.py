import os
import sys
sys.path.append("..") 
import pickle
import numpy as np
import torch_geometric.utils as tg
import networkx as nx
import torch_geometric
from utils import *
from tqdm import tqdm
import argparse
import scipy
import multiprocessing as mp
from skopt import gp_minimize
from skopt.plots import plot_convergence

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser(description='Training GCN on Cora/CiteSeer/PubMed/Reddit Datasets')

'''
    Dataset arguments
'''
parser.add_argument('--dataset', type=str, default='cora',
                    help='Dataset name: cora/citeseer/pubmed/Reddit')
parser.add_argument('--nhid', type=int, default=256,
                    help='Hidden state dimension')
parser.add_argument('--epoch_num', type=int, default= 100,
                    help='Number of Epoch')
parser.add_argument('--pool_num', type=int, default= 10,
                    help='Number of Pool')
parser.add_argument('--batch_num', type=int, default= 10,
                    help='Maximum Batch Number')
parser.add_argument('--batch_size', type=int, default=512,
                    help='size of output node in a batch')
parser.add_argument('--n_layers', type=int, default=5,
                    help='Number of GCN layers')
parser.add_argument('--n_iters', type=int, default=1,
                    help='Number of iteration to run on a batch')
parser.add_argument('--n_stops', type=int, default=200,
                    help='Stop after number of batches that f1 dont increase')
parser.add_argument('--samp_num', type=int, default=64,
                    help='Number of sampled nodes per layer')
parser.add_argument('--sample_method', type=str, default='lts',
                    help='Sampled Algorithms: ladies/fastgcn/full/lts')
parser.add_argument('--cuda', type=int, default=-1,
                    help='Avaiable GPU ID')
parser.add_argument('--sample_norm', type=str, default='L1',
                    help='Norm of sampling probability: L1/softmax')
parser.add_argument('--meta_param_norm', dest='meta_param_norm', action='store_true',
                    help='Norm of meta-parameters: T/F')
parser.add_argument('--model_tag', type=str, default='test',
                    help='name of folder to save the results')
parser.add_argument('--opt_iter', type=int, default=10,
                    help='Number of iterations for optimization')

args = parser.parse_args()

print (args.model_tag)
save_dir = '../results/' + args.model_tag
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print ('directory created!')

class GraphConvolution(nn.Module):
    def __init__(self, n_in, n_out, bias=True):
        super(GraphConvolution, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in,  n_out)
    def forward(self, x, adj):
        out = self.linear(x)
        return F.elu(torch.spmm(adj, out))


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, layers, dropout):
        super(GCN, self).__init__()
        self.layers = layers
        self.nhid = nhid
        self.gcs = nn.ModuleList()
        self.gcs.append(GraphConvolution(nfeat,  nhid))
        self.dropout = nn.Dropout(dropout)
        for i in range(layers-1):
            self.gcs.append(GraphConvolution(nhid,  nhid))
    def forward(self, x, adjs):
        '''
            The difference here with the original GCN implementation is that
            we will receive different adjacency matrix for different layer.
        '''
        for idx in range(len(self.gcs)):
            x = self.dropout(self.gcs[idx](x, adjs[idx]))
        return x

class SuGCN(nn.Module):
    def __init__(self, encoder, num_classes, dropout, inp):
        super(SuGCN, self).__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.linear  = nn.Linear(self.encoder.nhid, num_classes)
    def forward(self, feat, adjs):
        x = self.encoder(feat, adjs)
        x = self.dropout(x)
        x = self.linear(x)
        return x


def fastgcn_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    '''
        FastGCN_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is pre-computed based on the global degree (lap_matrix)
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    #     pre-compute the sampling probability (importance) based on the global degree (lap_matrix)
    pi = np.array(np.sum(lap_matrix.multiply(lap_matrix), axis=0))[0]
    p = pi / np.sum(pi)
    '''
        Sample nodes from top to bottom, based on the pre-computed probability. Then reconstruct the adjacency matrix.
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     sample the next layer's nodes based on the pre-computed probability (p).
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.         
        adj = row_norm(U[: , after_nodes].multiply(1/p[after_nodes]))
        #     Turn the sampled adjacency matrix into a sparse matrix. If implemented by PyG
        #     This sparse matrix can also provide index and value.
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes

def ladies_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth, prob_norm=None):
    '''
        LADIES_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is computed adaptively according to the nodes sampled in the upper layer.
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    '''
    for d in range(depth):
        #     row-select the lap_matrix (U) by previously sampled nodes
        U = lap_matrix[previous_nodes , :]
        #     Only use the upper layer's neighborhood to calculate the probability.
        pi = np.array(np.sum(U.multiply(U), axis=0))[0]
        p = pi / np.sum(pi)
        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     Add output nodes for self-loop
        after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.      
        adj = U[: , after_nodes].multiply(1/p[after_nodes])
        adjs += [sparse_mx_to_torch_sparse_tensor(row_normalize(adj))]
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes

def lts_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth, prob_norm):
    '''
        LTS_Sampler: Sample a fixed number of nodes per layer. The sampling probability (importance)
                         is computed adaptively according to the nodes sampled in the upper layer.
    '''
    np.random.seed(seed)
    previous_nodes = batch_nodes
    adjs  = []
    '''
        Sample nodes from top to bottom, based on the probability computed adaptively (layer-dependent).
    '''
    for d in range(depth):
        
        ''' finding the neighbors. We will find an array of size number of nodes in the graph
            in which only neighors of previous can have non zero probability. Non neighbors 
            should have zero probability. 
        '''
        temp_lap = lap_matrix
        # selecting only rows of nodes previous nodes to find neighbors of only those nodes.
        temp = temp_lap[previous_nodes, :]
        # Summing will give us non zero entries in respective columns if they are neighbors
        # ravel will convert it to 1D array
        nbrs = np.ravel(np.sum(temp, axis = 0))
        # entries in array/list can be lesser/greater than 1, making them 1 if node is neighbor
        # 0 elsewise
        for i in range(len(nbrs)):
            if nbrs[i]>0:
                nbrs[i] = 1
        # Multiply elementwise will give us array with neigbors with probability, 0 for non neighbors
        pi = np.multiply(nbrs, prob_norm)    
        # normalization
        p = pi / np.linalg.norm(pi, ord = 1)

        s_num = np.min([np.sum(p > 0), samp_num_list[d]])
        #     sample the next layer's nodes based on the adaptively probability (p).
        after_nodes = np.random.choice(num_nodes, s_num, p = p, replace = False)
        #     Add output nodes for self-loop
        after_nodes = np.unique(np.concatenate((after_nodes, batch_nodes)))
        #     col-select the lap_matrix (U), and then devided by the sampled probability for 
        #     unbiased-sampling. Finally, conduct row-normalization to avoid value explosion.      
        adj = temp[: , after_nodes].multiply(1/p[after_nodes])
        new_adj = sparse_mx_to_torch_sparse_tensor(row_normalize(adj))
        adjs += [new_adj]
        
        #     Turn the sampled nodes as previous_nodes, recursively conduct sampling.
        previous_nodes = after_nodes
    #     Reverse the sampled probability from bottom to top. Only require input how the lastly sampled nodes.
    adjs.reverse()
    return adjs, previous_nodes, batch_nodes

def default_sampler(seed, batch_nodes, samp_num_list, num_nodes, lap_matrix, depth):
    mx = sparse_mx_to_torch_sparse_tensor(lap_matrix)
    return [mx for i in range(depth)], np.arange(num_nodes), batch_nodes


def prepare_data(pool, sampler, process_ids, train_nodes, valid_nodes, samp_num_list, num_nodes, lap_matrix, depth, prob_norm):
    jobs = []
    for _ in process_ids:
        idx = torch.randperm(len(train_nodes))[:args.batch_size]
        batch_nodes = train_nodes[idx]
        p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes, samp_num_list, num_nodes, lap_matrix, depth, prob_norm))
        jobs.append(p)
    idx = torch.randperm(len(valid_nodes))[:args.batch_size]
    batch_nodes = valid_nodes[idx]
    p = pool.apply_async(sampler, args=(np.random.randint(2**32 - 1), batch_nodes, samp_num_list * 20, num_nodes, lap_matrix, depth, prob_norm))
    jobs.append(p)
    return jobs

def package_mxl(mxl, device):
    return [torch.sparse.FloatTensor(mx[0], mx[1], mx[2]).to(device) for mx in mxl]


f = open(os.path.join(save_dir, 'summary.txt'), 'w')

if args.cuda != -1:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

f.write(args.dataset + ' ' + args.sample_method + '\n')

edges, labels, feat_data, num_classes, train_nodes, valid_nodes, test_nodes = load_data(args.dataset, base_dir='../')

adj_matrix = get_adj(edges, feat_data.shape[0])

lap_matrix = row_normalize(adj_matrix + sp.eye(adj_matrix.shape[0]))
print (feat_data.shape)
if type(feat_data) == scipy.sparse.lil.lil_matrix:
    feat_data = torch.FloatTensor(feat_data.todense()).to(device) 
else:
    feat_data = torch.FloatTensor(feat_data).to(device)
labels    = torch.LongTensor(labels).to(device) 


if args.sample_method == 'ladies':
    sampler = ladies_sampler
elif args.sample_method == 'lts':
    sampler = lts_sampler
elif args.sample_method == 'fastgcn':
    sampler = fastgcn_sampler
elif args.sample_method == 'full':
    sampler = default_sampler


# converting to torch graph
adj_torch = torch_geometric.data.Data(edge_index = torch.tensor([edges[:,0], edges[:,1]]))

if not os.path.exists(os.path.join(save_dir, 'feature_set.pkl')):
    # converting to networkx graph
    adj_nx = tg.to_networkx(adj_torch)
    adj_nx.num_nodes = feat_data.shape[0]

    # calculating centralities
    eigen_cen = list(nx.eigenvector_centrality(adj_nx).values())
    bet_cen = list(nx.betweenness_centrality(adj_nx).values())
    clos_cen = list(nx.closeness_centrality(adj_nx).values())
    deg_cen = list(nx.degree_centrality(adj_nx).values())
    cen = [eigen_cen, bet_cen, clos_cen, deg_cen]
    print ('Feature set calculated!!')
    f.write('Feature set calculated!!' + '\n')
    feature_set = np.array(cen)
    with open(os.path.join(save_dir, 'feature_set.pkl'), 'wb') as handle:
        pickle.dump(feature_set, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    print ('using cached data!')
    pkl_loader = open(os.path.join(save_dir, 'feature_set.pkl'),"rb")
    feature_set = pickle.load(pkl_loader)

i = 0
def optimization_function(x):
    global feature_set
    global i
    start = time.time()
    print('iteration number', i)
    f.write('iteration number: ' + str(i) + '\n')
    i += 1

    ## should we normalise here?
    if (args.meta_param_norm):
        print ('meta_param_norm')
        x = x / np.sum(x)
    params = np.zeros(feature_set.shape) + np.array(x).reshape(-1,1)

    prob_unnorm = np.sum(params * feature_set, axis = 0)
    ## take norm or take softmax
    if (args.sample_norm == 'softmax'):
        prob_norm = np.exp(prob_unnorm) / sum(np.exp(prob_unnorm))
    elif (args.sample_norm == 'L1'):
        prob_norm = prob_unnorm / np.linalg.norm( prob_unnorm, ord = 1 )

    f.write('-------------------------------------------------------------------' + '\n')
    f.write('params = ' + str(x) + '\n')

    process_ids = np.arange(args.batch_num)
    samp_num_list = np.array([args.samp_num, args.samp_num, args.samp_num, args.samp_num, args.samp_num])

    pool = mp.Pool(args.pool_num)
    jobs = prepare_data(pool, sampler, process_ids, train_nodes, valid_nodes, samp_num_list, len(feat_data), lap_matrix, args.n_layers, prob_norm)
    all_res = []
    for oiter in range(1):
        encoder = GCN(nfeat = feat_data.shape[1], nhid=args.nhid, layers=args.n_layers, dropout = 0.2).to(device)
        susage  = SuGCN(encoder = encoder, num_classes=num_classes, dropout=0.5, inp = feat_data.shape[1])
        susage.to(device)
        
        optimizer = optim.Adam(filter(lambda p : p.requires_grad, susage.parameters()))
        best_val = 0
        best_tst = -1
        cnt = 0
        times = []
        res   = []
        f.write('-' * 10 + '\n')
        for epoch in np.arange(args.epoch_num):
            susage.train()
            train_losses = []
            train_data = [job.get() for job in jobs[:-1]]
            valid_data = jobs[-1].get()
            pool.close()
            pool.join()
            pool = mp.Pool(args.pool_num)
            '''
                Use CPU-GPU cooperation to reduce the overhead for sampling. (conduct sampling while training)
            '''
            jobs = prepare_data(pool, sampler, process_ids, train_nodes, valid_nodes, samp_num_list, len(feat_data), lap_matrix, args.n_layers, prob_norm)
            for _iter in range(args.n_iters):
                for adjs, input_nodes, output_nodes in train_data:    
                    adjs = package_mxl(adjs, device)
                    optimizer.zero_grad()
                    t1 = time.time()
                    susage.train()
                    output = susage.forward(feat_data[input_nodes], adjs)
                
                    if args.sample_method == 'full':
                        output = output[output_nodes]
                    loss_train = F.cross_entropy(output, labels[output_nodes])
                
                    loss_train.backward()
                    torch.nn.utils.clip_grad_norm_(susage.parameters(), 0.2)
                    optimizer.step()
                    times += [time.time() - t1]
                    train_losses += [loss_train.detach().tolist()]
                    del loss_train
            susage.eval()
            adjs, input_nodes, output_nodes = valid_data
            adjs = package_mxl(adjs, device)
            output = susage.forward(feat_data[input_nodes], adjs)
            if args.sample_method == 'full':
                output = output[output_nodes]
            loss_valid = F.cross_entropy(output, labels[output_nodes]).detach().tolist()
            valid_f1 = f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')
            st = ('Epoch: %d (%.1fs) Train Loss: %.2f    Valid Loss: %.2f Valid F1: %.3f \n') % (epoch, np.sum(times), np.average(train_losses), loss_valid, valid_f1) 
            f.write(st + '\n')

            if valid_f1 > best_val + 1e-2:
                best_val = valid_f1
                torch.save(susage, '../models/' + args.model_tag + '.pt')
                cnt = 0
            else:
                cnt += 1
            if cnt == args.n_stops // args.batch_num:
                break
    print('iteration time: ', time.time() - start)
    f.write('iteration time: ' + str(time.time() - start) + '\n')   
    return (1-best_val)

    '''
            If using batch sampling for inference:
    '''
        #     for b in np.arange(len(test_nodes) // args.batch_size):
        #         batch_nodes = test_nodes[b * args.batch_size : (b+1) * args.batch_size]
        #         adjs, input_nodes, output_nodes = sampler(np.random.randint(2**32 - 1), batch_nodes,
        #                                     samp_num_list * 20, len(feat_data), lap_matrix, args.n_layers)
        #         adjs = package_mxl(adjs, device)
        #         output = best_model.forward(feat_data[input_nodes], adjs)[output_nodes]
        #         test_f1 = f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')
        #         test_f1s += [test_f1]
        
    '''
            If using full-batch inference:
    
    '''

start = time.time()
res = gp_minimize(optimization_function, [(0.0, 1.0)] * feature_set.shape[0], n_calls = args.opt_iter)
print ('total time:', time.time() - start)
f.write(str(res) + '\n')
with open(os.path.join(save_dir, 'opt_res.pkl'), 'wb') as handle:
    pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)
print ('OPTIMIZATION COMPLETE')

fig = plt.figure()
ax = plot_convergence(res)
fig.axes.append(ax)
plt.savefig(os.path.join(save_dir, 'convergence.png'))

samp_num_list = np.array([args.samp_num, args.samp_num, args.samp_num, args.samp_num, args.samp_num])
best_model = torch.load('../models/' + args.model_tag + '.pt')
best_model.eval()
test_f1s = []
batch_nodes = test_nodes
adjs, input_nodes, output_nodes = default_sampler(np.random.randint(2**32 - 1), batch_nodes, samp_num_list * 20, len(feat_data), lap_matrix, args.n_layers)
adjs = package_mxl(adjs, device)

output = best_model.forward(feat_data[input_nodes], adjs)[output_nodes]
test_f1s = [f1_score(output.argmax(dim=1).cpu(), labels[output_nodes].cpu(), average='micro')]
f.write('test f1 score' + '\n')
f.write(str(test_f1s) + '\n')
print ('test f1 score')
print (str(test_f1s))

f.close()