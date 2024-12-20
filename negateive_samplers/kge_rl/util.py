import numpy as np
from torch.autograd import Variable
import torch
from scipy import stats
import constants
cache = dict()

def chunk(arr,chunk_size):
    if len(arr)==0:
        yield []

    for i in range(0,len(arr),chunk_size):
        yield arr[i:i+chunk_size]


def sample(data,num_samples,replace=False):

    if len(data) <= num_samples:
        return data
    np.random.shuffle(data)
    return data[:num_samples]


def ranks(scores, ascending = False):
    sign = 1 if ascending else -1
    scores = scores * sign
    ranks = [stats.rankdata(scores[i])[0] for i in range(scores.shape[0])]
    return ranks

def get_triples(batch,negs,is_target=True, volatile=False,is_pad=False):
    sources,rels,targets = ([],[],[])
    if negs is None:
        for ex in batch:
            sources.append([ex.s])
            targets.append([ex.t])
            rels.append(ex.r)
    else:
        for count,ex in enumerate(batch):
            s = [] if is_target else [n for n in negs[count]]
            t = [n for n in negs[count]] if is_target else []
            s.insert(0,ex.s)
            t.insert(0,ex.t)
            if is_pad:
                if is_target:
                    t = pad_arr(t,t[-1])
                else:
                    s = pad_arr(s,s[-1])
            sources.append(s)
            targets.append(t)
            rels.append(ex.r)

    return to_var(sources,volatile=volatile, requires_grad=False), to_var(targets,volatile=volatile, requires_grad=False),to_var(rels,volatile=volatile, requires_grad=False)

def to_var(x,volatile=False, requires_grad=False):
    if 'cuda' not in cache:
        cache['cuda'] = torch.cuda.is_available()
    cuda = cache['cuda']

    # ZAFAR EDIT: 16-Oct-2024 ******************************
    # var = Variable(torch.from_numpy(np.asarray(x)),volatile=volatile, requires_grad=requires_grad)
    # above parameter volatile is depreciated in newer version of Pytorch
    with torch.no_grad():
        var = torch.tensor(np.asarray(x), requires_grad=requires_grad)
    # ******************************************************
    
    if cuda:
        return var.cuda()
    return var

def pad_arr(arr,val):
    if len(arr)>=constants.fb13_ents:
        return arr
    else:
        zeros = [val]*(constants.fb13_ents-len(arr))
        arr.extend(zeros)
        return arr


def get_triples_lists(batch,negs,is_target=True, volatile=False,is_pad=False):
    sources,rels,targets = ([],[],[])
    if negs is None:
        for ex in batch:
            sources.append([ex.s])
            targets.append([ex.t])
            rels.append(ex.r)
    else:
        for count,ex in enumerate(batch):
            s = [] if is_target else [n for n in negs[count]]
            t = [n for n in negs[count]] if is_target else []
            s.insert(0,ex.s)
            t.insert(0,ex.t)
            if is_pad:
                if is_target:
                    t = pad_arr(t,t[-1])
                else:
                    s = pad_arr(s,s[-1])
            sources.append(s)
            targets.append(t)
            rels.append(ex.r)

    #return to_var(sources,volatile=volatile, requires_grad=False), to_var(targets,volatile=volatile, requires_grad=False),to_var(rels,volatile=volatile, requires_grad=False)
    #Zafar: Edit, I am returning simple list because I need to map them onto exsting tensors (WrapperNegativeSampler.corrupt_batch.negative_batch)
    # Beware, the first corresponding elements in the lists makes a true triple
    return sources,rels,targets