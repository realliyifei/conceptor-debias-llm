import logging as log
import math
import itertools as it
import numpy as np
import scipy.special
import scipy.stats
import torch
import json
import pandas as pd
import os

import utils
import constants

# X and Y are two sets of target words of equal size.
# A and B are two sets of attribute words.

def cossim(x, y):
    return np.dot(x, y) / math.sqrt(np.dot(x, x) * np.dot(y, y))

def construct_cossim_lookup(XY, AB):
    """
    XY: mapping from target string to target vector (either in X or Y)
    AB: mapping from attribute string to attribute vectore (either in A or B)
    Returns an array of size (len(XY), len(AB)) containing cosine similarities
    between items in XY and items in AB.
    """
    cossims = np.zeros((len(XY), len(AB)))
    for xy in XY:
        for ab in AB:
            cossims[xy, ab] = cossim(XY[xy], AB[ab])
    return cossims

def s_wAB(A, B, cossims):
    """
    Return vector of s(w, A, B) across w, where
        s(w, A, B) = mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    """
    return cossims[:, A].mean(axis=1) - cossims[:, B].mean(axis=1)

def s_XAB(X, s_wAB_memo):
    r"""
    Given indices of target concept X and precomputed s_wAB values,
    return slightly more computationally efficient version of WEAT
    statistic for p-value computation.
    Caliskan defines the WEAT statistic s(X, Y, A, B) as
        sum_{x in X} s(x, A, B) - sum_{y in Y} s(y, A, B)
    where s(w, A, B) is defined as
        mean_{a in A} cos(w, a) - mean_{b in B} cos(w, b).
    The p-value is computed using a permutation test on (X, Y) over all
    partitions (X', Y') of X union Y with |X'| = |Y'|.
    However, for all partitions (X', Y') of X union Y,
        s(X', Y', A, B)
      = sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
      = C,
    a constant.  Thus
        sum_{x in X'} s(x, A, B) + sum_{y in Y'} s(y, A, B)
      = sum_{x in X'} s(x, A, B) + (C - sum_{x in X'} s(x, A, B))
      = C + 2 sum_{x in X'} s(x, A, B).
    By monotonicity,
        s(X', Y', A, B) > s(X, Y, A, B)
    if and only if
        [s(X', Y', A, B) - C] / 2 > [s(X, Y, A, B) - C] / 2,
    that is,
        sum_{x in X'} s(x, A, B) > sum_{x in X} s(x, A, B).
    Thus we only need use the first component of s(X, Y, A, B) as our
    test statistic.
    """
    return s_wAB_memo[X].sum()

def s_XYAB(X, Y, s_wAB_memo):
    r"""
    Given indices of target concept X and precomputed s_wAB values,
    the WEAT test statistic for p-value computation.
    """
    return s_XAB(X, s_wAB_memo) - s_XAB(Y, s_wAB_memo)

def p_val_permutation_test(X, Y, A, B, n_samples, cossims, parametric=True):
    ''' Compute the p-val for the permutation test, which is defined as
        the probability that a random even partition X_i, Y_i of X u Y
        satisfies P[s(X_i, Y_i, A, B) > s(X, Y, A, B)]
    '''
    X = np.array(list(X), dtype=int)
    Y = np.array(list(Y), dtype=int)
    A = np.array(list(A), dtype=int)
    B = np.array(list(B), dtype=int)

    assert len(X) == len(Y)
    size = len(X)
    s_wAB_memo = s_wAB(A, B, cossims=cossims)
    XY = np.concatenate((X, Y))

    if parametric:
        log.info('Using parametric test')
        s = s_XYAB(X, Y, s_wAB_memo)

        log.info('Drawing {} samples'.format(n_samples))
        samples = []
        for _ in range(n_samples):
            np.random.shuffle(XY)
            Xi = XY[:size]
            Yi = XY[size:]
            assert len(Xi) == len(Yi)
            si = s_XYAB(Xi, Yi, s_wAB_memo)
            samples.append(si)

        # Compute sample standard deviation and compute p-value by
        # assuming normality of null distribution
        log.info('Inferring p-value based on normal distribution')
        (shapiro_test_stat, shapiro_p_val) = scipy.stats.shapiro(samples)
        log.info('Shapiro-Wilk normality test statistic: {:.2g}, p-value: {:.2g}'.format(
            shapiro_test_stat, shapiro_p_val))
        sample_mean = np.mean(samples)
        sample_std = np.std(samples, ddof=1)
        log.info('Sample mean: {:.2g}, sample standard deviation: {:.2g}'.format(
            sample_mean, sample_std))
        p_val = scipy.stats.norm.sf(s, loc=sample_mean, scale=sample_std)
        return p_val

    else:
        log.info('Using non-parametric test')
        s = s_XAB(X, s_wAB_memo)
        total_true = 0
        total_equal = 0
        total = 0

        num_partitions = int(scipy.special.binom(2 * len(X), len(X)))
        if num_partitions > n_samples:
            # We only have as much precision as the number of samples drawn;
            # bias the p-value (hallucinate a positive observation) to reflect that.
            total_true += 1
            total += 1
            log.info('Drawing {} samples (and biasing by 1)'.format(n_samples - total))
            for _ in range(n_samples - 1):
                np.random.shuffle(XY)
                Xi = XY[:size]
                assert 2 * len(Xi) == len(XY)
                si = s_XAB(Xi, s_wAB_memo)
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1

        else:
            log.info('Using exact test ({} partitions)'.format(num_partitions))
            for Xi in it.combinations(XY, len(X)):
                Xi = np.array(Xi, dtype=np.int)
                assert 2 * len(Xi) == len(XY)
                si = s_XAB(Xi, s_wAB_memo)
                if si > s:
                    total_true += 1
                elif si == s:  # use conservative test
                    total_true += 1
                    total_equal += 1
                total += 1

        if total_equal:
            log.warning('Equalities contributed {}/{} to p-value'.format(total_equal, total))

        return total_true / total

def mean_s_wAB(X, A, B, cossims):
    return np.mean(s_wAB(A, B, cossims[X]))

def stdev_s_wAB(X, A, B, cossims):
    return np.std(s_wAB(A, B, cossims[X]), ddof=1)

def effect_size(X, Y, A, B, cossims):
    """
    Compute the effect size, which is defined as
        [mean_{x in X} s(x, A, B) - mean_{y in Y} s(y, A, B)] /
            [ stddev_{w in X u Y} s(w, A, B) ]
    args:
        - X, Y, A, B : sets of target (X, Y) and attribute (A, B) indices
    """
    X, Y, A, B = list(X), list(Y), list(A), list(B)

    numerator = mean_s_wAB(X, A, B, cossims=cossims) - mean_s_wAB(Y, A, B, cossims=cossims)
    denominator = stdev_s_wAB(X + Y, A, B, cossims=cossims)
    return numerator / denominator

def convert_keys_to_ints(X, Y):
    return (
        dict((i, v) for (i, (k, v)) in enumerate(X.items())),
        dict((i + len(X), v) for (i, (k, v)) in enumerate(Y.items())),
    )

def get_seat(X, Y, A, B, show=True):
    (X, Y) = convert_keys_to_ints(X, Y)
    (A, B) = convert_keys_to_ints(A, B)

    XY = X.copy()
    XY.update(Y)
    AB = A.copy()
    AB.update(B)

    cossims = construct_cossim_lookup(XY, AB)
    pval = p_val_permutation_test(X, Y, A, B, cossims=cossims, n_samples=100000)
    esize = effect_size(X, Y, A, B, cossims=cossims)
    
    if show: 
        print(f"pval: {pval:.4f}; esize: {esize:.4f}") 
    else:   
        return pval, esize

def encode(args, texts, model, tokenizer):
    encs = {}
    for text in texts:
        input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0).to(args.device)
        segment_ids = torch.zeros_like(input_ids, dtype=int).to(args.device)
        hidden_states = torch.stack(model(input_ids, token_type_ids=segment_ids)['hidden_states']) # [L+1] x B x T x H 
        text_emb = hidden_states[args.layer_index].mean(dim=1) # T x H
        
        encs[text] = text_emb.detach().view(-1).to('cpu').numpy()
        encs[text] /= np.linalg.norm(encs[text])

    return encs


def get_seat_result(args, save_csv_path=None, negc_hardcode=None, bias_type_hardcode=None):

    negc_folder = utils.get_path(args, 'negc')
    # for testing gender conceptor p.p. for race, or for intersetional
    bias_type = args.bias_type if bias_type_hardcode is None else bias_type_hardcode
    jsonfile2name = constants.bias_type_jsonfile2name[bias_type]
    subspaces = ['original'] + constants.bias_type_to_subspace_types[bias_type]

    model, tokenizer = utils.load_model_tokenizer(args)
    pvals, esizes = [], []
    filenames = sorted(jsonfile2name.keys())

    for subspace in subspaces:
        pvals_line, esizes_line = [], []
        for filename in filenames:
            with open(f'{args.data_folder}/weat_seat/{filename}.jsonl', 'r') as f:
                encs = json.load(f)

            X = encode(args, encs["targ1"]["examples"], model, tokenizer)
            Y = encode(args, encs["targ2"]["examples"], model, tokenizer)
            A = encode(args, encs["attr1"]["examples"], model, tokenizer)
            B = encode(args, encs["attr2"]["examples"], model, tokenizer)            

            X_tmp, Y_tmp, A_tmp, B_tmp = X, Y, A, B

            if subspace != 'original':
                negc = utils.load_conceptor(f'{negc_folder}/{subspace}/negc.pkl') if negc_hardcode is None else negc_hardcode
                X_tmp = {k: (negc @ v.T).T for k, v in X_tmp.items()}
                Y_tmp = {k: (negc @ v.T).T for k, v in Y_tmp.items()}
                A_tmp = {k: (negc @ v.T).T for k, v in A_tmp.items()}
                B_tmp = {k: (negc @ v.T).T for k, v in B_tmp.items()}
            
            pval, esize = get_seat(X_tmp, Y_tmp, A_tmp, B_tmp, show=False)
            pvals_line.append(pval)
            esizes_line.append(esize)

        pvals.append(pvals_line)
        esizes.append(esizes_line)
    
    pvals, esizes = np.round(np.array(pvals), 3), np.round(np.array(esizes), 3)
    pvals_symbol = np.where(pvals < 0.01, "*", "")
    esizes_abs_avg = np.apply_along_axis(utils.get_abs_avg, 1, esizes)
    esizes_with_symbol = np.core.defchararray.add(esizes.astype(str), pvals_symbol.astype(str))
    esizes_with_symbol_result = np.append(esizes_with_symbol, esizes_abs_avg.reshape(-1,1), axis=1)

    seat_result_df = pd.DataFrame(
            esizes_with_symbol_result, 
            columns=[jsonfile2name[fname] for fname in filenames] + ['Avg. Abs.'],
            index=subspaces
    ).round(3)
    if save_csv_path is not None:
        seat_result_df.to_csv(save_csv_path)
        print('Saved to:', save_csv_path)
    
    return seat_result_df
    
def get_seat_result_all(args):
    seat_fname = "result-seat.csv"
    print(f"seat_fname: {seat_fname}")
    for args.corpus_type in args.corpus_type_list:
        for args.layer_index in args.layer_index_list:
            for args.wordlist_percentile in args.wordlist_percentile_list:
                save_csv_path = os.path.join(utils.get_path(args, 'negc'), seat_fname) 
                if os.path.exists(save_csv_path) and not args.overwrite:
                    continue


# main method
if __name__ == '__main__':
    args = utils.get_args() 
    print(args)
    get_seat_result_all(args)