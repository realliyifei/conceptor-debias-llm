import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import sys
import pickle
import json
from IPython.display import clear_output 
import argparse
import warnings

import utils
import constants
from dataloader import Wordlist_Loader, Corpus_Loader
import conceptor

## Ref. https://github.com/jsedoc/ConceptorDebias/blob/master/WEAT/WEAT_(Final).ipynb

## Task: Generate corpus embeds for each corpus type
def generate_corpus_embeds(args, model, tokenizer):
    corpus_type_to_corpus_instance = corpus_type_to_corpus_instance = Corpus_Loader(args.corpus_folder).get_corpora_dict()
    ## Use Huggingface transformers to generate embeds
    for args.corpus_type in args.corpus_type_list:
        for args.layer_index in args.layer_index_list:
            corpus_embeds_folder = utils.get_path(args, 'corpus_embeds', True)
            corpus_embeds_path = f'{corpus_embeds_folder}/corpus_embeds.pickle'
            if os.path.exists(corpus_embeds_path):
                continue
            if not os.path.exists(corpus_embeds_folder):
                os.makedirs(corpus_embeds_folder)
            corpus_sents, corpus_embeds = [], []
            for sent_raw in tqdm(corpus_type_to_corpus_instance[args.corpus_type]): # in total 57,340 sentence (brown corpus for demo)
                # Skip sentences that are too short. (Inspired by https://github.com/McGill-NLP/bias-bench/blob/0c4dbd5ba676c9eca1fbef8c2279eac53e8eae4b/bias_bench/dataset/inlp.py)
                if len(sent_raw) < 4:
                    continue 
                sent = ' '.join(sent_raw)
                input_ids = torch.tensor(tokenizer.encode(sent)).unsqueeze(0).to(args.device)
                # Skip sentences that are too long.
                if len(input_ids[0]) > 512:
                    continue
                segment_ids = torch.zeros_like(input_ids, dtype=int).to(args.device)
                hidden_states = torch.stack(model(input_ids, token_type_ids=segment_ids)['hidden_states']) # [L+1] x B x T x H 
                sent_emb = hidden_states[args.layer_index].squeeze().detach().cpu().numpy() # T x H
                corpus_embeds.append(sent_emb)
                corpus_sents.append(sent)
            
            pickle.dump({'corpus_embeds': corpus_embeds, 'corpus_sents': corpus_sents}, 
                        open(corpus_embeds_path, "wb" ))


## Task: Generate pick_embeddings_result.pkl 
def combine_subspace_types_for_all(args):
    subspace_types_for_all = constants.bias_type_to_subspace_types_for_all[args.bias_type]
    X, words, sents = [], [], []
    for subspace_type in subspace_types_for_all:
        pick_embeddings_path = os.path.join(utils.get_path(args, 'negc'), subspace_type, "pick_embeddings_result.pkl")
        X_sub, words_sub, sents_sub = utils.load_pick_embeddings(pick_embeddings_path)
        X += X_sub
        words += words_sub
        sents += sents_sub
    return X, words, sents


def get_pick_embeddings(args, word_list, corpus_embeds, corpus_sents, pick_embeddings_folder):
    '''
    Input:
        - word_list: list of strings, the attribute word list 
        - corpus_embeds: list of tensor, the embeddings of sentences in corpus
        - corpus_sents: list of string, the strings of sentences in corpus 
        - pick_embeddings_folder: string, the folder to save the pick_embeddings_result.pkl
    '''
    X, words, sents = [], [], []
    if args.subspace_type != 'all': 
        for sent_embed, sent in zip(corpus_embeds, corpus_sents):
            # lower if it is case insensitive (note that roberta is case sensitive)
            if args.model_version in ['bert-base-uncased', 'bert-large-uncased']:
                sent = sent.lower()
            sent_words = sent.split(' ')
            if len(sent_words) <= 1:
                continue 
            for word in word_list:
                if 'bert' in args.model_version: 
                    word = word.lower()
                if word in sent_words:
                    index_list_of_list = utils.get_token_index_word_in_sent(args, word, sent, tokenizer) 
                    # the frequency of word in sentence is 1 or more 
                    word_freq = len(index_list_of_list) 
                    is_single_token = (len(index_list_of_list[0]) == 1)
                    # print(word, '||', sent, index_list_of_list)
                    for word_idx in range(word_freq):
                        if is_single_token:
                            word_embed = sent_embed[index_list_of_list[word_idx][0]]
                        else:
                            word_embed = sent_embed[index_list_of_list[word_idx][0]:index_list_of_list[word_idx][-1]+1].mean(0)  
                        X.append(word_embed)
                        words.append(word)
                        sents.append(sent) 
    else: # Combine all subspace types for the `all` subspace
        X, words, sents = combine_subspace_types_for_all(args)
    
    pick_embeddings_result = {'X': X, 'words': words, 'sents': sents}
    if not os.path.exists(pick_embeddings_folder):
        os.makedirs(pick_embeddings_folder)
    with open(f'{pick_embeddings_folder}/pick_embeddings_result.pkl','wb') as f:
        pickle.dump(pick_embeddings_result, f)
    print(f'Saved as: {pick_embeddings_folder}/pick_embeddings_result.pkl')
    return pick_embeddings_result


def generate_pick_embeddings_all(args, model, get_outlier_remain_words=None, word_df=None):
    subspace_type_to_wordlist, _, _, _ = Wordlist_Loader(args.wordlist_folder).get_wordlists_and_words()
    #Note that wordlist 'survey' is from the paper (https://arxiv.org/pdf/2110.08527.pdf) and aims for comparison
    #So we don't set percentage to it (i.e. it's 100% percentile in all folders) 
    for args.corpus_type in args.corpus_type_list:
        for args.layer_index in args.layer_index_list:
            corpus_embeds_folder = utils.get_path(args, 'corpus_embeds', True)
            for args.wordlist_percentile in args.wordlist_percentile_list:
                for args.subspace_type in constants.bias_type_to_subspace_types[args.bias_type]:
                    pick_embeddings_folder = os.path.join(utils.get_path(args, 'negc'), args.subspace_type)
                    print("Working:", pick_embeddings_folder)
                    # Skip if the pick_embeddings_result.pkl already exists, or if the subspace_type is 'or' / 'and' (which is computed in the next step)
                    if os.path.exists(os.path.join(pick_embeddings_folder, "pick_embeddings_result.pkl")) or args.subspace_type == 'or' or args.subspace_type == 'and':
                        continue
                    word_list = subspace_type_to_wordlist[args.subspace_type]
                    if args.wordlist_percentile < 1 and args.subspace_type != 'survey': 
                        warnings.warn("The outlier filter word_df pipeline has not been moved from colab yet.")
                        continue
                    corpus_embeds, corpus_sents = [], []
                    if args.subspace_type not in ['all', 'or', 'and']:
                        corpus_embeds, corpus_sents = utils.load_embeds_sents(os.path.join(corpus_embeds_folder, "corpus_embeds.pickle"))
                    _ = get_pick_embeddings(args, word_list, corpus_embeds, corpus_sents, pick_embeddings_folder)
                        

## Task - generate_negc_matrix: generate negc matrix 
def get_negc_matrix(args, negc_path):
    '''
    negc matrix stands for negation conceptor matrix
    Input
        X: list of word embeddings from sentent whose sentence contain one of the word in word_list
        words: list of words from word_list where each word corrensponding to the sample in the same row of X 
    Return:
        negc matrix (and save to path)
    '''
    pick_embeddings_path = os.path.join(utils.get_path(args, 'negc'), args.subspace_type, "pick_embeddings_result.pkl")
    X, words, _ = utils.load_pick_embeddings(pick_embeddings_path)
    negC, _, _ = conceptor.post_process_cn_matrix(np.array(X).T)
    word_to_embeds = {}
    for word_embed, word in zip(X, words):
        if word not in word_to_embeds:
            word_to_embeds[word] = []
        word_to_embeds[word].append(word_embed)
    # if one key word appear more than once, then get the average tokens' value 
    embeds = np.array([np.average(word_to_embeds[w], 0) for w in list(word_to_embeds.keys())])

    with open(negc_path,'wb') as f:
        pickle.dump({'negC': negC, 'embeds': embeds, 'words': words}, f)
    print(f'Saved as: {negc_path}')
    return negC


def get_negc_matrix_for_or(args, negc_path): 
    '''Compute the OR subspace'''
    negc_folder = utils.get_path(args, 'negc')
    subspace_types_for_and = constants.bias_type_to_subspace_types_for_conceptor_operatoin[args.bias_type]
    negC_list = [utils.load_conceptor(os.path.join(negc_folder, subspace_type, 'negc.pkl')) for subspace_type in subspace_types_for_and]
    # Initialize negC as the first item in negC_list, then AND the rest iteratively as the negated OR
    # e.g. AND(negC1, AND(negC2, negC3)) = AND(negC1, NOT(OR(C2, C2))) = NOT(OR(C1, OR(C2, C3)))
    negC = negC_list[0]
    for i in range(1, len(negC_list)):
        negC = conceptor.AND(negC, negC_list[i])
    if not os.path.exists(os.path.join(negc_folder, "or")):
        os.mkdir(os.path.join(negc_folder, "or"))
    with open(negc_path,'wb') as f:
        pickle.dump({'negC': negC, 'embeds': None, 'words': None}, f)
    print(f'Saved as: {negc_path}')
    return negC


def get_negc_matrix_for_and(args, negc_path):
    '''Compute the AND subspace'''
    negc_folder = utils.get_path(args, 'negc')
    subspace_types_for_and = constants.bias_type_to_subspace_types_for_conceptor_operatoin[args.bias_type]
    negC_list = [utils.load_conceptor(os.path.join(negc_folder, subspace_type, 'negc.pkl')) for subspace_type in subspace_types_for_and]
    # Initialize negC as the first item in negC_list, then AND the negated of each one in the rest iteratively as the negated AND
    # e.g. NOT(AND(NOT(negC1), AND(NOT(negC2), NOT(negC3))) = NOT(AND(C1, AND(C2, C3)))
    negC = negC_list[0]
    for i in range(1, len(negC_list)):
        negC = conceptor.NOT(conceptor.AND(conceptor.NOT(negC), conceptor.NOT(negC_list[i])))
    if not os.path.exists(os.path.join(negc_folder, "and")):
        os.mkdir(os.path.join(negc_folder, "and"))
    with open(negc_path,'wb') as f:
        pickle.dump({'negC': negC, 'embeds': None, 'words': None}, f)
    print(f'Saved as: {negc_path}')
    return negC


def generate_negc_matrix_all(args):
    # Only for gender bias_type now
    for args.corpus_type in args.corpus_type_list:
        for args.layer_index in args.layer_index_list:
            for args.wordlist_percentile in args.wordlist_percentile_list:
                for args.subspace_type in constants.bias_type_to_subspace_types['gender']:
                    negc_path = os.path.join(utils.get_path(args, 'negc'), args.subspace_type, "negc.pkl")
                    print("Working:", negc_path)
                    if os.path.exists(negc_path):
                        continue
                    if args.subspace_type != 'or' and args.subspace_type != 'and':
                        _ = get_negc_matrix(args, negc_path)
                    elif args.subspace_type == 'or':
                        _ = get_negc_matrix_for_or(args, negc_path)
                    elif args.subspace_type == 'and':
                        _ = get_negc_matrix_for_and(args, negc_path)


# main method
if __name__ == '__main__':
    args = utils.get_args() 
    print(args)
    model, tokenizer = utils.load_model_tokenizer(args)
    assert args.task is not None
    if args.task == 'generate_corpus_embeds':
        generate_corpus_embeds(args, model, tokenizer)
    elif args.task == 'generate_wordlist':
        pass 
    elif args.task == 'generate_pick_embeddings':
        generate_pick_embeddings_all(args, model)
    elif args.task == 'generate_negc':
        generate_negc_matrix_all(args)
    elif args.task == 'all':
        generate_corpus_embeds(args, model, tokenizer)
        generate_pick_embeddings_all(args, model)
        generate_negc_matrix_all(args)    