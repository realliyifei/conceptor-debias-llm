import os
import numpy as np
import torch
import pickle
import argparse
import transformers
from model import models
import constants

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-device', default='cuda:0', choices=['cuda:0', 'cpu'])
    parser.add_argument('-bias_type', default='gender', choices=['gender', 'race'])
    parser.add_argument('-special_token', default="", choices=['', '-tsne', '-umap']) 

    parser.add_argument('-output_folder', default='output')
    parser.add_argument('-data_folder', default="data")
    parser.add_argument('-corpus_folder', default=os.path.join("data", "corpora"))
    parser.add_argument('-wordlist_folder', default=os.path.join("data", "wordlist"))

    parser.add_argument('-model_version', default=None, type=str) 
    parser.add_argument('-corpus_type', default='brown', type=str, choices=['brown', 'sst', 'reddit', 'wikipedia-2.5', 'wikipedia-10'])  
    parser.add_argument('-layer_index', default=0, type=int) # Generally, it is the last layer of the model; -1 means none (used for gpt3)
    parser.add_argument('-wordlist_percentile', default=1, type=float, choices=[1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]) 
    parser.add_argument('-corpus_type_list', nargs='+', type=str) 
    parser.add_argument('-layer_index_list', nargs='+', type=int) 
    parser.add_argument('-wordlist_percentile_list', nargs='+', type=float)
    parser.add_argument('-embed_type', type=str, default='avg') 
    parser.add_argument('-subspace_type', default='pronouns', type=str, choices=['pronouns', 'propernouns', 'extended', 'all', 'or', 'and', 'name', 'geo']) 

    # For seat
    parser.add_argument('-overwrite', action='store_true', default=False, help='Overwrite existing files')
    
    # For conceptor negation
    parser.add_argument('-task', default=None)
    parser.add_argument('-negc', default=None, type=str, help='Path to the negC file for post-processing')
    
    # For continued training
    parser.add_argument('-continue_train_folder', default='continue-train')
    parser.add_argument('-continue_train_type', default=None,
                        choices=['sst-percentile1-and', 'brown-percentile0.4-and', 'sst-percentile0.9-extended'])
    
    # sys.argv = ['-f'] # For jupyter notebook
    args = parser.parse_args()
    return args


## Get paths
def get_path(args, path_type, print_path=False):
    '''Get folder path of type: 'negc', 'corpus_embeds', 'train_result' '''
    path = ""
    if path_type == 'negc': 
        path = os.path.join(args.output_folder, args.corpus_type, f"{args.model_version}",
                            f"{args.bias_type}{args.special_token}", f"layer_{args.layer_index}", 
                            f"wordlist_percentile{args.wordlist_percentile}", args.embed_type)
    elif path_type == 'corpus_embeds': 
        path = os.path.join(args.output_folder, args.corpus_type, args.model_version, 
                            "corpus-embeds", f"layer_{args.layer_index}")
    elif path_type == 'train_result':
        path = os.path.join(args.continue_train_folder, args.model_version, args.continue_train_type)
    if print_path:
        print(f"Type: {path_type}; Path: {path}")
    return path


#LOAD
def sent_to_embed(args, sent, tokenizer, model):
    sent_embed = None
    input_ids = torch.tensor(tokenizer.encode(sent)).unsqueeze(0).to(args.device)
    segment_ids = torch.zeros_like(input_ids, dtype=int).to(args.device)
    hidden_states = torch.stack(model(input_ids, token_type_ids=segment_ids)['hidden_states']) # [L+1] x B x T x H 
    sent_embed = hidden_states[args.layer_index].squeeze().detach().cpu().numpy() # T x H
    return sent_embed 


def sentence_to_tokens(args, sent, tokenizer):
    input_ids = torch.tensor(tokenizer.encode(sent)).unsqueeze(0).to(args.device)
    return input_ids


def get_token_index_word_in_sent(args, word, sent, tokenizer):
    '''
    Input: 
        - word: string, one attribute word (or multi-word, which is rare)
        - sent: string, one sentence
    Return:
        - index_list_of_list: list of list, 
            the index of the word tokens witin the sentence (0-indexed)
            The number of list is the frequency of the word in the sentence
            Inside each list, 
                there's one item if the word is one token, i.e. [begin_index]
                there're two items if the word is multiple-token, i.e. [begin_index, end_index]  
    '''
    word_tokens, sent_tokens = [], []
    index_list_of_list = []
    if args.model_version in ['bert-base-uncased', 'bert-large-uncased']:
        # Should remove [CLS], [SEP] tokens in BERT
        word_tokens = sentence_to_tokens(args, word, tokenizer)[0][1:-1].tolist() 
        sent_tokens = sentence_to_tokens(args, sent, tokenizer)[0].tolist()
        for i, token in enumerate(sent_tokens):
            if token == word_tokens[0]:
                index_list_of_list.append([i])
    elif any(m in args.model_version for m in ['gpt']):
        is_word_at_start = (sent.split(' ')[0] == word) 
        word = word if is_word_at_start else ' ' + word
        word_tokens = sentence_to_tokens(args, word, tokenizer)[0].tolist()
        sent_tokens = sentence_to_tokens(args, sent, tokenizer)[0].tolist()
        for i, token in enumerate(sent_tokens):
            if token == word_tokens[0]:
                index_list_of_list.append([i])
    else:
        assert False, f"This model {args.model_version} is not assigned in index function yet."
    if len(word_tokens) > 1:
        for i in range(len(index_list_of_list)):
            index_list_of_list[i].append(index_list_of_list[i][0] + len(word_tokens)-1)
    return index_list_of_list


def load_embeds_sents(corpus_embeds_path):
    corpus_embeds = pickle.load(open(corpus_embeds_path, "rb"))['corpus_embeds'] # num_sent x [T x H]
    corpus_sents = pickle.load(open(corpus_embeds_path, "rb"))['corpus_sents'] # num_sent x
    return corpus_embeds, corpus_sents


def load_pick_embeddings(pick_embeddings_result_path):
    with open(pick_embeddings_result_path, 'rb') as f:
        pick_embeddings_result = pickle.load(f)
    return pick_embeddings_result.values() # return X, words, sents


def load_conceptor(path, return_all=False):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    if not return_all:
        return data['negC'] # return negC
    else:
        return data.values() # return negC, embeds, words


def load_model_tokenizer(args, output_hidden_states=True, config=None, cache_dir=constants.hf_cache):
    #src: https://github.com/W4ngatang/sent-bias/blob/e3559fb669/sentbias/encoders/bert.py
    '''
    Return: model, tokenizer, with the negc post-process hook if args.negc is not None.
    Note that the SequenceClassification models are only used for GLUE evaluation, and need config input.
    '''
    model, tokenizer = None, None
    print(f"Loading model: {args.model_version}...")
    
    negc = None
    if args.negc is not None:
        print(f"Post-process by the negc: \'{args.negc}\'.")
        negc = torch.tensor(load_conceptor(args.negc), dtype=torch.float32)
        
    if args.model_version == 'bert-base-uncased':
        tokenizer = transformers.BertTokenizer.from_pretrained(
            'bert-base-uncased', cache_dir=cache_dir
        )
        if args.negc is None:
            model = transformers.BertModel.from_pretrained(
                'bert-base-uncased', output_hidden_states=output_hidden_states, cache_dir=cache_dir
            )
        else:
            model = models.CABertModel(
                'bert-base-uncased', negc, output_hidden_states
            )
    
    elif args.model_version == 'bert-large-uncased':
        tokenizer = transformers.BertTokenizer.from_pretrained(
            'bert-large-uncased', cache_dir=cache_dir
        )
        if args.negc is None:
            model = transformers.BertModel.from_pretrained(
                'bert-large-uncased', output_hidden_states=output_hidden_states, cache_dir=cache_dir
            )
        else:
            model = models.CABertModel(
                'bert-large-uncased', negc, output_hidden_states
            )

    elif args.model_version == 'gpt2':
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            "gpt2", cache_dir=cache_dir
        )
        if args.negc is None:
            model = transformers.GPT2Model.from_pretrained(
                "gpt2", output_hidden_states=output_hidden_states, cache_dir=cache_dir
            )
        else:
            model = models.CAGPT2Model(
                'gpt2', negc, output_hidden_states
            )
    
    elif args.model_version == 'gpt2-large':
        tokenizer = transformers.GPT2Tokenizer.from_pretrained(
            "gpt2-large", cache_dir=cache_dir
        )
        if args.negc is None:
            model = transformers.GPT2Model.from_pretrained(
                "gpt2-large", output_hidden_states=output_hidden_states, cache_dir=cache_dir
            )
        else:
            model = models.CAGPT2Model(
                'gpt2-large', negc, output_hidden_states
            )

    elif args.model_version == 'gpt-j': 
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "EleutherAI/gpt-j-6B", cache_dir=cache_dir
        )
        if args.negc is None:
            model = transformers.GPTJModel.from_pretrained(
                "EleutherAI/gpt-j-6B", output_hidden_states=output_hidden_states, cache_dir=cache_dir
            )
        else:
            model = models.CAGPTJModel(
                'EleutherAI/gpt-j-6B', negc, output_hidden_states
            )  
    
    return model, tokenizer


#COMPUTE
def get_abs_avg(arr):
    return np.round(np.abs(arr).mean(), 3)