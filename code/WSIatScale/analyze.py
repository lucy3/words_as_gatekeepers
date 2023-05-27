'''
This script reads the replacements outputted by
write_mask_preds.py. 

It is called by clsuter_reps_per_token.py

This differs from the original in that the representative 
we have written are already filtered down to tokens in our
custom vocabulary that we care about. So there is no more 
filtering out of special tokens. 
'''

import argparse
from dataclasses import dataclass
import json
import os
import random
from typing import Tuple
from collections import defaultdict
from transformers import AutoTokenizer

import numpy as np
from tqdm import tqdm

from utils import get_model_stopwords

SEED = 111

MAX_REPS = 100

INVERTED_INDEX_DIR = 'inverted_index'

@dataclass
class Instance:
    reps: Tuple
    doc_id: int = None
    probs: np.array = None
    sent: np.array = None

class RepInstances:
    '''
    In the original WSI paper, they remove stopwords, single letter words / punctuation,
    and half words (assuming these are wordpieces that start with ##). They do this in a
    'special tokens' file and we reimplement those parts here. 
    '''
    def __init__(self, model_hf_path='globuslabs/ScholarBERT', \
                 tokenizer=None, stopwords=None, tokenizer_map=None):
        self.data = []
        self.lemmatized_vocab = tokenizer_map # this could be not lemmatized or None
        if not tokenizer: 
            self.tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_fast=True)
        else: 
            self.tokenizer = tokenizer
        if not stopwords: 
            self.stopwords = get_model_stopwords(self.tokenizer)
        else: 
            self.stopwords = stopwords
    
    def populate_just_reps(self, token_positions, reps):
        '''
        used by read_files() below
        '''
        for global_pos in token_positions:
            curr_reps = np.array(reps[global_pos])
            self.clean_and_populate_reps(curr_reps)

    def clean_and_populate_reps(self, reps):
        '''
        used by populate_just_reps()
        '''
        reps, _ = self.remove_specific_tokens(self.stopwords, reps, half_words=True, single_letters=True)
        if self.lemmatized_vocab:
            reps, _ = self.lemmatize_reps_and_probs(reps)
        self.data.append(Instance(reps=reps))

    def lemmatize_reps_and_probs(self, curr_reps, curr_probs=None):
        curr_reps = list(map(lambda x: int(self.lemmatized_vocab.get(str(x), x)), curr_reps))
        new_reps = []
        seen_lemmas = set()
        element_indices_to_delete = []
        for i, rep in enumerate(curr_reps):
            if rep in seen_lemmas:
                element_indices_to_delete.append(i)
            else:
                new_reps.append(rep)
            seen_lemmas.add(rep)
        if curr_probs is not None:
            curr_probs = np.delete(curr_probs, element_indices_to_delete)
            
        return new_reps, curr_probs

    def populate_specific_size(self, n_reps):
        '''
        This is used if we saved more reps than 
        we want to use. 
        '''
        if n_reps == MAX_REPS:
            return self

        for instance in self.data:
            instance.reps = instance.reps[:n_reps]
            if instance.probs is not None:
                instance.probs = instance.probs[:n_reps]
                instance.probs /= sum(instance.probs)

    def remove_empty_replacements(self):
        '''
        If there are replacements that are empty, remove them
        '''
        indices_to_remove = []
        for i, instance in enumerate(self.data):
            if len(instance.reps) == 0:
                indices_to_remove.append(i)

        for i in indices_to_remove[::-1]:
            assert len(self.data[i].reps) == 0
            del self.data[i]

    def remove_query_word(self, tokens_to_remove):
        '''
        @inputs: 
        - tokens_to_remove: token IDs, depending on whether case sensitive
        or lemmatize, should already include some case sensitive and lemmatized forms 
        '''
        if self.lemmatized_vocab: 
            # find additional word forms not in vocab
            for key, value in self.lemmatized_vocab.items():
                if value in tokens_to_remove:
                    tokens_to_remove.add(key)
                    
        tokens_to_remove = [int(r) for r in tokens_to_remove]

        for instance in self.data:
            if instance.probs is None:
                instance.reps = [r for r in instance.reps if r not in tokens_to_remove]
            else:
                # Not encountered yet
                instance.reps, instance.probs = zip(*[(r, p) for r, p in zip(instance.reps, instance.probs) if r not in tokens_to_remove])

    def remove_specific_tokens(self, tokens_to_remove, reps, half_words=True, single_letters=True, probs=None):
        if not probs:
            new_reps = []
            for r in reps: 
                if r in tokens_to_remove: continue
                str_tok = self.tokenizer.convert_ids_to_tokens(int(r))
                if str_tok.startswith('##') and half_words: continue 
                if len(str_tok) == 1 and single_letters: continue
                new_reps.append(r)
            reps = new_reps
        else:
            # Not encountered yet
            reps, probs = zip(*[(r, p) for r, p in zip(reps, probs) if r not in tokens_to_remove])
        return reps, probs

def tokenize(tokenizer, word):
    token = tokenizer.encode(word, add_special_tokens=False)
    if len(token) > 1:
        raise ValueError(f'Word {word} is more than a single wordpiece.')
    token = token[0]
    return token

def read_files(tokens,
               data_dir,
               sample_n_instances,
               model_hf_path,
               instance_attributes=['doc_id', 'reps', 'probs', 'tokens'],
               inverted_index_dir=INVERTED_INDEX_DIR,
               bar=tqdm,
               tokenizer_map=None,
               leave_out_ids=[]):
    '''
    This function reads in the output of write_mask_preds.py
    @inputs: 
    - tokens: token/s to get reps for. For case-sensitive this should be only
    one token but for case-unsensitive this may be several token IDs
    - data_dir: typically called "replacements"
    - sample_n_instances: in the original WSI paper this is 1000
    '''
    files_to_pos = read_inverted_index(inverted_index_dir, tokens, sample_n_instances, leave_out_ids) 

    n_matches = 0
    rep_instances = RepInstances(model_hf_path, tokenizer_map=tokenizer_map) 

    for file, token_positions in bar(files_to_pos.items()):
        # load the files of interest
        contents = file.split('_')
        this_dir = os.path.join(data_dir, '_'.join(contents[:2]))
        file_id = contents[-1]
        doc_ids = np.load(npy_file_path(this_dir, file_id, 'doc_ids'), mmap_mode='r')
        tokens = np.load(npy_file_path(this_dir, file_id, 'tokens'), mmap_mode='r') if 'tokens' in instance_attributes else None
        lengths = np.load(npy_file_path(this_dir, file_id, 'lengths'), mmap_mode='r') if 'lengths' in instance_attributes or 'tokens' in instance_attributes else None
        reps = np.load(npy_file_path(this_dir, file_id, 'reps'), mmap_mode='r') if 'reps' in instance_attributes else None
        probs = np.load(npy_file_path(this_dir, file_id, 'probs'), mmap_mode='r') if 'probs' in instance_attributes else None

        rep_instances.populate_just_reps(token_positions, reps) 

        n_matches += len(token_positions)

    msg = f"Found Total of {n_matches} Matches in {len(files_to_pos)} Files."
    return rep_instances, msg

def npy_file_path(data_dir, f, a):
    '''
    return the npy file path in a data directory
    '''
    return os.path.join(data_dir, f"{f}-{a}.npy")

def read_inverted_index(inverted_index, tokens, sample_n_instances, leave_out_ids):
    '''
    called by read_files()
    '''
    index = {}
    for token in tokens: 
        inverted_index_file = os.path.join(inverted_index, f"{token}.jsonl")
        if not os.path.exists(inverted_index_file):
            raise ValueError(f'token {token} is not in inverted index {inverted_index}')
        with open(inverted_index_file, 'r') as f:
            for line in f:
                index.update(json.loads(line))

    index = sample_instances(index, sample_n_instances, leave_out_ids)
    return index

def sample_instances(index, sample_n_instances, leave_out_ids):
    '''
    index: {unique_file_id: [position of token in file]} 
    sample_n_instances: number of instances to sample
    leave_out_ids: list of any unique_file_ids to leave out
    during sense induction/clustering phase 
    '''
    if sample_n_instances < 0:
        return index
    random.seed(SEED)

    ret = {}
    sample_n_instances = min(sample_n_instances, len(index))
    if sample_n_instances > 0:
        unique_file_ids = set(index.keys())
        if leave_out_ids: 
            old_length = len(unique_file_ids)
            unique_file_ids = unique_file_ids - set(leave_out_ids)
            new_length = len(unique_file_ids)
            assert old_length != new_length
        files = random.sample(list(unique_file_ids), sample_n_instances)
        ret = {file: [index[file][0]] for file in files}
    return ret

def prepare_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--replacements_dir", type=str, default="/home/matane/matan/dev/datasets/processed_for_WSI/CORD-19/replacements/done")
    parser.add_argument("--word", type=str, default='race')
    parser.add_argument("--inverted_index", type=str, default='/home/matane/matan/dev/datasets/processed_for_WSI/CORD-19/inverted_index.json')
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--sample_n_instances", type=int, default=1000)
    parser.add_argument("--n_bow_reps_to_report", type=int, default=10, help="How many different replacements to report")
    parser.add_argument("--n_sents_to_print", type=int, default=2, help="Num sents to print")
    parser.add_argument("--show_top_n_clusters", type=int, default=20)
    parser.add_argument("--show_top_n_words_per_cluster", type=int, default=100)

    parser.add_argument("--cluster_alg", type=str, default=None, choices=['kmeans', 'agglomerative_clustering', 'dbscan'])
    parser.add_argument("--n_clusters", type=int, help="n_clusters, for kmeans and agglomerative_clustering")
    parser.add_argument("--distance_threshold", type=float, help="for agglomerative_clustering")
    parser.add_argument("--affinity", type=str, help="for agglomerative_clustering")
    parser.add_argument("--linkage", type=str, help="for agglomerative_clustering", default='complete')
    parser.add_argument("--eps", type=float, help="for dbscan")
    parser.add_argument("--min_samples", type=float, help="for dbscan")
    args = parser.parse_args()

    return args

def assert_arguments(args):
    if args.cluster_alg == 'kmeans':
        assert args.n_clusters is not None, \
            "kmeans requires --n_clusters"
    elif args.cluster_alg == 'agglomerative_clustering':
        assert args.n_clusters is not None or args.distance_threshold is not None, \
            "agglomerative_clustering requires either --n_clusters or --distance_threshold"
    elif args.cluster_alg == 'dbscan':
        assert args.eps is not None and args.min_samples is not None, \
            "dbscan requires either --eps or --min_samples"
        assert args.n_clusters is None, \
            "dbscan doesn't need --n_clusters"