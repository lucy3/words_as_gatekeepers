# pylint: disable=no-name-in-module
# pylint: disable=import-error
import argparse
from collections import Counter
from copy import deepcopy
from functools import partial
import json
from multiprocessing import Pool, cpu_count
import os
import numpy as np
from tqdm import tqdm

from analyze import read_files
from utils import tokenizer_params 

from community_detection import CommunityFinder

from transformers import AutoTokenizer

SAMPLE_N_INSTANCES = 1000
MOST_COMMON_CLUSTER_REPS = 100

def main(args):
    model_hf_path = tokenizer_params[args.dataset]
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_fast=True)
    
    # load target vocab map and tokenizer vocab map
    if args.lemmatize: 
        with open(os.path.join(args.index_dir, "SKIP_lemma.json"), 'r') as infile: 
            str_to_wordID_map = json.load(infile) # {token string : token IDs}
        with open(os.path.join(args.index_dir, "SKIP_lemma_all.json"), 'r') as infile: 
            tokenizer_map = json.load(infile) # {token string : token IDs}
    elif args.case_sensitive: 
        with open(os.path.join(args.index_dir, "SKIP_identity.json"), 'r') as infile: 
            str_to_wordID_map = json.load(infile) # {token string : token IDs}
        tokenizer_map = None
    else: 
        with open(os.path.join(args.index_dir, "SKIP_lower_to_upper.json"), 'r') as infile: 
            str_to_wordID_map = json.load(infile) # {token string : token IDs}
        with open(os.path.join(args.index_dir, "SKIP_lower_to_upper_all.json"), 'r') as infile: 
            tokenizer_map = json.load(infile) # {token string : token IDs}

    tokens_to_index = set(str_to_wordID_map.keys())
    
    if args.wiki_eval: 
        with open(args.wiki_json, 'r') as infile: 
            label_to_words = json.load(infile)
        new_tokens_to_index = set()
        for group in label_to_words: 
            this_group = set(label_to_words[group]) & tokens_to_index
            new_tokens_to_index.update(this_group)
        tokens_to_index = new_tokens_to_index
    
    # don't cluster tokens that have already been clustered 
    already_done = set([f.split('_')[0] for f in os.listdir(out_dir) if not f.startswith('SKIP')])
    tokens_to_index -= already_done
    
    print(f"{len(tokens_to_index)} tokens to index")
    
    # get list of unique file IDs to not use during training
    leave_out_ids = []
    if args.leave_out_list: 
        with open(os.path.join(args.index_dir, 'SKIP_leave_out_ids.txt'), 'r') as infile: 
            for line in infile: 
                leave_out_ids.append(line.strip())

    # now cluster the reps for each token
    partial_write_communities_to_disk = partial(write_communities_to_disk, tokenizer=tokenizer, 
                                                model_hf_path=model_hf_path, 
                                                str_to_wordID_map=str_to_wordID_map, 
                                               tokenizer_map=tokenizer_map, out_dir=out_dir,
                                               leave_out_ids=leave_out_ids)

    with Pool(cpu_count() // 2) as p:
        list(tqdm(p.imap(partial_write_communities_to_disk, tokens_to_index), total=len(tokens_to_index)))

def write_communities_to_disk(token, tokenizer, model_hf_path, str_to_wordID_map, 
                              tokenizer_map, out_dir, leave_out_ids):
    '''
    @inputs: 
    - token: str representing token
    '''
    tokens = str_to_wordID_map[token]
    rep_instances, _ = read_files(tokens, args.data_dir,
        SAMPLE_N_INSTANCES,
        model_hf_path,
        instance_attributes=['doc_id', 'reps'],
        inverted_index_dir=args.index_dir,
        bar=lambda x: x,
        tokenizer_map=tokenizer_map,
        leave_out_ids=leave_out_ids,
    )
    n_reps = 5
    clustering_data_by_n_reps = {} # output of clustering 
    
    curr_rep_instances = deepcopy(rep_instances)
    curr_rep_instances.remove_query_word(tokens) 
    #curr_rep_instances.populate_specific_size(n_reps) # we only saved five, this is for if we had saved more
    curr_rep_instances.remove_empty_replacements()

    if len(curr_rep_instances.data) == 0: 
        return 
    clustering_data_by_n_reps[n_reps] = community_detection_clustering(curr_rep_instances, token)
    json.dump(clustering_data_by_n_reps, open(os.path.join(out_dir, f"{token}_clustering.json"), 'w'))

def community_detection_clustering(rep_instances, token, query_n_reps=10):
    '''
    @inputs: 
    - rep_instances: rep instances read from output of write_mask_preds.py
    - query_n_reps: in case we want to set a cutoff for the number of reps to use. We are
    starting with n_reps = 5 so any query_n_reps greater than that is fine. 
    '''
    community_finder = CommunityFinder(rep_instances, query_n_reps)
    # find the best partition
    communities, res_info = community_finder.find(resolution=args.resolution) # {community: [replacements] }
    if args.resolution == 0: 
        with open(os.path.join(args.out_dir, 'resolution_optimal.log'), 'a') as outfile: 
            outfile.write(token + ' ' + str(res_info[0]) + ' ' + str(res_info[1]) + '\n')
    
    # tokens in each community, communities assigned to each set of replacements based on argmax
    community_tokens, communities_sents_data, _ = community_finder.argmax_voting(communities, rep_instances)
    community_tokens = sort_community_tokens_by_popularity(rep_instances, community_tokens)
    
    # this only saves the top MOST_COMMON_CLUSTER_REPS=100 most common reps to represent each cluster
    # there must be at least 2 tokens in each cluster as representatives
    # and the second word needs to appear at least 10 times 
    clustering_data = []
    for com_tokens, _ in zip(community_tokens, communities_sents_data):
        most_common_tokens = [(int(t), v) for t, v in com_tokens[:MOST_COMMON_CLUSTER_REPS]]
        if community_big_enough_heuristics(most_common_tokens):
            clustering_data.append(most_common_tokens)
        
    return clustering_data

def community_big_enough_heuristics(most_common_tokens):
    minimum_second_word_instances = 10
    return len(most_common_tokens) > 1 and most_common_tokens[1][1] > minimum_second_word_instances

def sort_community_tokens_by_popularity(rep_instances, community_tokens):
    '''
    sorts tokens in each community based on how often they appear as replacements for an instance of the target word
    '''
    ret = []
    for comm in community_tokens:
        community_tokens_by_popularity = {t: 0 for t in comm}
        for rep_inst in rep_instances.data:
            for token in comm:
                if token in rep_inst.reps:
                    community_tokens_by_popularity[token] += 1
        community_tokens_by_popularity = [(k, v) for k, v in sorted(community_tokens_by_popularity.items(), key=lambda item: item[1], reverse=True)]
        ret.append(community_tokens_by_popularity)

    return ret

def read_clustering_data(cluster_dir, token):
    cluster_file = os.path.join(cluster_dir, f"{token}_clustering.json")
    if not os.path.exists(cluster_file): 
        return {}
    return json.load(open(cluster_file, 'r'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="replacements")
    parser.add_argument("--index_dir", type=str, default="inverted_index")
    parser.add_argument("--out_dir", type=str, default="word_clusters")
    parser.add_argument("--dataset", type=str, choices=['s2orc'])
    parser.add_argument("--case_sensitive", type=bool, default=False)
    parser.add_argument("--lemmatize", type=bool, default=False)
    parser.add_argument("--resolution", type=float, default=1.0, help="parameter for clustering, will be appended as a subfolder in out_dir. If 0, uses optimal resolution")
    parser.add_argument("--wiki_eval", type=bool, default=False, help="True if only cluster words in wiktionary evaluation set")
    parser.add_argument("--wiki_json", type=str, default="/home/lucyl/language-map-of-science/logs/wiktionary/label_to_words.json")
    parser.add_argument("--leave_out_list", type=bool, default=True, help="whether to use a SKIP_leave_out_ids.txt file in index_dir")
    args = parser.parse_args()
    
    if args.leave_out_list: 
        assert os.path.exists(os.path.join(args.index_dir, 'SKIP_leave_out_ids.txt'))

    out_dir = os.path.join(args.out_dir, str(args.resolution))
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    # assert len(os.listdir(out_dir)) == 0, f"{out_dir} already exists."

    main(args)