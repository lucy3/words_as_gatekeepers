# pylint: disable=no-name-in-module
# pylint: disable=import-error
import argparse
import os
import numpy as np
from functools import partial
from operator import itemgetter
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import json
import pandas as pd

from transformers import AutoTokenizer

from analyze import npy_file_path, RepInstances
from cluster_reps_per_token import read_clustering_data
from utils import tokenizer_params, jaccard_score_between_elements, get_model_stopwords

SENTS_BY_CLUSTER = 'sents_by_cluster'
ALIGNED_SENSE_IDX_FOLDER = 'aligned_sense_idx'

def main(args):
    '''
    The output files are named after each batch ID in write_mask_preds.py
    '''
    model_hf_path = tokenizer_params[args.dataset]
    tokenizer = AutoTokenizer.from_pretrained(model_hf_path, use_fast=True)
    stopwords = get_model_stopwords(tokenizer)
    tokens_with_clusters = [f.replace('.jsonl', '') for f in os.listdir(args.index_dir) if not f.startswith('SKIP')]
    
    if args.lemmatize: 
        with open(os.path.join(args.index_dir, "SKIP_lemma_all.json"), 'r') as infile: 
            tokenizer_map = json.load(infile) # {token string : token IDs}
        with open(os.path.join(args.index_dir, "SKIP_lemma.json"), 'r') as infile: 
            str_to_wordID_map = json.load(infile) # {token string : token IDs}
    elif args.case_sensitive: 
        tokenizer_map = None
        with open(os.path.join(args.index_dir, "SKIP_identity.json"), 'r') as infile: 
            str_to_wordID_map = json.load(infile) # {token string : token IDs}
    else: 
        with open(os.path.join(args.index_dir, "SKIP_lower_to_upper_all.json"), 'r') as infile: 
            tokenizer_map = json.load(infile) # {token string : token IDs}
        with open(os.path.join(args.index_dir, "SKIP_lower_to_upper.json"), 'r') as infile: 
            str_to_wordID_map = json.load(infile) # {token string : token IDs}
            
    if args.wiki_eval: 
        # Only assign senses for on tokens that appear in cluster_dir
        cluster_dir = os.path.join(args.cluster_dir, str(args.resolution))
        str_toks = [f.replace('_clustering.json', '') for f in os.listdir(cluster_dir)]
        wordIDs_with_clusters = set()
        for tok in str_toks: 
            wordIDs = [str(w) for w in str_to_wordID_map[tok]]
            wordIDs_with_clusters.update(wordIDs)
        tokens_with_clusters = set(tokens_with_clusters) & wordIDs_with_clusters
    
    token_to_str_tok = get_token_to_str_tok(tokens_with_clusters, tokenizer, str_to_wordID_map)

    for folder in os.listdir(args.data_dir): 
        out_path = os.path.join(args.out_dir, str(args.resolution), folder)
        if os.path.exists(out_path): 
            print(folder, "is already done")
            continue
        
        files = data_files(os.path.join(args.data_dir, folder))

        print(f"total {len(files)} files in {folder}")
        partial_find_and_write = partial(find_and_write,
            args=args,
            tokens_with_clusters=tokens_with_clusters, 
              token_to_str_tok=token_to_str_tok,
                tokenizer=tokenizer, 
                stopwords=stopwords,
                tokenizer_map=tokenizer_map,
                this_dir=folder)
        
        with Pool(cpu_count() // 2) as p:
            all_dfs = list(tqdm(p.imap(partial_find_and_write, files), total=len(files)))
        df = pd.concat(all_dfs, ignore_index=True)
        df.to_parquet(out_path) 
        del all_dfs # free memory
        del df # free memory

def get_token_to_str_tok(tokens, tokenizer, str_to_wordID_map):
    token_to_str_tok = {} # {token ID : str_tok}
            
    for s in str_to_wordID_map: 
        for wordID in str_to_wordID_map[s]: 
            token_to_str_tok[int(wordID)] = s
    return token_to_str_tok

def find_and_write(filename, args, tokens_with_clusters, token_to_str_tok, tokenizer, stopwords, tokenizer_map, this_dir):
    '''
    called by main()
    @inputs: 
    - filename: name of write_mask_preds.py outputs without the '-*.npy'
    - args: I didn't think we need to pass these in but the original script does this
    '''
    this_datadir = os.path.join(args.data_dir, this_dir)
    cluster_dir = os.path.join(args.cluster_dir, str(args.resolution))
    tokens_to_clusters = find_clusters(filename, this_datadir, cluster_dir, \
                                       tokens_with_clusters, token_to_str_tok, tokenizer, stopwords, tokenizer_map)
        
    return tokens_to_clusters

def data_files(replacements_dir):
    '''
    called by main()
    '''
    files = set()
    for file in os.listdir(replacements_dir):
        splits = file.split('-')
        files.add(splits[0]) # batch_ID, e.g. "1707" in 1707-tokens.npy
    
    return files

def get_pos_to_doc(lengths, doc_ids):
    '''
    @output: 
    - {index in tokens : doc_id} 
    '''
    pos_to_doc = {}
    pos = 0
    for length, doc_id in zip(lengths, doc_ids):
        for i in range(length): 
            pos_to_doc[pos] = doc_id
            pos += 1
    return pos_to_doc

def find_clusters(filename, data_dir, cluster_dir, tokens_with_clusters, token_to_str_tok, tokenizer, stopwords, tokenizer_map):
    '''
    @inputs: 
    - filename: name of write_mask_preds.py outputs without the '-*.npy'. Essentially, the output of one batch of inputs. 
    - data_dir: replacements folder 
    
    This is changed from their original implementation where we only assign clusters
    for words that have clusters induced, not all tokens. We also make sure
    every instance of a vocab word is assigned a sense. That is, if there is no overlap,
    we assign the word to an additional "other" sense. 
    
    The output should be a dataframe.
    '''
    n_reps = 5
    tokens_to_clusters = {
        'filename': [],
        'doc_id': [], 
        'str_tok': [],
        'cluster_id': [], 
        'pos': [],  
        'best_jaccard_score': [],
    }
    all_tokens = np.load(npy_file_path(data_dir, filename, 'tokens'), mmap_mode='r')
    all_reps = np.load(npy_file_path(data_dir, filename, 'reps'), mmap_mode='r')
    all_docs = np.load(npy_file_path(data_dir, filename, 'doc_ids'), mmap_mode='r')
    all_lengths = np.load(npy_file_path(data_dir, filename, 'lengths'), mmap_mode='r')
    
    pos_to_doc = get_pos_to_doc(all_lengths, all_docs) # position to doc ID
    
    for pos, (token, token_reps) in enumerate(zip(all_tokens, all_reps)):
        if str(token) in tokens_with_clusters: 
            doc_id = pos_to_doc[pos]
            # if we were to do lemmatization, add it here
            rep_inst = RepInstances(tokenizer=tokenizer, stopwords=stopwords, tokenizer_map=tokenizer_map)
            rep_inst.clean_and_populate_reps(reps=token_reps)
            #rep_inst.populate_specific_size(n_reps) # we only saved 5 so no need for this
            top_token_reps = rep_inst.data[0].reps
            str_tok = token_to_str_tok[token]
            clustering_data = read_clustering_data(cluster_dir, str_tok) 
            if not clustering_data: 
                # no clusters were created because all reps were removed / not enough examples
                continue
            token_precomputed_clusters = clustering_data[str(n_reps)]
            jaccard_scores = []
            for pre_computed_cluster in token_precomputed_clusters:
                pre_computed_cluster_set = set([d[0] for d in pre_computed_cluster])
                similarity = jaccard_score_between_elements(pre_computed_cluster_set, top_token_reps)
                jaccard_scores.append(similarity)

            extra_cluster_id = len(jaccard_scores) # no overlap with any precomputed cluster
            if len(jaccard_scores) > 0:
                cluster_id, best_jaccard_score = max(enumerate(jaccard_scores), key=itemgetter(1))
                if best_jaccard_score > 0:
                    tokens_to_clusters = populate_df_dict(doc_id, str_tok, cluster_id, pos, \
                                                          best_jaccard_score, filename, tokens_to_clusters)
                else: 
                    tokens_to_clusters = populate_df_dict(doc_id, str_tok, extra_cluster_id, \
                                                          pos, -1, filename, tokens_to_clusters)
            else: 
                # no precomputed clusters (no community big enough to be considered)
                # assign everyone to a single cluster
                tokens_to_clusters = populate_df_dict(doc_id, str_tok, extra_cluster_id, \
                                                          pos, -1, filename, tokens_to_clusters)
                    
    tokens_to_clusters = pd.DataFrame(data=tokens_to_clusters)
    return tokens_to_clusters

def populate_df_dict(doc_id, str_tok, cluster_id, pos, best_jaccard_score, filename, tokens_to_clusters): 
    tokens_to_clusters['doc_id'].append(doc_id)
    tokens_to_clusters['str_tok'].append(str_tok)
    tokens_to_clusters['cluster_id'].append(cluster_id)
    tokens_to_clusters['pos'].append(pos)
    tokens_to_clusters['best_jaccard_score'].append(best_jaccard_score)
    tokens_to_clusters['filename'].append(int(filename))
    return tokens_to_clusters

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="replacements")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--cluster_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, choices=['s2orc'])
    parser.add_argument("--index_dir", type=str, required=True)  
    parser.add_argument("--case_sensitive", type=bool, default=False)
    parser.add_argument("--lemmatize", type=bool, default=False)
    parser.add_argument("--wiki_eval", type=bool, default=False, help="True if only cluster words in wiktionary evaluation set")
    parser.add_argument("--resolution", type=float, default=1.0, help="parameter for clustering, will be appended as a subfolder in out_dir")
    
    args = parser.parse_args()
    
    assert not (args.case_sensitive and args.lemmatize), "Cannot be both case sensitive and lemmatize"

    out_dir = os.path.join(args.out_dir, str(args.resolution))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    main(args)