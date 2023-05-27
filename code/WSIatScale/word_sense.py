"""
This file computes npmi of senses in each FOS group / journal. 

    PMI for word senses is defined as 
    log(p(sense|group) / p(sense)) 
    or 
    log of 
    (frequency of sense in group j / # of times word appears in group j) 
    ______________________________________________________________ 
    (frequency of sense in all j's / total number of word occurrences)
    This is then normalized by h(w, j), or
    
    -log p(w, j) = -log ((frequency of sense in journal j) / (total number of word occurrences))
    
We don't actually use the per_abstract flag here for sense counts. 
"""
import json
import os
from collections import Counter, defaultdict
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count, Queue
from functools import partial
import math
import csv
import numpy as np
import random
import time
import gc
import argparse

# ROOT = '/home/lucyl/language-map-of-science/'
# DATA = ROOT + 'data/'
# LOGS = ROOT + 'logs/'
# REPLACEMENTS = LOGS + 'replacements/replacements/'
# INDEX_DIR = LOGS + 'inverted_index/'
# TYPE_NPMI = LOGS + 'type_npmi/'

def get_docID_to_fos(): 
    '''
    Loads the output of fos_analysis.py, 
    which is FOS to list of s2orc paper IDs. 
    
    Then, creates a mapping from sense_input
    doc IDs to FOS
    '''
    with open(os.path.join(args.input_dir, 'docID2fos.json'), 'r') as infile: 
        docID2fos = json.load(infile)
    return docID2fos

def get_docID_to_journal(): 
    '''
    creates a mapping from sense_input doc IDs to journals
    '''
    with open(os.path.join(args.input_dir, 'docID2journal.json'), 'r') as infile: 
        docID2journal = json.load(infile)
    return docID2journal

def process_batch(data_split, sense_assign_folder, docID2group, group_name='fos'): 
    '''
    The output of sense assignment are parquets, one per metadata split
    
    Look into this in the future: https://github.com/jmcarpenter2/swifter
    
    For FOS, each doc has multiple FOS labels, while for journals, it is a one-to-one mapping. 
    '''
    in_path = os.path.join(sense_assign_folder, data_split)
    df = pd.read_parquet(in_path)
    df['token_clust'] = df['str_tok'] + '_' + df['cluster_id'].astype("string")
    df['key'] = data_split + '_' + df['doc_id'].astype("string")
    if data_split.startswith('split_'): # wikipedia senses, add to background
        df[group_name] = 'all'
    elif data_split.startswith('metadata_'): # s2orc senses
        df = df.loc[df['key'].isin(set(docID2group.keys()))]
        if group_name == 'fos': 
            # getting fos list and exploding it
            df[group_name] = df["key"].apply(lambda x: docID2group[x] + ['all'])
            df = df.explode(group_name, ignore_index=True)
        elif group_name == 'journal': 
            df[group_name] = df["key"].apply(lambda x: docID2group[x])
    df = df[[group_name, 'token_clust']] 
    # get lists of token cluster assignments 
    groups = df.groupby(group_name, sort=False)['token_clust'].apply(list).to_dict()

    group_sense_counts = defaultdict(Counter) # {group : { word_senseID : count } } 
    for group in groups: 
        group_sense_counts[group] = Counter(groups[group])

    return group_sense_counts

def chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    https://stackoverflow.com/questions/312443/how-do-i-split-a-list-into-equally-sized-chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def count_senses(sense_assign_folder, out_path, group_name='fos'): 
    '''
    Inputs: 
    - sense assignments in each doc as a parquet with columns: 
        filename (batch number), doc_id, str_tok, cluster_id, pos, and best_jaccard_score
    
    Outputs: 
    - parquet containing 'word', 'cluster_id', 'fos', 'count'
    - frequency of each vocab word in each group j 
    - frequency of each sense in each group j 
    '''
    if group_name == 'fos': 
        docID2group = get_docID_to_fos()
    elif group_name == 'journal': 
        docID2group = get_docID_to_journal()
    else: 
        print("Invalid group_name.")
        return 
    
    group_sense_counts = defaultdict(Counter) # {group : { word_senseID : count } } 
    files = os.listdir(sense_assign_folder)
    
    partial_process_batch = partial(process_batch,
                                    sense_assign_folder=sense_assign_folder,
                                    docID2group=docID2group,
                                    group_name=group_name,
                                    )
    print("Processing data splits...")
    # results are too big for memory, so need to do file ranges 
    for chunk in chunks(files, 20): 
        with Pool(cpu_count()) as p:
            results = list(tqdm(p.imap(partial_process_batch, chunk), total=len(chunk)))
        # reduce 
        for res in results: 
            for group in res: 
                group_sense_counts[group].update(res[group])
        results = [] 

    d = {
        'word': [], 
        'sense': [],
        group_name: [], 
        'count': [],  
    }
    
    for group in tqdm(group_sense_counts): 
        for word_senseID in group_sense_counts[group]: 
            parts = word_senseID.split('_')
            d['word'].append(parts[0])
            d['sense'].append(parts[1])
            d[group_name].append(group)
            d['count'].append(group_sense_counts[group][word_senseID])
            
    del group_sense_counts
    gc.collect()
            
    df = pd.DataFrame(data=d)
    del d
    gc.collect()
    df.to_parquet(out_path) 

def calc_npmi(df, out_folder, vocab, count_dict, group_name='fos', wiki_only=False): 
    '''
    This function calculates sense NPMI, and is called by calc_npmi_main(). 
    '''
    if group_name not in ['fos', 'journal']: return
    print("Calculating NPMI...")
    
    if not os.path.exists(out_folder): 
        os.makedirs(out_folder)
        
    df['word_sense'] = df[['word', 'sense']].agg('_'.join, axis=1)
    print("Get background, groups, and total sense counts...")
    if group_name == 'fos': 
        background = df[df[group_name] == 'all']
        if wiki_only: 
            with open(os.path.join(args.input_dir, 'label_to_words.json'), 'r') as infile: 
                label_to_words = json.load(infile) 
        else: 
            label_to_words = {}
        # exclude background 'all'
        groups = set(df[group_name].to_list()) - set(['all'])
        # total counts of each word sense overall
        total_sense = Counter(background.groupby(['word_sense']).sum().to_dict()['count'])
        result_d = { # write all results into a parquet 
            'fos': [],
            'word_sense': [],
            'npmi': [], 
            'count': []
        }
        
    elif group_name == 'journal': 
        background = df
        label_to_words = {}
        # exclude wikipedia background 'all'
        groups = set(df[group_name].to_list()) - set(['all'])
        # total counts of each word sense overall
        total_sense = background.groupby(['word_sense'], sort=False).sum().to_dict()['count']
        result_d = { # write all results into a parquet 
            'journal': [],
            'word_sense': [],
            'npmi': [], 
            'count': []
        }
    
    df = df.sort_values(group_name) # speeds up things later
    
    print("Get total counts...")
    # total counts of each word overall
    total_counts = background.groupby(['word'], sort=False).sum().to_dict()['count']
    
    print("Iterating through groups...")
    for group in tqdm(groups): 
        if wiki_only and group_name == 'fos' and group.lower() not in label_to_words: continue
        pmi_d = {}
        start_idx = df[group_name].searchsorted(group, 'left')
        end_idx = df[group_name].searchsorted(group, 'right')
        this_df = df[start_idx:end_idx]
        # total counts of each word in group
        word_counts = Counter(this_df.groupby(['word'], sort=False).sum().to_dict()['count'])
        sense_counts = Counter(pd.Series(this_df['count'].values, index=this_df.word_sense).to_dict())
        for tup in word_counts.most_common(): 
            w = tup[0]
            c = tup[1]
            # w must have a type NPMI score 
            if w not in vocab[group]: continue
            word_df = this_df[this_df['word'] == w]
            senseIDs = set(word_df['sense'].to_list())
            for sense in senseIDs:
                w_s = w + '_' + sense
                total_sense_c = total_sense[w_s]
                sense_c = sense_counts[w_s]
                if count_dict: # use all words event space
                    p_w_given_j = sense_c / count_dict[group]
                    p_w = total_sense_c / count_dict['background']
                    pmi = math.log(p_w_given_j / p_w)
                    h = -math.log(sense_c / count_dict['background'])
                else: # use target sense's word's event space
                    p_w_given_j = sense_c / c
                    p_w = total_sense_c / total_counts[w]
                    pmi = math.log(p_w_given_j / p_w)
                    h = -math.log(sense_c / total_counts[w])
                pmi_d[w_s] = pmi / h
                if group_name == 'journal' or (group_name == 'fos' and not wiki_only): 
                    result_d[group_name].append(group)
                    result_d['word_sense'].append(w_s)
                    result_d['npmi'].append(pmi_d[w_s])
                    result_d['count'].append(sense_c)
                
        if group_name == 'fos' and wiki_only: 
            with open(out_folder + group, 'w') as outfile: 
                sorted_d = sorted(pmi_d.items(), key=lambda kv: kv[1])
                writer = csv.writer(outfile)
                writer.writerow(['word_sense', 'npmi', 'count'])
                for tup in sorted_d: 
                    writer.writerow([tup[0], str(tup[1]), str(sense_counts[tup[0]])])
                    
    if group_name == 'journal' or (group_name == 'fos' and not wiki_only): 
        df = pd.DataFrame(data=result_d)
        df.to_parquet(out_folder + 'ALL_scores.parquet') 
                
def calc_npmi_main(in_path, out_folder, type_npmi_path, group_name='fos', wiki_only=False, type_event_space=False): 
    '''
    Calls calc_npmi() for various models. 
    '''
    # get type NPMI vocabulary so that vocabs are the same
    print("Getting vocabulary for calculating NPMI...")
    if group_name == 'fos': 
        vocab = defaultdict(set)
        for fos in os.listdir(type_npmi_path): 
            df = pd.read_csv(os.path.join(type_npmi_path, fos))
            vocab[fos] = set(df['word'].to_list())
    elif group_name == 'journal': 
        df = pd.read_parquet(type_npmi_path + 'ALL_scores.parquet')
        df = df[['journal', 'word']].drop_duplicates()
        vocab = df.groupby('journal')['word'].apply(set).to_dict()
        vocab = defaultdict(set, vocab)
    else: 
        return
    
    count_dict = None
    if type_event_space and group_name == 'fos': 
        # need to load in total number of words per fos, total number of words overall
        count_df = pd.read_parquet(os.path.join(args.input_dir, 'word_counts_fos_set-False.parquet'))
        count_dict = count_df.groupby(['fos'], sort=False).sum().to_dict()['count']
        background = pd.read_parquet(os.path.join(args.input_dir, 'word_counts_wiki_set-False.parquet'))
        wiki_background = background[background['dataset'] == 'enwikifos']
        wikipedia_total = sum(wiki_background['count'].to_list())
        count_dict['background'] += wikipedia_total
    elif type_event_space and group_name == 'journal': 
        print("TYPE EVENT SPACE FOR JOURNALS NOT IMPLEMENTED YET")
    
    df = pd.read_parquet(in_path) 
    calc_npmi(df, out_folder, vocab, count_dict, group_name=group_name, wiki_only=wiki_only)
                
def get_best_candidate_nodef(in_folder, out_folder, prefix='', group_name='fos'): 
    '''
    Two methods for choosing the best candidate sense without definitions: 
    - 'most': most common sense 
    - 'max': sense with the max npmi 
    
    Output: 
    - { group { word : NPMI of best candidate } } 
    
    This function is written to handle group_name == 'journal' if we need it
    but it seems like for our science of science analyses we might instead just use word_sense
    like we use word for type NPMI. 
    '''
    print("Getting the best candidates...")
    
    all_most_res = {}
    all_max_res = {}
    
    if group_name == 'fos': 
        groups = os.listdir(in_folder)
    elif group_name == 'journal': 
        df = pd.read_parquet(in_folder + 'ALL_scores.parquet') 
        groups = df['journal'].unique()
    else: 
        return
    
    for group in tqdm(groups): 
        if group_name == 'fos': 
            sense_counts = defaultdict(Counter) # { word : { sense : count } } 
            sense_npmi = defaultdict(Counter) # { word : { sense : npmi } } 
            with open(in_folder + group, 'r') as infile: 
                reader = csv.DictReader(infile) 
                for row in reader: 
                    parts = row['word_sense'].split('_')
                    w = parts[0]
                    sense = parts[1]
                    sense_counts[w][sense] = int(row['count'])
                    sense_npmi[w][sense] = float(row['npmi'])
        elif group_name == 'journal': 
            this_df = df[df['journal'] == group]
            this_sense_counts = Counter(pd.Series(this_df['count'].values, index=this_df.word_sense).to_dict())
            sense_counts = defaultdict(Counter) # { word : { sense : count } } 
            for k in this_sense_counts: 
                parts = k.split('_')
                w = parts[0]
                sense = parts[1]
                sense_counts[w][sense] = this_sense_counts[k]
            this_sense_npmi = Counter(pd.Series(this_df['npmi'].values, index=this_df.word_sense).to_dict())
            sense_npmi = defaultdict(Counter) # { word : { sense : npmi } } 
            for k in this_sense_npmi: 
                parts = k.split('_')
                w = parts[0]
                sense = parts[1]
                sense_npmi[w][sense] = this_sense_npmi[k]
                
        most_res = {} # { word_sense : npmi } 
        max_res = {} # { word_sense : npmi } 
        for w in sense_counts: 
            tup = sense_counts[w].most_common(1)[0]
            most_common_count = tup[1]
            npmi = sense_npmi[w][tup[0]]
            most_res[w] = npmi
            
            if most_common_count <= 20: 
                # back off onto the most common sense 
                max_res[w] = npmi
            else: 
                for tup in sense_npmi[w].most_common(): 
                    sense_c = sense_counts[w][tup[0]]
                    if sense_c > 20: 
                        # common enough
                        max_res[w] = tup[1]
                        break
                    # otherwise keep going down the ordered list
            
        all_most_res[group] = most_res
        all_max_res[group] = max_res
                
    candidate_type='most'
    outpath = os.path.join(out_folder, prefix + candidate_type + '_' + group_name + '_sense_npmi.json')
    with open(outpath, 'w') as outfile: 
        json.dump(all_most_res, outfile)

    candidate_type='max'
    outpath = os.path.join(out_folder, prefix + candidate_type + '_' + group_name + '_sense_npmi.json')
    with open(outpath, 'w') as outfile:
        json.dump(all_max_res, outfile)
        
def wiktionary_eval_pipeline(): 
    '''
    This function does the following for different resolution values: 
    - count senses 
    - calculate npmi of each word's senses
    - get the best candidate npmi score for each word 
    
    This operates at the level of FOS (so the npmi_folder is fos_senses). 
    '''
    resolution_vals = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    
    for res in tqdm(resolution_vals): 
        sense_assign_folder = os.path.join(args.input_dir, str(res))
        sense_counts_folder = 'sense_counts/fos_eval/' + str(res)
        os.makedirs(sense_counts_folder, exist_ok=True)
        npmi_folder_es = os.path.join(args.output_dir, 'fos_senses_eval/es-True_res-' + str(res) + '/')
        os.makedirs(npmi_folder_es, exist_ok=True)
        npmi_folder = os.path.join(args.output_dir, 'fos_senses_eval/res-' + str(res) + '/')
        os.makedirs(npmi_folder, exist_ok=True)
        best_cand_folder = os.path.join(args.output_dir, 'wiktionary_cands/')
        os.makedirs(best_cand_folder, exist_ok=True)
        best_cand_folder_es = os.path.join(args.output_dir, 'wiktionary_cands_es/')
        os.makedirs(best_cand_folder_es, exist_ok=True)
        type_npmi_path = os.path.join(args.input_dir, 'type_npmi/fos_set-False_lemma-True')
        
        sense_count_parquet = os.path.join(sense_counts_folder, 'sense_counts.parquet')
        
        count_senses(sense_assign_folder, sense_count_parquet)
        calc_npmi_main(sense_count_parquet, npmi_folder, type_npmi_path, wiki_only=True, type_event_space=False)
        calc_npmi_main(sense_count_parquet, npmi_folder_es, type_npmi_path, wiki_only=True, type_event_space=True)
        get_best_candidate_nodef(npmi_folder, best_cand_folder, prefix='res-' + str(res) + '_')
        get_best_candidate_nodef(npmi_folder_es, best_cand_folder_es, prefix='res-' + str(res) + '_')

def journal_dataset_pipeline(): 
    '''
    This function does the following for the entire dataset: 
    - count senses 
    - calculate npmi of each word's senses
    - get the best candidate npmi score for each word 
    '''
    print("WARNING: this is depcrated/needs to be updated. Currently we are using all Wikipedia sense splits, we need to filter to a smaller subset for journal sense NPMI.")
    return
    res = 0.0
    sense_assign_folder = os.path.join(args.input_dir, str(res))
    sense_counts_folder = 'sense_counts/journals/' + str(res)
    npmi_folder = os.path.join(args.output_dir, 'journal_senses/res-' + str(res) + '/')
    type_npmi_path = os.path.join(args.input_dir, 'type_npmi/journal_lemma-False/')
    
    if not os.path.exists(sense_counts_folder): 
        os.makedirs(sense_counts_folder)
    if not os.path.exists(npmi_folder): 
        os.makedirs(npmi_folder)
        
    sense_count_parquet = os.path.join(sense_counts_folder, 'sense_counts.parquet')

    count_senses(sense_assign_folder, sense_count_parquet, group_name='journal')
    calc_npmi_main(sense_count_parquet, npmi_folder, type_npmi_path, group_name='journal')
    
def fos_dataset_pipeline(): 
    '''
    This function does the following for the entire dataset: 
    - count senses 
    - calculate npmi of each word's senses
    - get the best candidate npmi score for each word 
    '''
    res = 0.0
    type_event_space = True
    sense_assign_folder = os.path.join(args.input_dir, str(res))
    sense_counts_folder = 'sense_counts/fos/' + str(res)
    if type_event_space: 
        npmi_folder = os.path.join(args.output_dir, 'fos_senses/es-True_res-' + str(res) + '/')
    else: 
        npmi_folder = os.path.join(args.output_dir, 'fos_senses/res-' + str(res) + '/')
    type_npmi_path = os.path.join(args.input_dir, 'type_npmi/fos_set-False_lemma-True')
    
    if not os.path.exists(sense_counts_folder): 
        os.makedirs(sense_counts_folder)
    if not os.path.exists(npmi_folder): 
        os.makedirs(npmi_folder)
        
    sense_count_parquet = os.path.join(sense_counts_folder, 'sense_counts.parquet')

    count_senses(sense_assign_folder, sense_count_parquet, group_name='fos')
    calc_npmi_main(sense_count_parquet, npmi_folder, type_npmi_path, group_name='fos', type_event_space=type_event_space)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exp", type=str, required=True)
    args = parser.parse_args()
    if args.exp == 'fos': 
        fos_dataset_pipeline()
    elif args.exp == 'wiki': 
        wiktionary_eval_pipeline()
    elif args.exp == 'journal': 
        journal_dataset_pipeline()
