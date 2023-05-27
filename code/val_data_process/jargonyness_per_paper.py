import csv 
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
import json
import os
import time
import multiprocessing
import boto3
import subprocess
from transformers import BasicTokenizer
import random
import numpy as np
import argparse
import re

ROOT = '/data0/lucy/language-map-of-science/'
LOGS = ROOT + 'logs/'
DATA = ROOT + 'data/'
SENSE_DIR = LOGS + 'sense_assignments_lemmed/'
S3_S2ORC_BUCKET = 'ai2-s2-s2orc'
S3_S2ORC_PREFIX = '20200705v1/full/metadata/'
METADATA = ROOT + 'metadata/'

def get_type_scores(fos_type_folder, cutoff=0.1): 
    print("Getting type scores")
    type_scores = defaultdict(set) # { fos : [words whose npmi > cutoff ]}
    type_word_scores = defaultdict(Counter)
    for fos in tqdm(os.listdir(fos_type_folder)): 
        df = pd.read_csv(os.path.join(fos_type_folder, fos))
        tups = list(zip(df.word, df.pmi))
        for tup in tups: 
            type_word_scores[fos][tup[0]] = tup[1]
            if tup[1] <= cutoff: continue
            type_scores[fos].add(tup[0])
    return type_scores, type_word_scores

def get_sense_scores(fos_sense_file, type_word_scores, cutoff=0.1): 
    '''
    Only senses who are above the cutoff and greater 
    than their word's type NPMI are included in the output set of senses. 
    '''
    print("Getting sense scores")
    fos_sense_df = pd.read_parquet(fos_sense_file)
    fos_sense_df = fos_sense_df[fos_sense_df['count'] > 20]
    fos_sense_df[['word', 'sense']] = fos_sense_df['word_sense'].str.split('_', expand=True)
    sense_scores = defaultdict(set) # { fos : [senses whose npmi > cutoff and > type npmi]}
    sense_word_scores = defaultdict(Counter) # { fos : { sense : npmi} } 
    for fos in tqdm(fos_sense_df['fos'].unique()):
        df = fos_sense_df[fos_sense_df['fos'] == fos]
        tups = list(zip(df.word_sense, df.npmi))
        for tup in tups: 
            sense_word_scores[fos][tup[0]] = tup[1]
            word = tup[0].split('_')[0]
            type_npmi = type_word_scores[fos][word]
            if tup[1] <= cutoff or tup[1] <= type_npmi: continue
            sense_scores[fos].add(tup[0])
    return sense_scores, sense_word_scores

def get_fos_hierch(): 
    child_to_parents = defaultdict(set)
    with open(DATA + 'mag_parent_child.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            parent = row['parent_display_name'].lower()
            child = row['child_display_name'].lower()
            child_to_parents[child].add(parent)
    return child_to_parents

def get_papers_of_interest(exp_name): 
    print("Getting papers of interest")
    papers = set() # pool of papers we need to search for 
    paper_discp = {} # {paper: discipline}
    
    if exp_name == 'general_specific': 
        # only papers under single discipline
        with open(LOGS + 'general_specific/papers_of_interest.json', 'r') as infile: 
            general_specific_papers = json.load(infile)

        for key in general_specific_papers: 
            for journal in tqdm(general_specific_papers[key]): 
                tups = general_specific_papers[key][journal]
                for tup in tups: 
                    papers.add(str(tup[0]))
                    paper_discp[str(tup[0])] = tup[1]
                    
    if exp_name == 'fos_sample': 
        # only papers under single discipline
        with open(LOGS + 'wiktionary/s2orc_fos.json', 'r') as infile: 
            s2orc_fos = json.load(infile)
            
        for paper_id in s2orc_fos: 
            if len(s2orc_fos[paper_id]) == 1:
                if 'OTHER ' in s2orc_fos[paper_id][0]: continue
                paper_discp[paper_id] = s2orc_fos[paper_id][0]
                papers.add(paper_id)
                
    if exp_name == 'regression_sample': 
        paper_discp = defaultdict(dict) # {paper: {level 0 fos: [level 1 fos]}}
        
        with open(LOGS + 'regression/papers_to_regress.json', 'r') as infile: 
            papers_left = json.load(infile)
            
        child_to_parents = get_fos_hierch()
            
        for parent_fos in papers_left:
            for paper_dict in papers_left[parent_fos]: 
                paper_id = paper_dict['paper_id']
                child_fos = paper_dict['level 1 fos']
                paper_discp[paper_id][parent_fos] = child_fos
                papers.add(paper_id)
                
    return papers, paper_discp

def get_paper2senses(f): 
    sense_assign_df = pd.read_parquet(os.path.join(SENSE_DIR + '0.0/', f))
    sense_assign_df['s2orc_id'] = sense_assign_df['doc_id'].apply(lambda x: sents[f + '_' + str(x)])
    sense_assign_df = sense_assign_df[sense_assign_df['s2orc_id'].isin(papers)]
    sense_assign_df['word_clust'] = sense_assign_df['str_tok'] + '_' + sense_assign_df['cluster_id'].astype("string")
    sense_assign_df = sense_assign_df[['s2orc_id', 'word_clust', 'pos']]
    sense_assign_df['sense_pos'] = list(zip(sense_assign_df.word_clust, sense_assign_df.pos))
    paper2senses = sense_assign_df.groupby('s2orc_id')['sense_pos'].apply(list).to_dict()
    all_papers2senses = defaultdict(list)
    for paper in tqdm(paper2senses): 
        sorted_senses = sorted(paper2senses[paper], key=lambda x: x[1])
        paper_senses = defaultdict(list)
        for x in sorted_senses: 
            parts = x[0].split('_') 
            w = parts[0]
            sense_num = parts[-1]
            paper_senses[w].append(sense_num)
        all_papers2senses[paper] = paper_senses
    return all_papers2senses

def get_paper_jargon(filename, f, all_papers2senses): 
    '''
    @output: {paper : {'sense': float, 'type': float}}
    
    The WSI tokenization scheme is different from a basictokenizer 
    so we need to align them carefully, e.g. "ID" gets tokenized
    to "I D" for one and "id" for the other.  
    
    This is only for papers with a single level 1 FOS. 
    '''
    this_split_papers = set(all_papers2senses.keys())

    s3_url = f's3://ai2-s2-s2orc/{filename}'
    download_target_dir = METADATA
    download_target_path = METADATA + f + '.jsonl.gz'
    gunzip_target_path = download_target_path.replace('.gz', '')
    download = False 
    if not os.path.exists(download_target_path): 
        download = True
        args = ['aws', 's3', 'cp', s3_url, f'{download_target_dir}']
        subprocess.run(args)
    args = ['gunzip', '-f', download_target_path]
    subprocess.run(args)
    
    tokenizer = BasicTokenizer(do_lower_case=True)
    
    paper_prop_jargon = defaultdict(dict) # {paper : {'sense': float, 'type': float}}
    paper_max_npmi = defaultdict(list) # {paper : [max npmi of first n token for each index n]} 
    
    with open(gunzip_target_path, 'r') as infile: 
        for line in tqdm(infile): 
            d = json.loads(line)
            paper_id = str(d['paper_id'])
            if paper_id not in this_split_papers: continue
            paper_senses = all_papers2senses[paper_id]
            discp = paper_discp[paper_id] 

            # we don't count jargon words in titles, only abstracts
            # but that means we need to skip over word senses that appear
            # in titles
            title_tokens = tokenizer.tokenize(d['title'])
            for token in title_tokens: 
                if token in sense_vocab: 
                    # iterate through sense list for each word paper
                    lemma = lemma_storage[token]
                    if lemma in paper_senses and len(paper_senses[lemma]) > 0: 
                        paper_senses[lemma] = paper_senses[lemma][1:]
            
            abstract_tokens = tokenizer.tokenize(d['abstract'])
            
            sense_count = 0
            type_count = 0
            
            token_scores = []
            for token in abstract_tokens: 
                if not re.search('[a-z]', token): continue # exclude numbers & punctuation
                npmi = type_word_scores[discp][token]
                if token in sense_vocab: 
                    lemma = lemma_storage[token]
                    npmi = type_word_scores[discp][lemma]
                    # find lemma's list of senses in paper_senses
                    if lemma in paper_senses and len(paper_senses[lemma]) > 0: 
                        sense_num = paper_senses[lemma][0]
                        word_sense = lemma + '_' + sense_num
                        sense_npmi = sense_word_scores[discp][word_sense]
                        if sense_npmi > npmi: 
                            npmi = sense_npmi
                        if word_sense in sense_scores[discp]: 
                            # field specific sense
                            sense_count += 1
                        elif lemma in type_scores[discp] or token in type_scores[discp]: 
                            # if not a field specific sense, check if word is field specific
                            type_count += 1
                        paper_senses[lemma] = paper_senses[lemma][1:]
                    elif lemma in type_scores[discp] or token in type_scores[discp]: 
                        type_count += 1
                elif token in type_scores[discp]: 
                    type_count += 1
                token_scores.append(npmi)
            if not token_scores: continue
            max_npmi_first_n = [0] * len(token_scores)
            i = 0
            for token in abstract_tokens: 
                if not re.search('[a-z]', token): continue # exclude numbers & punctuation
                max_npmi_first_n[i] = max(token_scores[:i+1])
                i += 1
                if i == 100: break
            paper_max_npmi[paper_id] = max_npmi_first_n
            paper_prop_jargon[paper_id]['sense'] = sense_count / float(len(token_scores))
            paper_prop_jargon[paper_id]['type'] = type_count / float(len(token_scores))
    
    os.remove(gunzip_target_path)
    
    return paper_prop_jargon, paper_max_npmi

def get_paper_jargon_multiple_fos(filename, f, all_papers2senses): 
    '''
    This is for papers that have one or two level 1 FOS per level 0 FOS
    that they are associated with
    
    We don't calculate expected maximum NPMI for this. 
    '''
    this_split_papers = set(all_papers2senses.keys())

    s3_url = f's3://ai2-s2-s2orc/{filename}'
    download_target_dir = METADATA
    download_target_path = METADATA + f + '.jsonl.gz'
    gunzip_target_path = download_target_path.replace('.gz', '')
    download = False 
    if not os.path.exists(download_target_path): 
        download = True
        args = ['/home/lucyl/bin/aws', 's3', 'cp', s3_url, f'{download_target_dir}']
        subprocess.run(args)
    args = ['gunzip', '-f', download_target_path]
    subprocess.run(args)
    
    tokenizer = BasicTokenizer(do_lower_case=True)
    
    paper_prop_jargon = defaultdict(dict) # {paper_fos : {'sense': float, 'type': float}}
    
    with open(gunzip_target_path, 'r') as infile: 
        for line in tqdm(infile): 
            d = json.loads(line)
            paper_id = str(d['paper_id'])
            if paper_id not in this_split_papers: continue
            paper_senses = all_papers2senses[paper_id]

            # we don't count jargon words in titles, only abstracts
            # but that means we need to skip over word senses that appear
            # in titles
            title_tokens = tokenizer.tokenize(d['title'])
            for token in title_tokens: 
                if token in sense_vocab: 
                    # iterate through sense list for each word paper
                    lemma = lemma_storage[token]
                    if lemma in paper_senses and len(paper_senses[lemma]) > 0: 
                        paper_senses[lemma] = paper_senses[lemma][1:]
            
            abstract_tokens = tokenizer.tokenize(d['abstract'])
            
            sense_count = Counter() # {parent fos: count}
            type_count = Counter() # {parent fos: count}
            
            total_token_count = 0.0
            for token in abstract_tokens: 
                if not re.search('[a-z]', token): continue # exclude numbers & punctuation
                    
                for parent_fos in paper_discp[paper_id]: 
                    sense_scores_discp = set()
                    type_scores_discp = set()
                    for discp in paper_discp[paper_id][parent_fos]: 
                        sense_scores_discp.update(sense_scores[discp])
                        type_scores_discp.update(type_scores[discp])
                        
                    if token in sense_vocab: 
                        lemma = lemma_storage[token]
                        # find lemma's list of senses in paper_senses
                        if lemma in paper_senses and len(paper_senses[lemma]) > 0: 
                            sense_num = paper_senses[lemma][0]
                            word_sense = lemma + '_' + sense_num
                            if word_sense in sense_scores_discp: 
                                sense_count[parent_fos] += 1
                            elif lemma in type_scores_discp or token in type_scores_discp: 
                                type_count[parent_fos] += 1
                            paper_senses[lemma] = paper_senses[lemma][1:]
                        elif lemma in type_scores_discp or token in type_scores_discp: 
                            type_count[parent_fos] += 1
                    elif token in type_scores_discp: 
                        type_count[parent_fos] += 1
                        
                total_token_count += 1.0
            if total_token_count == 0: continue
                
            for parent_fos in paper_discp[paper_id]: 
                paper_fos_key = str(paper_id) + '@' + parent_fos
                paper_prop_jargon[paper_fos_key]['sense'] = sense_count[parent_fos] / total_token_count
                paper_prop_jargon[paper_fos_key]['type'] = type_count[parent_fos] / total_token_count
    
    os.remove(gunzip_target_path)
    
    return paper_prop_jargon, None

def process_batch(batch): 
    all_papers2senses = get_paper2senses(batch['f'])
    if args.exp_name == 'regression_sample': 
        paper_scores, paper_max_npmi = get_paper_jargon_multiple_fos(batch['filename'], 
                                                    batch['f'], 
                                                    all_papers2senses)
    else: 
        paper_scores, paper_max_npmi = get_paper_jargon(batch['filename'], 
                                                    batch['f'], 
                                                    all_papers2senses)
    
    with open(os.path.join(batch['paper_score_folder'], batch['f'] + '.json'), 'w') as outfile: 
        json.dump(paper_scores, outfile)
        
    if paper_max_npmi: 
        with open(os.path.join(batch['max_npmi_folder'], batch['f'] + '.json'), 'w') as outfile: 
            json.dump(paper_max_npmi, outfile)
        
def parse_inputs():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--cutoff", type=float, default=0.1)
    parser.add_argument("--exp_name", type=str, required=True, choices=['general_specific', 
                                                                        'fos_sample', 
                                                                        'regression_sample'])
    
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_inputs()
    
    with open(LOGS + 'inverted_index/SKIP_paperID_sent.json', 'r') as infile: 
        sents = json.load(infile) # {split_sentID : s2orc ID}

    fos_type_folder = LOGS + 'type_npmi/fos_set-False_lemma-True/'
    type_scores, type_word_scores = get_type_scores(fos_type_folder, cutoff=args.cutoff)

    # load senses for each fos 
    fos_sense_file = LOGS + 'fos_senses/es-True_res-0.0/ALL_scores.parquet'
    sense_scores, sense_word_scores = get_sense_scores(fos_sense_file, type_word_scores, cutoff=args.cutoff)

    papers, paper_discp = get_papers_of_interest(args.exp_name)

    sense_vocab = set()
    with open(LOGS + 'sense_vocab/wsi_vocab_set_98_50.txt', 'r') as infile: 
        for line in infile: 
            sense_vocab.add(line.strip())

    with open(LOGS + 'sense_vocab/' + 'all_lemmas-for-type-npmi.json', 'r') as infile: 
        lemma_storage = json.load(infile)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(S3_S2ORC_BUCKET)
    metadata_files = {}
    for obj in bucket.objects.filter(Prefix=S3_S2ORC_PREFIX + 'metadata'): 
        filename = obj.key
        short_name = filename.split('/')[-1].replace('.jsonl.gz', '')
        metadata_files[short_name] = filename

    paper_score_folder = LOGS + args.exp_name + '/paper_scores/' + str(args.cutoff) + '/'
    os.makedirs(paper_score_folder, exist_ok=True)
    max_npmi_folder = LOGS  + args.exp_name + '/paper_first_n_max/' + str(args.cutoff) + '/'
    os.makedirs(max_npmi_folder, exist_ok=True)

    print("making batches...")
    batches = [{
        'f': f,
        'filename': metadata_files[f],
        'paper_score_folder': paper_score_folder,
        'max_npmi_folder': max_npmi_folder,
    } for f in sorted(os.listdir(SENSE_DIR + '0.0/')) if f.startswith('metadata_')]

    print("Batches to do:", len(batches))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 3) as p:
        p.map(process_batch, batches) 

    print("done") 
