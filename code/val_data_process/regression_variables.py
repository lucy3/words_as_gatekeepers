"""
This script generates data for
some regression variables

and takes ~1 hr to run. 
"""
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

ROOT = '/home/lucyl/language-map-of-science/'
LOGS = ROOT + 'logs/'
DATA = ROOT + 'data/'
SENSE_DIR = LOGS + 'sense_assignments_lemmed/'
S3_S2ORC_BUCKET = 'ai2-s2-s2orc'
S3_S2ORC_PREFIX = '20200705v1/full/metadata/'
METADATA = '/net/nfs.cirrascale/s2-research/lucyl/full_metadata/'

def get_paper_year(): 
    time_fos_df = pd.read_parquet(LOGS + 'citing_papers/')
    paper_year = dict(zip(time_fos_df.paper_id, time_fos_df.year))
    
    return paper_year

def process_batch(batch): 
    start = time.time()
    short_name = batch['short_name']
    s3_url = batch['s3_infile']
    download_target_dir = batch['local_folder']
    download_target_path = batch['local_folder'] + short_name + '.jsonl.gz'
    gunzip_target_path = download_target_path.replace('.gz', '')
    outfolder = batch['citation_count_folder']
    papers_to_keep = batch['papers_to_keep']
    paper_year = batch['paper_year']

    # download
    download = False 
    if not os.path.exists(download_target_path): 
        download = True
        args = ['/home/lucyl/bin/aws', 's3', 'cp', s3_url, f'{download_target_dir}']
        subprocess.run(args)

    args = ['gunzip', '-f', download_target_path]
    subprocess.run(args)

    file_length = sum(1 for line in open(gunzip_target_path, 'r'))
    tokenizer = BasicTokenizer(do_lower_case=True)
    
    result_d = { 
        'paper id': [], 
        'citations': [], # within five years after publication
        'year bin': [], 
        'abstract length': [], 
        'num authors': [], 
        'num refs': [], 
    }
    
    with open(gunzip_target_path, 'r') as infile: 
        for line in tqdm(infile, total=file_length): 
            d = json.loads(line)
            if not d['paper_id'] in papers_to_keep: continue
            if not d['has_inbound_citations']: continue
                
            curr_year = d['year']
            tokens = tokenizer.tokenize(d['abstract'])
            # exclude words that don't contain letters (e.g. punctuation, numbers)
            tokens = [t for t in tokens if re.search('[a-z]', t)]
            
            time_range = set(range(curr_year, curr_year + 6))
            inbound_cites = []
            for other_id in d['inbound_citations']: 
                if other_id in paper_year and paper_year[other_id] in time_range: 
                    inbound_cites.append(other_id)
                
            result_d['abstract length'].append(len(tokens))
            result_d['paper id'].append(d['paper_id'])
            if curr_year < 2005: 
                result_d['year bin'].append(0)
            elif curr_year < 2010: 
                result_d['year bin'].append(1)
            else: 
                result_d['year bin'].append(2)
            result_d['num authors'].append(len(d['authors']))
            result_d['num refs'].append(len(d['outbound_citations']))
            result_d['citations'].append(inbound_cites)
    
    df = pd.DataFrame.from_dict(result_d)
    df.to_parquet(outfolder + 
                  batch['short_name'] + '.parquet') 
    
    end = time.time()
    os.remove(gunzip_target_path)

def get_five_year_citations(papers_to_regress):
    citation_count_folder = LOGS + 'regression/some_variables/'
    os.makedirs(citation_count_folder, exist_ok=True)
    
    paper_year = get_paper_year() # { paper_id : year } 
    
    s3 = boto3.resource('s3')
    bucket = s3.Bucket(S3_S2ORC_BUCKET)
    filenames = []
    for obj in bucket.objects.filter(Prefix=S3_S2ORC_PREFIX + 'metadata'): 
        filenames.append(obj.key)
        
    print("making batches...")
    batches = [{
        'short_name': filename.split('/')[-1].replace('.jsonl.gz', ''), 
        'filename': filename,
        'local_folder': METADATA,
        'papers_to_keep': papers_to_regress, 
        'paper_year': paper_year, 
        's3_infile': f's3://ai2-s2-s2orc/{filename}',
        'citation_count_folder': citation_count_folder,
    } for filename in sorted(filenames)]
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 3) as p:
        p.map(process_batch, batches)

def main(): 
    with open(LOGS + 'regression/papers_to_regress.json', 'r') as infile: 
        papers_to_regress_dict = json.load(infile)
    papers_to_regress = set()
    for fos in papers_to_regress_dict: 
        for paper_dict in papers_to_regress_dict[fos]: 
            papers_to_regress.add(paper_dict['paper_id'])
        
    get_five_year_citations(papers_to_regress)

if __name__ == '__main__':
    main()