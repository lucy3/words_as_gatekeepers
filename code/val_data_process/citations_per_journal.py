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

def get_journal_venue(paper_id, journal, venue): 
    '''
    The bottom half of this must match General Dataset 
    Statistics steps for standardizing journal/venue
    '''
    if journal and journal != 'null': 
        journal = journal.strip().lower() # case insensitive
        journal = ' '.join([i for i in journal.split(' ') if not any(char.isdigit() for char in i)]).strip()
    else: 
        journal = ''
    if venue and venue != 'null': 
        venue = venue.strip().lower()
        venue = ' '.join([i for i in venue.split(' ') if not any(char.isdigit() for char in i)]).strip()
    else: 
        venue = ''

    if journal == '' and venue == '': 
        return None
    elif journal == '': 
        new_k = venue # use venue
    elif venue == '': 
        new_k = journal # use journal
    elif journal == venue: 
        new_k = journal # both same
    elif journal != venue: 
        new_k = journal # use journal
    
    return new_k

def process_batch(batch): 
    start = time.time()
    short_name = batch['short_name']
    s3_url = batch['s3_infile']
    download_target_dir = batch['local_folder']
    download_target_path = batch['local_folder'] + short_name + '.jsonl.gz'
    gunzip_target_path = download_target_path.replace('.gz', '')
    outfolder = batch['citation_count_folder']
    journals_to_keep = batch['journals_to_keep']

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
    
    result_d = defaultdict(list) # {journal: [list of citation counts]}
    
    with open(gunzip_target_path, 'r') as infile: 
        for line in tqdm(infile, total=file_length): 
            d = json.loads(line) 
            journal = d['journal']
            venue = d['venue']
            paper_id = d['paper_id']
            journal = get_journal_venue(paper_id, journal, venue)
            if not journal or journal not in journals_to_keep: 
                continue
                
            citation_count = len(d['inbound_citations'])
            result_d[journal].append(citation_count)
    
    with open(outfolder + batch['short_name'] + '.json', 'w') as outfile:
        json.dump(result_d, outfile)
    
    end = time.time()
    os.remove(gunzip_target_path)

def get_citations_per_journal(journals):
    citation_count_folder = LOGS + 'regression/journal_citations/'
    os.makedirs(citation_count_folder, exist_ok=True)
    
    # create batches 
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
        'journals_to_keep': journals, 
        's3_infile': f's3://ai2-s2-s2orc/{filename}',
        'citation_count_folder': citation_count_folder,
    } for filename in sorted(filenames)]
    
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 3) as p:
        p.map(process_batch, batches)

def main(): 
    journal_df = pd.read_csv(DATA + 'input_paper_ids/journal_df.csv', index_col=0)
    journals_to_keep = set(journal_df['clean journal/venue'].unique())
    get_citations_per_journal(journals_to_keep)

if __name__ == '__main__':
    main()