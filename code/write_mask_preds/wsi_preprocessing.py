"""
Output csv file format: 
sentence_ID,tokens,s2orc_id
"""
import os
import json
from collections import defaultdict, Counter
import time
import multiprocessing
import boto3
from tqdm import tqdm
import subprocess
import pandas as pd
import random
from nltk.tokenize import sent_tokenize
import numpy as np
import re
import csv
import argparse

#ROOT = '/net/nfs2.s2-research/lucyl/language-map-of-science/'
ROOT = '/home/lucyl/language-map-of-science/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
S3_S2ORC_BUCKET = 'ai2-s2-s2orc'
S3_S2ORC_PREFIX = '20200705v1/full/metadata/'
METADATA = '/net/nfs.cirrascale/s2-research/lucyl/full_metadata/'
#METADATA = '/net/nfs2.s2-research/lucyl/full_metadata/'

def get_journal_venue(paper_id, journal, venue, papers_to_keep): 
    '''
    This must match General Dataset Statistics steps for standardizing journal/venue
    '''
    if paper_id not in papers_to_keep: 
        return None

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
    random.seed(0)
    start = time.time()
    short_name = batch['short_name']
    
    # big file paths
    s3_url = batch['s3_infile']
    download_target_dir = batch['local_folder']
    download_target_path = batch['local_folder'] + short_name + '.jsonl.gz'
    gunzip_target_path = download_target_path.replace('.gz', '')
    
    # output paths
    outpath = batch['out_folder'] + short_name + '.csv'
    logpath = batch['log_folder'] + short_name + '.json'
    
    # misc
    papers_to_keep = batch['papers_to_keep']

    download = False 
    if not os.path.exists(download_target_path): 
        download = True
        args = ['/home/lucyl/bin/aws', 's3', 'cp', s3_url, f'{download_target_dir}']
        subprocess.run(args)

    args = ['gunzip', '-f', download_target_path]
    subprocess.run(args)
    
    journal_count = Counter() # { journal : number of abstracts included }
    sent_ID = 0
    max_len = 512
    outfile = open(outpath, 'w')
    writer = csv.writer(outfile)
    with open(gunzip_target_path, 'r') as infile: 
        for line in tqdm(infile): 
            d = json.loads(line)
            journal = d['journal']
            venue = d['venue']
            paper_id = d['paper_id']
            journal = get_journal_venue(paper_id, journal, venue, papers_to_keep)
            if not journal: 
                continue
            
            journal_count[journal] += 1
            title = d['title']
            abstract = d['abstract']
            title_abs = title + '\n\n' + abstract 
            # simple truncating for now to save space, the real truncating will happen later
            sent = ' '.join(title_abs.split(' ')[:max_len])
            writer.writerow([sent_ID, sent, paper_id])
            sent_ID += 1

    outfile.close()

    end = time.time()
    os.remove(gunzip_target_path)
    
    with open(logpath, 'w') as outfile: 
        json.dump(journal_count, outfile)

def write_sense_input(): 
    '''
    Most of this function matches the same preprocessing / filtering as word_counts.py
    '''
    out_folder = DATA + 'sense_input/actual_data/'
    log_folder = LOGS + 'sense_input_logs/'

    os.makedirs(log_folder, exist_ok=True)
    os.makedirs(out_folder, exist_ok=True)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(S3_S2ORC_BUCKET)
    filenames = []
    for obj in bucket.objects.filter(Prefix=S3_S2ORC_PREFIX + 'metadata_'): 
        filenames.append(obj.key)
        
    papers_to_keep = set()
    with open(DATA + 'input_paper_ids/fos_analysis.txt', 'r') as infile: 
        for line in infile: 
            papers_to_keep.add(line.strip())
    with open(DATA + 'input_paper_ids/journal_analysis.txt', 'r') as infile: 
        for line in infile: 
            papers_to_keep.add(line.strip())

    print("making batches...")
    batches = [{
        'short_name': filename.split('/')[-1].replace('.jsonl.gz', ''), 
        's3_infile': f's3://ai2-s2-s2orc/{filename}',
        'local_folder': METADATA,
        'out_folder': out_folder,
        'log_folder': log_folder,
        'papers_to_keep': papers_to_keep,
    } for filename in sorted(filenames)]

    
    print("Batches to do:", len(batches))
   
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 3) as p:
        p.map(process_batch, batches)

    print("done") 
    
def get_paper_id_to_sent(): 
    '''
    This is saved in the inverted_index folder 
    since it keeps track of sentences to papers. 
    '''
    out_folder = DATA + 'sense_input/actual_data/'
    mapping = {} # {sent_ID : s2orcID}
    for filename in os.listdir(out_folder): 
        with open(out_folder + filename, 'r') as infile: 
            split = filename.replace('.csv', '')
            reader = csv.reader(infile)
            for row in reader: 
                sent_ID = row[0]
                s2orc_ID = row[2]
                mapping[split + '_' + str(sent_ID)] = s2orc_ID
                
    os.makedirs(LOGS + 'inverted_index/', exist_ok=True)
    with open(LOGS + 'inverted_index/SKIP_paperID_sent.json', 'w') as outfile: 
        json.dump(mapping, outfile)

def main(): 
    write_sense_input()
    get_paper_id_to_sent()

if __name__ == '__main__':
    main()
