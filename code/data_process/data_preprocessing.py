'''
This contains functions for helping with
data organization or counting.
'''
import os
from collections import Counter, defaultdict
import json
import time
import multiprocessing
import boto3
from tqdm import tqdm
import subprocess
import pandas as pd

ROOT = '/home/lucyl/language-map-of-science/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
S3_S2ORC_BUCKET = 'ai2-s2-s2orc'
S3_S2ORC_PREFIX = '20200705v1/full/metadata/'
METADATA = '/net/nfs.cirrascale/s2-research/lucyl/full_metadata/'

def process_batch(batch): 
    start = time.time()
    short_name = batch['short_name']
    s3_url = batch['s3_infile']
    download_target_dir = batch['local_folder']
    download_target_path = batch['local_folder'] + short_name + '.jsonl.gz'
    gunzip_target_path = download_target_path.replace('.gz', '')
    outfolder = batch['out_folder']
    logpath = batch['log_folder'] + short_name + '.log'
    not_keep_set = batch['not_keep']

    log_out = open(logpath, 'w')

    # download
    download = False 
    if not os.path.exists(download_target_path): 
        download = True
        log_out.write("downloading from s3 aws...\n")
        args = ['/home/lucyl/bin/aws', 's3', 'cp', s3_url, f'{download_target_dir}']
        subprocess.run(args)
    else: 
        log_out.write("gz file already downloaded.\n")
    log_out.flush()

    args = ['gunzip', '-f', download_target_path]
    subprocess.run(args)
    log_out.write("done unzipping")
    log_out.flush()

    file_length = sum(1 for line in open(gunzip_target_path, 'r'))

    journal_counts = Counter() # {journal : count}
    abstract_wc = Counter() # {journal : abstract white-space token count}
    journal_abs_c = Counter() # { journal : number of abstracts }
    year_count = Counter() # { year : count }  
    contempt_count = Counter() # { year : count } 
    removal_count = Counter() # {reason : count}
    
    result_d = { # for figuring out dataset and stratified sampling
        'paper id': [], 
        'journal/venue': [], 
        'valid': [], # true or false
        'fos': [], # comma delimited strings
    }
    
    with open(gunzip_target_path, 'r') as infile: 
        for line in tqdm(infile, total=file_length): 
            d = json.loads(line)
            year = d['year']
            year_count[year] += 1
            
            # contemporary papers only
            if year is None or year == 'null' or year < 2000 or year >= 2020:
                removal_count['time'] += 1
                continue

            abstract = d['abstract']
            title = d['title'] 
            journal = d['journal'] 
            venue = d['venue']
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
                
            publication = journal + ' -#- ' + venue # consolidate journal & venue

            journal_counts[publication] += 1
            
            mag_id = d['mag_id']
            if mag_id in mag_fos: 
                fos = mag_fos[mag_id]
            else: 
                fos = ['none']
            
            result_d['journal/venue'].append(publication)
            result_d['fos'].append(','.join(fos)) 
            result_d['paper id'].append(d['paper_id']) 
            
            # publication needs to be English
            if journal in not_keep_set or venue in not_keep_set: 
                result_d['valid'].append(False)
                removal_count['non_english'] += 1
                continue
            # abstract, title, and publication need to be valid
            if abstract is None or title is None or publication == ' -#- ': 
                result_d['valid'].append(False)
                removal_count['bad_metadata'] += 1
                continue
            result_d['valid'].append(True)
            
            journal_abs_c[publication] += 1
            num_tokens = len(title.split()) + len(abstract.split())
            abstract_wc[publication] += num_tokens
            contempt_count[year] += 1

    with open(outfolder + batch['short_name'] + '_journal_counts.json', 'w') as outfile: 
        json.dump(journal_counts, outfile)
    with open(outfolder + batch['short_name'] + '_journal_abs_word_count.json', 'w') as outfile: 
        json.dump(abstract_wc, outfile)
    with open(outfolder + batch['short_name'] + '_journal_abs_count.json', 'w') as outfile: 
        json.dump(journal_abs_c, outfile)
    with open(outfolder + batch['short_name'] + '_year_count.json', 'w') as outfile: 
        json.dump(year_count, outfile)
    with open(outfolder + batch['short_name'] + '_contempt_count.json', 'w') as outfile: 
        json.dump(contempt_count, outfile)
    with open(outfolder + batch['short_name'] + '_removal_count.json', 'w') as outfile: 
        json.dump(removal_count, outfile)
        
    df = pd.DataFrame.from_dict(result_d)
    df.to_parquet(outfolder + 'result_parquets/' + \
                  batch['short_name'] + '_paper_labels.parquet') 

    end = time.time()
    log_out.write("done with reading and dump in " + str(end-start) + " seconds.\n")
    log_out.flush()
    os.remove(gunzip_target_path)

    log_out.close()

def count_abstracts(): 
    os.makedirs(LOGS + 'metadata_counts_logs/', exist_ok=True)
    os.makedirs(LOGS + 'metadata_counts/', exist_ok=True)
    os.makedirs(LOGS + 'metadata_counts/result_parquets/', exist_ok=True)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(S3_S2ORC_BUCKET)
    filenames = []
    for obj in bucket.objects.filter(Prefix=S3_S2ORC_PREFIX + 'metadata'): 
        filenames.append(obj.key)
        
    not_keep_journals = set()
    with open(LOGS + 'non_english_journals_80.txt', 'r') as infile: 
        for line in infile: 
            journal = line.strip().split('\t')[0]
            not_keep_journals.add(journal)

    print("making batches...")
    batches = [{
        'short_name': filename.split('/')[-1].replace('.jsonl.gz', ''), 
        's3_infile': f's3://ai2-s2-s2orc/{filename}',
        'local_folder': METADATA,
        'not_keep': not_keep_journals,
        'out_folder': LOGS + 'metadata_counts/',
        'log_folder': LOGS + 'metadata_counts_logs/',
    } for filename in sorted(filenames)]

    remaining_batches = [batch for batch in batches]
    print("Batches to do:", len(remaining_batches))
   
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 3) as p:
        p.map(process_batch, remaining_batches)

    print("done") 

def aggregate_counts(): 
    in_folder = LOGS + 'metadata_counts/'

    all_journal_c = Counter()
    all_abstract_wc = Counter()
    all_journal_abs_c = Counter()
    all_year_count = Counter()
    all_contempt_count = Counter()
    all_removal_count = Counter()

    for i in range(100): 
        short_name = 'metadata_' + str(i)
        with open(in_folder + short_name + '_journal_counts.json', 'r') as infile: 
            journal_counts = json.load(infile)
        all_journal_c += Counter(journal_counts)
        with open(in_folder + short_name + '_journal_abs_word_count.json', 'r') as infile: 
            abstract_wc = json.load(infile)
        all_abstract_wc += Counter(abstract_wc)
        with open(in_folder + short_name + '_journal_abs_count.json', 'r') as infile: 
            journal_abs_c = json.load(infile)
        all_journal_abs_c += Counter(journal_abs_c)
        with open(in_folder + short_name + '_year_count.json', 'r') as infile: 
            year_count = json.load(infile)
        all_year_count += Counter(year_count)
        with open(in_folder + short_name + '_contempt_count.json', 'r') as infile: 
            contempt_count = json.load(infile)
        all_contempt_count += Counter(contempt_count)
        with open(in_folder + short_name + '_removal_count.json', 'r') as infile: 
            removal_count = json.load(infile)
        all_removal_count += Counter(removal_count)
    
    os.makedirs(LOGS + 'data_counts/', exist_ok=True)
    with open(LOGS + 'data_counts/journal_counts.json', 'w') as outfile: 
        json.dump(all_journal_c, outfile)
    with open(LOGS + 'data_counts/journal_abs_word_count.json', 'w') as outfile: 
        json.dump(all_abstract_wc, outfile)
    with open(LOGS + 'data_counts/journal_abs_count.json', 'w') as outfile: 
        json.dump(all_journal_abs_c, outfile)
    with open(LOGS + 'data_counts/year_count.json', 'w') as outfile: 
        json.dump(all_year_count, outfile)
    with open(LOGS + 'data_counts/contempt_count.json', 'w') as outfile: 
        json.dump(all_contempt_count, outfile)
    with open(LOGS + 'data_counts/removal_count.json', 'w') as outfile: 
        json.dump(all_removal_count, outfile)


def main(): 
    count_abstracts()
    aggregate_counts()

if __name__ == '__main__':
    print("Loading mag ID to FOS...")
    with open(LOGS + 'wiktionary/mag_fos.json', 'r') as infile: 
        mag_fos = json.load(infile)
    print("Done loading.")
    main()

