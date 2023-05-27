"""
Run language ID on the dataset
"""
import os
import json
from collections import defaultdict, Counter
import time
import multiprocessing
import boto3
from tqdm import tqdm
import langid
import subprocess
from helper import keep_abstract

ROOT = '/home/lucyl/language-map-of-science/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'
S3_S2ORC_BUCKET = 'ai2-s2-s2orc'
S3_S2ORC_PREFIX = '20200705v1/full/metadata/'
METADATA = '/net/nfs.cirrascale/s2-research/lucyl/full_metadata/'

def process_batch(batch): 
    '''
    Process a single metadata jsonl.gz
    '''
    start = time.time()
    short_name = batch['short_name']
    s3_url = batch['s3_infile']
    download_target_dir = batch['local_folder']
    download_target_path = batch['local_folder'] + short_name + '.jsonl.gz'
    gunzip_target_path = download_target_path.replace('.gz', '')
    outpath = batch['out_folder'] + short_name + '.jsonl'
    logpath = batch['log_folder'] + short_name + '.log'
    not_keep_set = set()

    log_out = open(logpath, 'w')

    # download
    download = False 
    if not os.path.exists(download_target_path): 
        download = True
        log_out.write("downloading from s3 aws...\n")
        args = ['/home/lucyl/bin/aws', 's3', 'cp', s3_url, f'{download_target_dir}/']
        subprocess.run(args)
    else: 
        log_out.write("gz file already downloaded.\n")
    log_out.flush()

    args = ['gunzip', '-f', download_target_path]
    subprocess.run(args)
    log_out.write("done unzipping")
    log_out.flush()

    file_length = sum(1 for line in open(gunzip_target_path, 'r'))

    journal_languages = defaultdict(Counter) # {journal : {language: count}}
    with open(gunzip_target_path, 'r') as infile: 
        for line in tqdm(infile, total=file_length): 
            d = json.loads(line)
            abstract = d['abstract']
            title = d['title'] 
            journal = d['journal']
            year = d['year']
            venue = d['venue']
            journal = keep_abstract(year, abstract, title, journal, venue, not_keep_set)
            if not journal: 
                continue
            
            title_abs = title + '\n\n' + abstract 
            lang = langid.classify(title_abs)[0]
            journal_languages[journal][lang] += 1

    with open(outpath, 'w') as outfile: 
        json.dump(journal_languages, outfile)

    end = time.time()
    log_out.write("done with reading and dump in " + str(end-start) + " seconds.\n")
    log_out.flush()
    os.remove(gunzip_target_path)

    log_out.close()

def get_languages(): 
    os.makedirs(LOGS + 'language_id_logs', exist_ok=True)
    os.makedirs(LOGS + 'language_id', exist_ok=True)

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(S3_S2ORC_BUCKET)
    filenames = []
    for obj in bucket.objects.filter(Prefix=S3_S2ORC_PREFIX + 'metadata_'): 
        filenames.append(obj.key)

    print("making batches...")
    batches = [{
        'short_name': filename.split('/')[-1].replace('.jsonl.gz', ''), 
        's3_infile': f's3://ai2-s2-s2orc/{filename}',
        'local_folder': METADATA,
        'out_folder': LOGS + 'language_id/',
        'log_folder': LOGS + 'language_id_logs/',
    } for filename in sorted(filenames)]

    remaining_batches = [batch for batch in batches if not os.path.exists(batch['out_folder'] + batch['short_name'] + '.jsonl')]
    print("Batches to do:", len(remaining_batches))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 3) as p:
        p.map(process_batch, remaining_batches)

    print("done") 

def save_non_english_journals(cutoff=0.8): 
    print("Saving for cutoff", cutoff)
    all_journals_lang = defaultdict(Counter)
    for filename in os.listdir(LOGS + 'language_id'): 
        json_path = LOGS + 'language_id/' + filename
        with open(json_path, 'r') as infile: 
            d = json.load(infile) 
            for j in d: 
                all_journals_lang[j] += d[j]
    with open(LOGS + 'non_english_journals_' + str(int(cutoff*100)) + '.txt', 'w') as outfile: 
        for j in all_journals_lang: 
            total = sum(all_journals_lang[j].values())
            frac = all_journals_lang[j]['en'] / total
            if frac <= cutoff: 
                outfile.write(j + '\t' + str(frac) + '\n')
        
def main(): 
    get_languages()
    save_non_english_journals(cutoff=0.8)

if __name__ == '__main__':
    main()
