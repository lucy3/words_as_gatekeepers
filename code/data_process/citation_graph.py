'''
To use graph-tool, create and activate an environment, which I call "gt": 
https://git.skewed.de/count0/graph-tool/-/wikis/installation-instructions#conda

conda activate gt

In this graph, each node is a paper, with metadata that includes
paper_id and journal/venue name. Edges are directed citations. 
You can modify this script to include other metadata for nodes
and edges, but it provides a starting point for a graph if needed. 
'''
import os
from collections import Counter, defaultdict
import json
import time
import multiprocessing
import boto3
from tqdm import tqdm
import subprocess
from graph_tool.all import Graph, load_graph
from graph_tool import stats
import numpy as np
import pandas as pd
from helper import keep_abstract

ROOT = '/data0/lucy/language-map-of-science/'
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
        args = ['aws', 's3', 'cp', s3_url, f'{download_target_dir}']
        subprocess.run(args)
    else: 
        log_out.write("gz file already downloaded.\n")
    log_out.flush()

    args = ['gunzip', '-f', download_target_path]
    subprocess.run(args)
    log_out.write("done unzipping")
    log_out.flush()

    file_length = sum(1 for line in open(gunzip_target_path, 'r'))

    # number of papers with citations of each type 
    inbound_journal_counts = Counter() # {journal : count}
    outbound_journal_counts = Counter() # {journal : count}
    
    citation_df_dict = {
        'source': [], # paper_id str
        'dest': [], # paper_id str
        'type': [], # inbound (1) or outbound (0)
    }
    
    # here the only metadata we save attached to each paper is aggregated journal/venue info
    # however you can also save dictionaries of other metadata that you may need for
    # your purposes. The more metadata included the "heavier" the resulting graph will later get
    journal_paper_ids = defaultdict(list)
    
    with open(gunzip_target_path, 'r') as infile: 
        for line in tqdm(infile, total=file_length): 
            d = json.loads(line)
            abstract = d['abstract']
            title = d['title'] 
            journal = d['journal']
            year = d['year']
            venue = d['venue']
            # this is where the filtering/preprocessing happens
            # you can change this based on your purposes
            journal = keep_abstract(year, abstract, title, journal, venue, not_keep_set)
            if not journal: 
                continue
            
            if d['has_inbound_citations']: 
                inbound_journal_counts[journal] += 1
            if d['has_outbound_citations']: 
                outbound_journal_counts[journal] += 1
            paper_id = d['paper_id']
            journal_paper_ids[journal].append(paper_id)
            
            for other_id in d['outbound_citations']: 
                citation_df_dict['source'].append(paper_id)
                citation_df_dict['dest'].append(other_id)
                citation_df_dict['type'].append(0)
            
            for other_id in d['inbound_citations']: 
                citation_df_dict['source'].append(paper_id)
                citation_df_dict['dest'].append(other_id)
                citation_df_dict['type'].append(1)

    with open(outfolder + batch['short_name'] + '_inbound_journal_counts.json', 'w') as outfile: 
        json.dump(inbound_journal_counts, outfile)
    with open(outfolder + batch['short_name'] + '_outbound_journal_counts.json', 'w') as outfile: 
        json.dump(outbound_journal_counts, outfile)
        
    with open(outfolder + batch['short_name'] + '_journal_paper_ids.json', 'w') as outfile: 
        json.dump(journal_paper_ids, outfile)
        
    df = pd.DataFrame.from_dict(citation_df_dict)
    df.to_parquet(outfolder + batch['short_name'] + '_edges.parquet')
        
    end = time.time()
    log_out.write("done with reading and dump in " + str(end-start) + " seconds.\n")
    log_out.flush()
    os.remove(gunzip_target_path)

    log_out.close()

def count_citations(): 
    os.makedirs(LOGS + 'citation_counts_logs/', exist_ok=True)
    os.makedirs(LOGS + 'citation_counts/', exist_ok=True)

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
        'out_folder': LOGS + 'citation_counts/',
        'log_folder': LOGS + 'citation_counts_logs/',
    } for filename in sorted(filenames)]

    remaining_batches = [batch for batch in batches] # can add condition here if I want
    print("Batches to do:", len(remaining_batches))
   
    with multiprocessing.Pool(processes=multiprocessing.cpu_count() // 5) as p:
        p.map(process_batch, remaining_batches)

    print("done") 
    
def aggregate_citations(): 
    '''
    Aggregates citations per journal across data splits into a graph-tool
    network. 
    
    One graph is an outbound graph and another is an inbound graph. 
    '''
    print("Aggregating citations...")
    citation_folder = LOGS + 'citation_counts/'
    graph_folder = LOGS + 'citation_graph/'
    os.makedirs(graph_folder, exist_ok=True)
    
    inbound_journal_counts = Counter() # {journal : count}
    outbound_journal_counts = Counter() # {journal : count}
    
    inbound_G = Graph(directed=False)
    inbound_G.vertex_properties["journal"] = inbound_G.new_vertex_property("string")
    inbound_G.vertex_properties["paper_id"] = inbound_G.new_vertex_property("string")
    # can also add other vertex properties / metadata too if needed using same syntax
    inbound_vertex_paper_ids = {} # { paper id : vertex id } 
    
    outbound_G = Graph(directed=False)
    outbound_G.vertex_properties["journal"] = outbound_G.new_vertex_property("string")
    outbound_G.vertex_properties["paper_id"] = outbound_G.new_vertex_property("string")
    outbound_vertex_paper_ids = {} # { paper id : vertex id } 
    
    for i in range(100): 
        short_name = 'metadata_' + str(i)
        print(short_name)
        with open(citation_folder + short_name + '_inbound_journal_counts.json', 'r') as infile: 
            inbound_counts = json.load(infile)
        inbound_journal_counts += inbound_counts
        with open(citation_folder + short_name + '_outbound_journal_counts.json', 'r') as infile: 
            outbound_counts = json.load(infile)
        outbound_journal_counts += outbound_counts

        with open(citation_folder + short_name + '_journal_paper_ids.json', 'r') as infile: 
            journal_paper_ids = json.load(infile)
            
        # populate the graph with nodes 
        for journal in journal_paper_ids: 
            for paper_id in journal_paper_ids[journal]: 
                if paper_id in outbound_vertex_paper_ids: 
                    # seen before, need to add journal info
                    v1 = outbound_G.vertex(outbound_vertex_paper_ids[paper_id])
                    outbound_G.vertex_properties["journal"][v1] = journal
                else: 
                    v1 = outbound_G.add_vertex()
                    idx = outbound_G.vertex_index[v1]
                    outbound_G.vertex_properties["paper_id"][v1] = paper_id
                    outbound_G.vertex_properties["journal"][v1] = journal
                    outbound_vertex_paper_ids[paper_id] = idx
                
                if paper_id in inbound_vertex_paper_ids: 
                    # seen before, need to add journal info
                    v1 = inbound_G.vertex(inbound_vertex_paper_ids[paper_id])
                    inbound_G.vertex_properties["journal"][v1] = journal
                else: 
                    v1 = inbound_G.add_vertex()
                    idx = inbound_G.vertex_index[v1]
                    inbound_G.vertex_properties["paper_id"][v1] = paper_id
                    inbound_G.vertex_properties["journal"][v1] = journal
                    inbound_vertex_paper_ids[paper_id] = idx
            
        edge_df = pd.read_parquet(citation_folder + short_name + '_edges.parquet')
        edge_df = edge_df.reset_index()  # make sure indexes pair with number of rows

        for index, row in tqdm(edge_df.iterrows(), total=edge_df.shape[0]):
            paper_id = row['source']
            other_id = row['dest']
            if row['type'] == 0: 
                # outbound graph edge
                assert paper_id in outbound_vertex_paper_ids
                v1 = outbound_G.vertex(outbound_vertex_paper_ids[paper_id])
                    
                if other_id in outbound_vertex_paper_ids: 
                    # seen before, retrieve
                    v2 = outbound_G.vertex(outbound_vertex_paper_ids[other_id])
                else: 
                    # some citing/cited papers do not have journal
                    # info because they were filtered out (e.g. non-English, not
                    # within time range)
                    v2 = outbound_G.add_vertex()
                    idx = outbound_G.vertex_index[v2]
                    outbound_G.vertex_properties["paper_id"][v2] = other_id
                    outbound_G.vertex_properties["journal"][v2] = ''
                    outbound_vertex_paper_ids[other_id] = idx 
                    
                outbound_G.add_edge(v1, v2)
                    
            if row['type'] == 1: 
                # inbound graph edge
                assert paper_id in inbound_vertex_paper_ids
                v1 = inbound_G.vertex(inbound_vertex_paper_ids[paper_id])
                    
                if other_id in inbound_vertex_paper_ids: 
                    # seen before, retrieve
                    v2 = inbound_G.vertex(inbound_vertex_paper_ids[other_id])
                else: 
                    v2 = inbound_G.add_vertex()
                    idx = inbound_G.vertex_index[v2]
                    inbound_G.vertex_properties["paper_id"][v2] = other_id
                    inbound_G.vertex_properties["journal"][v2] = ''
                    inbound_vertex_paper_ids[other_id] = idx  
                    
                inbound_G.add_edge(v1, v2)

    inbound_G.save(graph_folder + "ALL_inbound.gt")
    outbound_G.save(graph_folder + "ALL_outbound.gt")
    
    with open(graph_folder + 'ALL_inbound_journal_counts.json', 'w') as outfile: 
        json.dump(inbound_journal_counts, outfile)
        
    with open(graph_folder + 'ALL_outbound_journal_counts.json', 'w') as outfile: 
        json.dump(outbound_journal_counts, outfile)
        
def main(): 
    count_citations()
    aggregate_citations()

if __name__ == '__main__':
    main()
