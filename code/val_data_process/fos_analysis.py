"""
This file does the following: 
1) calculates level 0 agreement between s2 fos and mag fos for each paper
2) gets mapping from fos to s2orc ID

Command if you want to filter a subset of mag_plus_s2fos to only some mag_ids, such as those belonging to medicine: 
cat mag_plus_s2fos.csv | csv2tsv | tsv-join -f mag_id_medicine.txt -k 1 -d 1 > filtered_med_fos_level-0-and-1.tsv
"""
from tqdm import tqdm
from collections import Counter
import csv
from collections import defaultdict
import os
import json

ROOT = '/home/lucyl/language-map-of-science/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'

def fos_agreement(): 
    '''
    Outputs how many MAG level 0 FOS not in S2 FOS list, depending
    on whether it's non-empty or just in general 
    
    Also outputs common disagreements. 
    
    Also calculates precision and recall for each MAG level 0 FOS. 
    tp: MAG FOS matches S2 FOS
    fp: FOS is in S2 FOS but not in MAG 
    fn: FOS is in MAG FOS but not in S2 FOS
    
    mag_plus_s2fos.csv is formatted to include: MAG paper ID, field of study ID, 
    score, level, MAG FOS, S2 FOS. 
    
    This function did not end up being used to create any result in the final paper, but
    keeping it in case it's handy for others comparing MAG FOS and S2 FOS. 
    '''
    # only look at level 0 
    num_lines = 477054453 # obtained using wc -l
    total = 0
    recalled = 0
    recalled_nonempty = 0
    missing_s2 = 0
    disagreement_count = Counter()
    label_pr = defaultdict(Counter) # { FOS: {'tp': #, 'fp': #, 'fn': #}} 
    with open(DATA + 'mag_plus_s2fos.csv', 'r') as infile: 
        reader = csv.DictReader(infile)
        for row in tqdm(reader, total=num_lines): 
            if row['level'] == 0 or row['level'] == '0': 
                mag_label = row['displayname'].lower()
                s2orc_label = row['s2fieldsofstudy'].lower()
                s2_label_list = s2orc_label.split(',')
                if mag_label in s2_label_list: 
                    recalled += 1
                    label_pr[mag_label]['tp'] += 1
                else: 
                    # disagreement 
                    disagreement_count[mag_label + '-' + s2orc_label] += 1
                    label_pr[mag_label]['fn'] += 1
                if s2orc_label != '': 
                    if mag_label in s2_label_list: 
                        recalled_nonempty += 1
                    else: 
                        for s2_label in s2_label_list: 
                            label_pr[s2_label]['fp'] += 1
                else: 
                    missing_s2 += 1
                total += 1

    outfile = open(LOGS + 'wiktionary/fos_agreement.txt', 'w') 
    for label in label_pr: 
        if 'tp' in label_pr[label]: # is a MAG FOS 
            precision = round(label_pr[label]['tp'] / (label_pr[label]['tp'] + label_pr[label]['fp']), 5)
            recall = round(label_pr[label]['tp'] / (label_pr[label]['tp'] + label_pr[label]['fn']), 5)
            outfile.write('* ' + label.upper() + ': Precision = ' + str(precision) + ', Recall = ' + str(recall) + '\n')
    outfile.write('# MAG level 0 FOS is in S2 FOS list: ' + str(recalled / total) + '\n') 
    outfile.write('# S2 FOS list is empty: ' + str(missing_s2 / total) + '\n')
    outfile.write('# MAG level 0 FOS is in non-empty S2 FOS list: ' + str(recalled_nonempty / total) + '\n') 
    # common disagreements
    for tup in disagreement_count.most_common(): 
        outfile.write(tup[0] + ',' + str(tup[1]) + '\n')
    
    outfile.close()
    
def create_mag_mapping(): 
    '''
    fos_level-0-and-1.csv is a file that contains
    MAG Paper ID, FOD ID, score, level (1 or 0), and FOS name. 
    '''
    mag_fos = defaultdict(list)
    num_lines = 450723557 # obtained using wc -l
    with open(DATA + 'fos_level-0-and-1.csv', 'r') as infile: 
        reader = csv.reader(infile)
        for row in tqdm(reader, total=num_lines): 
            magID = row[0]
            fos = row[4].lower()
            mag_fos[magID].append(fos)
                
    os.makedirs(LOGS + 'wiktionary/', exist_ok=True)
    with open(LOGS + 'wiktionary/mag_fos.json', 'w') as outfile:
        json.dump(mag_fos, outfile)

def main(): 
    #fos_agreement()
    create_mag_mapping()

if __name__ == '__main__':
    main()