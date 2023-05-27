import requests
import json
import wikitextparser as wtp
from tqdm import tqdm

ROOT = '/home/lucyl/language-map-of-science/'
DATA = ROOT + 'data/'
LOGS = ROOT + 'logs/'

def scrape_wiktionary(): 
    '''
    For ~9k words this takes around 40 minutes. 
    '''
    vocab = set()
    with open(LOGS + 'sense_vocab/wsi_vocab_set_98_50.txt', 'r') as infile: 
        for line in infile: 
            vocab.add(line.strip())

    definitions = {}
    with open(LOGS + 'wiktionary.log', 'w') as outfile: 
        for w in tqdm(vocab): 
            wiki_title = w
            defs = []
            url = 'https://en.wiktionary.org/w/api.php?action=parse&page=' + wiki_title + '&prop=wikitext&formatversion=2&format=json'
            response = requests.get(url)
            if not response.ok: 
                outfile.write("Problem with " + wiki_title + "\n")
            response_dict = json.loads(response.text)
            if 'error' in response_dict: 
                outfile.write("Problem with " + wiki_title + "\n")
                outfile.write(json.dumps(response_dict) + '\n')
            else: 
                text = response_dict['parse']['wikitext']
                lines = text.split('\n')
                curr_pos = ''
                start = False
                for line in lines: 
                    if line == '==English==': 
                        start = True
                    elif (line.startswith('==') or line.startswith('=')) and not line.startswith('==='): 
                        break
                    if line.startswith('===') and start: 
                        curr_pos = line.replace('=', '').strip()
                    if start and 'Noun' in curr_pos and 'en-plural noun' in line: 
                        curr_pos += '_plural'
                    if start and line.startswith('# '): 
                        defs.append((curr_pos,line))
            if defs: 
                definitions[w] = defs
        outfile.write("Found " + str(len(definitions)*100 / len(vocab)) + "% of vocab words.\n")
    with open(DATA + 'validation/wiktionary_definitions.json', 'w') as outfile: 
        json.dump(definitions, outfile)
            
def main(): 
    scrape_wiktionary()

if __name__ == '__main__':
    main()