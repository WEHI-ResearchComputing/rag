import argparse

from Bio import Entrez

def main(xml, search_term, retmax):
    Entrez.email = 'an.email@mail.com'
    retstart = 1
    cnt = 0
    while retstart < retmax:
        handle = Entrez.esearch(db='pubmed', term=search_term, retmax=retmax, retstart=retstart)
        record = Entrez.read(handle)
        record_ids = record['IdList']
        n_recs = len(record_ids)
        cnt += n_recs
        if n_recs == 0: break
        
        print(f'downloading {n_recs} starting at {retstart} ', end='')
        stream = Entrez.efetch(db='pubmed', id=record_ids, retmode='xml')
        buf_len = 1024*1024
        while True:
            print('.', end='')
            buf = stream.read(buf_len)
            xml.write(buf)
            if len(buf) < buf_len: break
        print(' done')
        stream.close()
        retstart += len(record['IdList'])
        if retstart >= 9999:
            print('\nMax download limit reached')
            break
        
    print(f'Downloaded {cnt} records')
    
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-records", 
                        default=1000,
                        type=int, 
                        help="The maximum number of records to return. This tool will not return more that 9998.")
    parser.add_argument("--search-term", 
                        type=str, 
                        required=True,
                        help="The Pubmed search term.")
    parser.add_argument("--output-path", 
                        type=str, 
                        required=True,
                        help="The output file path.")
    return parser.parse_args()

if __name__ == '__main__':
    args = arg_parse()
    with open(args.output_path, 'wb') as xml:
        main(xml, args.search_term, args.max_records)