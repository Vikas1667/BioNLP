import copy

source_file='TaggerOne-0.2.1/data/BC5CDR/CDR.2.PubTator'

def pmid_extractor():
    all_pmids = []
    pmids = []

    with open(source_file, 'r') as f:
        for line in f:
            if '|t|' in line:
                l = line.strip().split('|')
                pmid = l[0]
                pmids.append(pmid)
                all_pmids.append(pmid)

    print(all_pmids)


def abstract_title_extraction():
    entire_file = ''
    one_title_and_abst = ''

    with open(source_file, 'r') as f:
        for line in f:
            if line.strip() == "":
                entire_file += one_title_and_abst + '\n'
                one_title_and_abst = copy.copy('')
            else:
                if '|t|' in line:
                    one_title_and_abst += line
                elif '|a|' in line:
                    one_title_and_abst += line
                else:
                    if 'CID' in line:
                        continue
                    if line.strip().split('\t')[5] == '-1':
                        continue
                    if len(line.strip().split('\t')[5].split('|')) != 1:
                        continue

                    one_title_and_abst += line
        print(entire_file)
                    # print(one_title_and_abst)


if __name__ == '__main__':
    abstract_title_extraction()
