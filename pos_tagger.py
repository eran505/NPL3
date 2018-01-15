

import pandas as pd
import os
#from nltk.tag.stanford import CoreNLPNERTagger
os.environ['STANFORD_MODELS'] = '/home/ise/NLP/stanfordNLP/stanford-ner-2017-06-09/classifiers'
os.environ['CLASSPATH'] = '/home/ise/NLP/stanfordNLP/stanford-ner-2017-06-09'
#from nltk.tag.stanford import CoreNLPPOSTagger

from nltk.tag import StanfordPOSTagger
from nltk import sent_tokenize, word_tokenize, pos_tag, ne_chunk
import re
from nltk import ngrams
from nltk.tree import Tree
import numpy as np
from nltk.parse.corenlp import CoreNLPDependencyParser
import nltk
from nltk.tag import StanfordNERTagger
from nltk.parse.stanford import StanfordParser
import matplotlib.pyplot as plt
from nltk.tag import StanfordNERTagger
from nltk.tag import StanfordNERTagger
#from nltk.tag.stanford import CoreNLPTagger
from nltk.tag import StanfordNERTagger
#from nltk.tag.stanford import CoreNLPPOSTagger, CoreNLPNERTagger
##start the server
##java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000

counter=1

class PN_System:
    def __init__(self, data_path='Res.tsv', data_columns=None):
        if data_columns is None:
            data_columns = ['year', 'president', 'text']
        self.path = data_path
        self.columns=data_columns
        self.top_k=None
        self.df=None
        self.ctr = 1
        self.top_k_grams=[]
        self.k=10
        self.ctr2 = 1
        self._read_data()
        self.read_pos_file()
    def read_pos_df(self):
        df = pd.read_csv('pos_ner.csv', sep='\t', names=['year', 'president', 'text','pos'], quoting=3)
        print df['pos'][:3]
    def _NER(self,txt,id):
        print id
        if str(id) == '194':
            return
        from nltk.tag.stanford import StanfordTagger
        #text = 'Rami Eid is studying at Stony Brook University in NY'
        #text_i='WASHINGTON -- In the wake of a string of abuses by New York police officers in the 1990s, Loretta E. Lynch, the top federal prosecutor in Brooklyn, spoke forcefully about the pain of a broken trust that African-Americans felt and said the responsibility for repairing generations of miscommunication and mistrust fell to law enforcement.'

        filepath='/home/ise/Desktop/prob.txt'
        with open(filepath, 'w') as file_handler:
            file_handler.write("{}\n".format(txt))


        tokens = nltk.word_tokenize(txt)
        path_to_jar = '/home/ise/NLP/stanfordNLP/stanford-ner-2017-06-09/'
        name_jar = 'stanford-ner.jar'
        path_to_model='/home/ise/NLP/stanfordNLP/stanford-ner-2017-06-09/classifiers/'
        name_model='english.all.3class.distsim.crf.ser.gz'
        model = path_to_model+name_model
        jar=path_to_jar+name_jar
        tagger = StanfordNERTagger(model, path_to_jar=jar, encoding='UTF-8')
        ne_tagged_sent =  tagger.tag(tokens)
        ne_tree = stanfordNE2tree(ne_tagged_sent)
        print ne_tree
        out = pars_tree(ne_tree)
        print "--"*55
        filepath='/home/ise/NLP/NLP3/ner/ner_{}.txt'.format(id)
        with open(filepath, 'w') as file_handler:
            for item in out:
                file_handler.write("{}\n".format(item))

        return out

    def read_pos_file(self):
        path = '/home/eran/NLP/NLP3/pos/'
        list_pos=[]
        arr_txt = [x for x in os.listdir(path) if x.endswith(".txt")]
        #print(arr_txt)
        list_bi_gram=[]
        list_tri_gram=[]
        print "size",len(arr_txt)
        for x in arr_txt:
            id_num=(x[4:-4])
            id_num= int(id_num)
            pos_seq=[]
            word_seq=[]
            d_pos={'ID':id_num}

            with open(path+x, 'r') as file_pos:
                ctr_pos = 0
                for line in file_pos:
                    #print line
                    if len(line)>1:
                        arr=line[1:-2].split(', ')
                        pos_i = arr[1][2:-1]
                        word_i = arr[0][2:-1]
                        pos_seq.append(pos_i)
                        word_seq.append(word_i)
                        if pos_i in d_pos:
                            d_pos[pos_i]+=1
                        else:
                            d_pos[pos_i]=1
                        ctr_pos+=1
                d_pos['All_POS']=ctr_pos
                biGram_d = self.freq_ngrams_dico(pos_seq,2)
                triGram_d = self.freq_ngrams_dico(pos_seq,3)
                triGram_d['ID'] = id_num
                biGram_d['ID'] = id_num
                list_bi_gram.append(biGram_d)
                list_tri_gram.append(triGram_d)
                list_pos.append(d_pos)
        df_uni = pd.DataFrame(list_pos)
        df_bi = pd.DataFrame(list_bi_gram)
        df_tri = pd.DataFrame(list_tri_gram)

        print self.time_stat(df_uni,'NN')#   ['NN','NNS','JJ'])

        exit(0)
        print "{} size: {}".format("uni",df_uni.shape)
        print "{} size: {}".format("bi", df_bi.shape)
        print "{} size: {}".format("tri", df_tri.shape)
        res  = self.get_statistics(df_uni)
        res  = self.get_statistics(df_bi)
        res  = self.get_statistics(df_tri)

        print "sorting..."

        new_d = {}
        sortedList = sorted(res.values())
        for sortedKey in sortedList:
            for key,val in res.items():
                if val == sortedKey:
                    new_d[key]=val

        result_df = pd.merge(self.df,df,on='ID')
        self.df = result_df.fillna(0)
        print "merge.df = ", self.df.shape
        print "done"


    def freq_ngrams_dico(self,arr_tokens,n):
        ctr = 0
        d={}
        token_grams = ngrams(arr_tokens,n)
        for gram in token_grams:
            if gram in d :
                d[gram]+=1
            else:
                d[gram]=1
            ctr=+1
        d['All_POS'] = ctr
        return d

    def get_statistic_by_name(self,df,name,arr_pos):
        result_df = pd.merge(self.df, df, on='ID')
        ans_dict = {}
        df_tmp = result_df.loc[result_df['president'] == name ]
        self.get_statistics(df_tmp)
        for pos in arr_pos:
            if pos in df_tmp:
                ans_dict[pos] = df_tmp[pos].sum()
        return ans_dict

    def get_statistics(self,df):
        d_stat={}
        self.top_k = [0]*self.k
        self.top_k_grams=[None]*self.k
        list_start = list(df)
        print "list_start",len(list_start)
        df = df.drop(['ID'],axis=1)
        df = df._get_numeric_data()
        list_col = list(df)
        print "list_col ", len(list_col )
        for col in list_col:
            sum_i = df[col].sum()
            d_stat[col]= sum_i
            self._manger_array_min(sum_i,col)
        print self.top_k_grams
        print "#"*200
        print self.top_k
        return d_stat

    def time_stat(self,df,pos):
        result_df = pd.merge(self.df, df, on='ID')
        result_df.groupby(['year'])
        print list(result_df)
        print "making plot.."
        result_df['norm_'+pos] = result_df[pos] / result_df['All_POS']
        result_df.plot('year','norm_'+pos,logy=True)
        plt.show()


    def _manger_array_min(self,num,gram):
        #print self.top_k_grams
        min = self.top_k[0]
        max = self.top_k[-1]
       # print "num:",num
        if num > max:
            tmp = self.top_k[1:]
            tmp.append(num)
            self.top_k = tmp
            self.top_k_grams.append([gram])
            self.top_k_grams = self.top_k_grams[1:]
        elif num >= min and max >= num :
            for x in range(self.k):
                if self.top_k[x] < num:
                    continue
                if self.top_k[x] == num:
                    self.top_k_grams[x].append(gram)
                    break
                else:
                    tmp = self.top_k[1:x]
                    tmp.append(num)
                    tmp = tmp + self.top_k[x:]
                    self.top_k = tmp
                    tmp  = self.top_k_grams[1:x]
                    tmp.append([gram])
                    tmp = tmp + self.top_k_grams[x:]
                    self.top_k_grams  =tmp
                    break

            #self.top_k_grams[x] = [gram]






    def _read_data(self):
        #TODO: problem with 194 sample
        df = pd.read_csv(self.path, sep='\t', names=self.columns, quoting=3)
        df.insert(0, 'ID', range(1, 1 + len(df)))
        print df.shape
        self.df=df
        print df['president'][:30]
        return
        self.df['ner'] = np.nan
        self.df['ner'] = df.apply(lambda x: self._NER(x['text'],x['ID']), axis=1)
        self.df['pos'] = np.nan
        self.df['pos'] =  df.apply(lambda x: self._POS(x['text'],x['ID']), axis=1)
        self.df.to_csv('pos_ner.csv')

    def _POS(self,txt,id):
        self.df[['ID','pos']].to_csv('pos_ner.csv',sep='\t')
        path_pos= '/home/ise/NLP/stanfordNLP/stanford-postagger-full-2017-06-09/stanford-postagger.jar'
        model_path = '/home/ise/NLP/stanfordNLP/stanford-postagger-full-2017-06-09/models/english-bidirectional-distsim.tagger'
        from nltk.tag.stanford import StanfordPOSTagger
        tagger = StanfordPOSTagger(model_path, path_pos)
        tagger.java_options = '-mx8096m'  ### Setting higher memory limit for long sentences
        tokens = nltk.word_tokenize(txt)
        pos_res =  tagger.tag(tokens)
        filepath='/home/ise/NLP/NLP3/pos/pos_{}.txt'.format(id)
        with open(filepath, 'w') as file_handler:
            for item in pos_res:
                file_handler.write("{}\n".format(item))
        return pos_res

def stanfordNE2BIO(tagged_sent):
    bio_tagged_sent = []
    prev_tag = "O"
    for token, tag in tagged_sent:
        if tag == "O": #O
            bio_tagged_sent.append((token, tag))
            prev_tag = tag
            continue
        if tag != "O" and prev_tag == "O": # Begin NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag == tag: # Inside NE
            bio_tagged_sent.append((token, "I-"+tag))
            prev_tag = tag
        elif prev_tag != "O" and prev_tag != tag: # Adjacent NE
            bio_tagged_sent.append((token, "B-"+tag))
            prev_tag = tag

    return bio_tagged_sent


def stanfordNE2tree(ne_tagged_sent):
    bio_tagged_sent = stanfordNE2BIO(ne_tagged_sent)
    sent_tokens, sent_ne_tags = zip(*bio_tagged_sent)
    sent_pos_tags = [pos for token, pos in pos_tag(sent_tokens)]

    sent_conlltags = [(token, pos, ne) for token, pos, ne in zip(sent_tokens, sent_pos_tags, sent_ne_tags)]
    ne_tree = nltk.conlltags2tree(sent_conlltags)
    return ne_tree

def pars_tree(tree_ne):
    ne_in_sent = []
    for subtree in tree_ne:
        if type(subtree) == Tree:  # If subtree is a noun chunk, i.e. NE != "O"
            ne_label = subtree.label()
            ne_string = " ".join([token for token, pos in subtree.leaves()])
            ne_in_sent.append((ne_string, ne_label))
    return ne_in_sent

if __name__ == "__main__":
    PN = PN_System()

