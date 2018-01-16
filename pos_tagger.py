import pandas as pd
import os
os.environ['STANFORD_MODELS'] = '/home/ise/NLP/stanfordNLP/stanford-ner-2017-06-09/classifiers'
os.environ['CLASSPATH'] = '/home/ise/NLP/stanfordNLP/stanford-ner-2017-06-09'

from nltk import  pos_tag
from nltk import ngrams
from nltk.tree import Tree
import numpy as np
import nltk

import matplotlib.pyplot as plt

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
        self.read_ner_file()
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

    def sample_sentences_by_name(self,name):
        df_tmp = self.df.loc[self.df['president'] == name]


    def read_ner_file(self,norm=True,lower=True):
        path =  '/home/ise/NLP/NLP3/ner/'
        list_ner = []
        list_word = []
        arr_txt = [x for x in os.listdir(path) if x.endswith(".txt")]
        print "size", len(arr_txt)
        for x in arr_txt:
            id_num = (x[4:-4])
            id_num = int(id_num)

            d_ner_ctr = {'ID': id_num}
            d_ner_word = {'ID': id_num}
            with open(path + x, 'r') as file_pos:
                ctr_ner_number = 0
                for line in file_pos:
                    if len(line) > 2:
                        if id_num == 194:
                            arr=str(line).split(': ')
                            ner_i = arr[0]
                            word_i = arr[1]
                        else:
                            arr = line[1:-2].split(', ')
                            ner_i = arr[1][2:-1]
                            word_i = arr[0][2:-1]
                        if lower:
                            word_i = str(word_i).lower()
                        key="{} | {}".format(word_i,ner_i)
                        if word_i in d_ner_word:
                            d_ner_word[key]=d_ner_word[key]+1
                        elif len(str(word_i)) > 1:
                            d_ner_word[key]=1
                        if ner_i in d_ner_ctr:
                            d_ner_ctr[ner_i]=d_ner_ctr[ner_i]+1
                        elif len(str(ner_i))>1:
                            d_ner_ctr[ner_i]=1
                        ctr_ner_number+=1
                d_ner_ctr['All_NER'] = ctr_ner_number
                d_ner_word['All_NER'] = ctr_ner_number
                list_ner.append(d_ner_ctr)
                list_word.append(d_ner_word)
        print "done"
        df_word = pd.DataFrame(list_word)
        df_ctr_ner = pd.DataFrame(list_ner)
        df_ctr_ner.fillna(float(0),inplace=True)
        df_word.fillna(float(0),inplace=True)
        print "list df_ctr_ner:{}".format(list(df_ctr_ner))
        if norm:
            df_ctr_ner = self.norm_df(df_ctr_ner,'All_NER')
            df_word = self.norm_df(df_word,'All_NER')
        else:
            df_ctr_ner = df_ctr_ner.drop(['All_NER'], axis=1)
            df_word = df_word.drop(['All_NER'], axis=1)

        #self.get_top_NER(df_word, 'LOCATION')
        #self.get_top_NER(df_word, 'PERSON')
        #self.get_top_NER(df_word, 'ORGANIZATION')
        #exit(0)
        df = self.group_by_col(self.mereg_df(df_word),'year')
        argss=['spain | LOCATION','china | LOCATION']
        df.plot('year',argss)
        plt.show()
        exit()
        self.get_top_word_each_year(df)



        #self.time_stat(df_ctr_ner,['ORGANIZATION','PERSON','LOCATION'],'All_NER')

    def get_top_word_each_year(self,df):
        if 'president' in df:
            df =  df.drop(['president'], axis=1)
        if 'ID' in df:
            df = df.drop(['ID'], axis=1)
        all_years = df['year'].unique().tolist()
        for x in all_years:
            print "----->> year:{}".format(x)
            result_df = df.loc[df['year'] == x]
            result_df = result_df.drop(['year'], axis=1)
            self.k=1
           # self.get_top_NER(result_df,'ORGANIZATION')
           # self.get_top_NER(result_df, 'PERSON')
            self.get_top_NER(result_df, 'LOCATION')

    def get_top_NER(self,_df,name): #['ORGANIZATION','PERSON','LOCATION']
        self.top_k = [0]*self.k
        self.top_k_grams=[None]*self.k
        df = _df.copy(deep=True)
        list_col = list(df)
        print "size list :  {}".format(len(list_col))
        if 'ID' in list_col:
            list_col.remove('ID')
        #print list(list_col)
        list_col = [x for x in list_col if str(x).split(' | ')[1] == name ]
        #list_col.append('ID')
        df_name = df[list_col]
        list_col = list(df_name)
        print "size list after del :  {}".format(len(list_col))
        d_stat={}
        #list_col.remove('ID')
        for col in list_col:
            sum_i= df[col].sum()
            d_stat[col]=sum_i
            self._manger_array_min(sum_i,col)
        print self.top_k_grams
        print "#"*200
        print self.top_k

    def group_by_col(self, df, col_name):
        list_col = list(df)
        list_col = [x for x in list_col if x not in ['president', 'year', 'ID']]
        d_agg={}
        for x in list_col:
            d_agg[x]='mean'
        list_col.append(col_name)
        df = df[list_col]
        res = df.groupby(col_name, as_index=False).agg(d_agg)
        return res

    def filter_by_col_val(self,df,col_name,val):
        return df.loc[df[col_name] == val ]

    def mereg_df(self,df):
        return  pd.merge(self.df, df, on='ID')


    def norm_df(self,df,name_all):
        df2 = pd.DataFrame(df['ID'])
        list_col = list(df)
        list_col.remove('ID')
        list_col.remove(name_all)
        for pos in list_col:
            df2[pos] = df[pos] / df[name_all]
        return df2
    def read_pos_file(self):
        path = '/home/ise/NLP/NLP3/pos/'
        list_pos=[]
        d = {}
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

        for name_p in ["George Washington", "Abraham Lincoln", "Richard Nixon", "Ronald Reagan" ,"Barack Obama","Donald J. Trump"]:
            uni = df_uni.copy(deep=True)
            bi = df_bi.copy(deep=True)
            tri = df_tri.copy(deep=True)
            list_df=[[uni,'uni'],[bi,'bi'],[tri,'tri']]
            self.extract_df_president(list_df,name_p)
        exit(0)




        print "{} size: {}".format("uni",df_uni.shape)
        print "{} size: {}".format("bi", df_bi.shape)
        print "{} size: {}".format("tri", df_tri.shape)
        print "````normalize:\n\n"
        res  = self.get_statistics(self.make_df_pos_normalizer(df_uni))
        print "----count:\n\n"
        res = self.get_statistics(df_uni)
        print "````normalize:\n\n"
        res  = self.get_statistics(self.make_df_pos_normalizer(df_bi))
        print "---count:\n\n"
        res = self.get_statistics(df_bi)
        print "````normalize:\n\n"
        res  = self.get_statistics(self.make_df_pos_normalizer(df_tri))
        print "----count:\n\n"
        res = self.get_statistics(df_tri)
        exit(0)
        print "sorting..."

        new_d = {}
        sortedList = sorted(res.values())
        for sortedKey in sortedList:
            for key,val in res.items():
                if val == sortedKey:
                    new_d[key]=val


       # result_df = pd.merge(self.df,df,on='ID')
       # self.df = result_df.fillna(0)
       # print "merge.df = ", self.df.shape
       # print "done"

    def extract_df_president(self,list_df,name):
        for df_i in list_df:
            print self.df.shape
            result_df = pd.merge(self.df, df_i[0], on='ID')
            print result_df.shape
            result_df = result_df.loc[result_df['president'] == name]
            file_name = "{}_{}_gram".format(name,df_i[1])
            result_df = result_df.drop(['president','year'], axis=1)
            result_df = self.make_df_pos_normalizer(result_df)
            self.drop_to_disk(self.get_statistics(result_df),file_name)

    def drop_to_disk(self,d,name):
        arr= name.split('_')
        path = '/home/ise/NLP/data_npl3/'
        if os.path.isdir(path+arr[0]) is False:
            os.mkdir(path + arr[0])
        to_dump = pd.DataFrame(d)
        to_dump.to_csv('{}/{}.csv'.format(path+arr[0],name))

    def freq_ngrams_dico(self,arr_tokens,n):
        ctr = 0
        d={}
        token_grams = ngrams(arr_tokens,n)
        for gram in token_grams:
            if gram in d :
                d[gram]+=1
            else:
                d[gram]=1
            ctr=ctr+1
           # print "ctr",ctr
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

    def make_df_pos_normalizer(self,df):

        df2 = pd.DataFrame(df['ID'])
    #    df2 = df['ID'].copy(deep=True)
        list_col = list(df)
        list_col.remove('ID')
        list_col.remove('All_POS')
        for pos in list_col:
            df2[pos]=df[pos]/df['All_POS']
        return df2

    def get_statistics(self,df):
        d_stat={}
        self.top_k = [0]*self.k
        self.top_k_grams=[None]*self.k
        list_start = list(df)
        #print "list_start",(list_start)
        if 'All_POS' in df:
            df = df.drop(['All_POS'], axis=1)
        df = df.drop(['ID'],axis=1)
        df = df._get_numeric_data()
        list_col = list(df)
        #print "list_col ", len(list_col )
        for col in list_col:
            sum_i = df[col].sum()
            d_stat[col]= sum_i
            self._manger_array_min(sum_i,col)
        print self.top_k_grams
        print "#"*200
        print self.top_k
        d=[]
        for i in range(len(self.top_k_grams)):
            d.append({'pos':str(self.top_k_grams[i]),'frequency':self.top_k[i]})
        return d

    def time_stat(self,df,pos_arr,all='All_POS'):
        result_df = pd.merge(self.df, df, on='ID')
        result_df.groupby(['year'])
        #print list(result_df)
        print "making plot.."
        for pos in pos_arr:
            result_df['norm_'+pos] = result_df[pos] / result_df[all]
        list_pos = ['norm_'+x for x in pos_arr]
       # print result_df[pos_arr][:100]
        result_df.plot('year',list_pos)
        plt.show()

    def _manger_array_min(self,num,gram):
        if num<=float(0):
            return
        #print self.top_k_grams
       # print "topk=",self.top_k
       # print "top grams = ",self.top_k_grams
       # print "adding: ",num,gram
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
        dfList = df['text'].tolist()
        df.drop('text', axis=1, inplace=True)
        self.df=df
        print "done upload self.df"
        #print self.df["president"].unique()
        #exit(0)
        return
        self.df['ner'] = np.nan
        self.df['ner'] = df.apply(lambda x: self._NER(x['text'],x['ID']), axis=1)
        self.df['pos'] = np.nan
        self.df['pos'] =  df.apply(lambda x: self._POS(x['text'],x['ID']), axis=1)
        self.df.to_csv('pos_ner.csv')

    def _drop_txt(self,txt,id):
        filepath='/home/ise/NLP/NLP3/txt/txt_{}.txt'.format(id)
        with open(filepath, 'w') as file_handler:
            file_handler.write("{}".format(txt))
        return True
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