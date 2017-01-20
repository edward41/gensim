
#-*- coding: utf-8 -*-
import logging, gensim
import re
import csv
import time

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)

regr_exp = ur'[가-힣ᅵa-zA-Z]+'
#if you want including number, you can fix it to assign regr_exp as ur'[가-힣ᅵa-zA-Z0-9]+'
TOKEN_LENGTH_LOW = 1
GENREATE_TOKEN = 100000

def checkElapsedTime(func):

    def clock(*args):

        startTime = time.time()

        result = func(*args)

        elapsedTime = time.time() - startTime

        functionName = func.__name__
        argument = ', '.join(repr(arg) for arg in args)
        print('[%0.8fs] %s' % (elapsedTime, functionName))

        return result
    return clock

@checkElapsedTime
#sorted vocab and give a sorted word list
def vocab_list(W2V_Model) :
    model = W2V_Model
    count_list=[model.wv.vocab[w].count for w in model.wv.vocab]
    word_list=[w for w in model.wv.vocab]
    word =  [x for y, x in sorted(zip(count_list, word_list))][::-1]
    return word

@checkElapsedTime
#refine word.
def refine_word(word_list , regr_exp) :
    new_word_list = []
    for word in word_list :
        word = re.findall(regr_exp, word)
        try :
            #Word length is 2 or more
            if TOKEN_LENGTH_LOW < len(word[0]) :
                new_word_list.append(word[0])
        except IndexError :
            pass
    #Top GENREATE_TOKEN(20000) Token Frequecy return
    return new_word_list[0:GENREATE_TOKEN]

@checkElapsedTime
#generate csv.
def csv_generate(model, word_list):
    with open('gsw2v-dict_KVtype.csv', 'w') as csvfile2:
        writer = csv.writer(csvfile2)
        for word in word_list :
            try :
                nw_list = model.most_similar(positive=word)
                #Cosine similarity : most similar 10 word Search
                row_list=[word]
                for nw in nw_list :
                    #Refine word through regular expression only English and Korean
                    word=re.findall(regr_exp, nw[0])
                    if word != [] :
                        row_list+=word
                #if you want to generate arr_type csv file
                '''

                writer.writerow(row_list[0:8])
                '''
                #if you want to generate keyVlaue type csv file
                for i in row_list[1:8] :
                    writer.writerow([row_list[0],i])

            except KeyError:
                pass
def main():

    # load word2vec model.
    model = gensim.models.Word2Vec.load('./savefile')
    word_list = vocab_list(model)
    word_list = refine_word(word_list,regr_exp)
    csv_generate(model, word_list)

if __name__ == '__main__':
    main()
