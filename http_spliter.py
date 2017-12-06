

import gzip
import sys
import re
import gensim, logging
from optparse import OptionParser

filename = "/home/yodzeb/Downloads/http.log.gz"
load     = False
#filename = False

parser = OptionParser()
parser.add_option("-f", "--file", action="store", type="string", dest="filename")
parser.add_option("-m", "--max",  action="store", type="string", dest="max")
(options, args) = parser.parse_args()
if options.filename:
    filename  = options.filename
    load=True
max_lines = 300000 # default
if options.max:
    max_lines=int(options.max)

class gzip_sentences():
    def __iter__(self):
        i=0
        handle = gzip.open(filename, "r")
        for line in handle:
            i=i+1
            if (i%1000 == 0):
                print ".",
                sys.stdout.flush()
            if (i==max_lines):
                # Only take 300k entries
                break
            req=line.split("\t")[9]
            words = re.split(r'(?:\/|=|\?|\\|\&|\s|\'|,|\+)', req)
            words = map(str.lower, words)
            words = [ w[0:32] for w in words ][1:]
            #if (i%50000 == 0):
            #print line
            #print words
            #  print words
            yield words # sentences.append(words)
        handle.close()

model=""
if load:
    sentences= gzip_sentences()
    model = gensim.models.Word2Vec(sentences, min_count=1)
    model.save('./mymodel-300k')
else:
    model = gensim.models.Word2Vec.load('./mymodel-300k')

print model
if not 'zzzzzzz' in model.wv.vocab:
    print "not foufound"
print model.most_similar(positive=['etceee'], topn=10)
#print model['admin']
# new_model = gensim.models.Word2Vec.load('/tmp/mymodel')
