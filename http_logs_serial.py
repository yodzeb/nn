
from optparse import OptionParser
import gzip
import tensorflow as tf
import re
import sys
import gensim
import numpy as np
import pickle

sentence_size=64

filename = "/home/yodzeb/Downloads/http.log.gz"


def max_list(list):
    maxx = 0
    for i in list:
        if len(list[i]) > maxx:
            maxx=len(list[i])
    return maxx

def classify (line):
    # Classification
    # [ SQL, LFI, BIN, WP, OTHER ]
    labels=[0,0,0,0]
    # XSS
    if re.search(r'script', line) and re.search(r'<', line):
        labels = [1,0,0,0,0]
    # LFI
    elif re.search(r'\.(?:\/|\\)\.', line) or re.search(r'(?:passwd|boot\.ini|winnt)', line):
        labels = [0,1,0,0,0]
    # SHELLCODE
    elif re.search(r'\\x..\\x', line):
        labels = [0,0,1,0,0]
    # RFI
    elif re.search(r'http:', line):
        labels = [0,0,0,1,0]
    # OTHER
    else:
        labels = [0,0,0,0,1]
    return labels

def writable ( list_cat, cat ):
    cur=list_cat[cat]
    for i in range ( 0, 5 ):
        i = str(i)
        if list_cat[i] > cur:
            return True
        elif list_cat[i] < cur-400:
            return False
    return True

def do_file (filename, c, model, vec_size, out_file):
    record_list = {
        "0": 0,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0
    }

    handle = gzip.open (filename, "r");
    i=0
    final_list=[];
    for line in handle:
        i=i+1
        if (i%100 == 0):
            print ".",
            sys.stdout.flush()
        if i==c:
            break
        req=line.split("\t")[9]
        words = re.split(r'(?:\/|=|\?|\\|\&|\s|\'|,|\+)', req)
        words = map(str.lower, words)
        output = []
        count = 0
        for w in words:
            count=count+1
            if w in model.wv.vocab:
                output.append(model[w].tolist())
            else:
                output.append([0]*vec_size)
            if count == sentence_size:
                break
        #Padding
        while count < sentence_size:
            output.append([0]*vec_size)
            count=count+1
                
        labels = classify(line)
        record = { "la": labels, "wo": np.reshape(output, -1)   }
        cat = str(labels.index(1));
        #print record
        if writable ( record_list, cat):
            print "!",
            record_list[cat] = record_list[cat] + 1
            req = req[0:sentence_size]
            req = req.ljust(sentence_size)
            #write_record(writer, record, req)
            final_list.append(record);
        else:
            print 'x',
    return final_list
        

# output files
train_filename="./http_train.tfrec";
valid_filename="./http_valid.tfrec";

parser = OptionParser()
parser.add_option("-f", "--file", action="store", type="string", dest="filename")
(options, args) = parser.parse_args()
if (options.filename):
    filename=options.filename

# Load dictionary
model = gensim.models.Word2Vec.load('./mymodel-300k')
vec_size = len(model['index.php'])
print "Vectors size: "+str(vec_size)

print "train"
rec_train = do_file("/home/yodzeb/Downloads/http_train.gz", 200000, model, vec_size, train_filename)
print "valid"
rec_valid = do_file("/home/yodzeb/Downloads/http_valid.gz", 3000,  model, vec_size, valid_filename)

with open('train.pkl', 'wb') as f:
    pickle.dump(rec_train, f, pickle.HIGHEST_PROTOCOL)

with open('valid.pkl', 'wb') as f:
    pickle.dump(rec_valid, f, pickle.HIGHEST_PROTOCOL)

# max_train = max_list(rec_train)
# max_valid = max_list(rec_valid)

# print "max_train: "+str(max_train)
# print "max_valid: "+str(max_valid)


# write_max( train_filename, rec_train, max_train )
# write_max( valie_filename, rec_valid, max_valid )

#write_dataset(data_train, train_filename)
#write_dataset(data_valid, valid_filename)
    
