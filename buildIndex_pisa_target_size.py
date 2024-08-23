import json
import sys
import gzip
import numpy as np
import struct
from numpy.core.defchararray import encode
from pyfastpfor import getCodec
from tqdm import tqdm

def convertBinary(num):
    n = int(num)
    return struct.pack('<I', n)

def binarySequence(arr, fout):
    size = len(arr)
    fout.write(convertBinary(size))
    for i in arr:
        fout.write(convertBinary(i))

def deltaEncoding(arr):
    assert(len(arr) % 2 == 0)
    for i in range(len(arr) - 2, 0, -2):
        arr[i] = arr[i] - arr[i - 2]

if __name__ == "__main__":

    json_path = sys.argv[1]

    posting = {}

    length = []

    lens = int(sys.argv[3])
    filen = int(sys.argv[4])
    thres = 25
    lower = thres
    upper = thres

    i = 0
    while i  < filen:
        print(i)
        try:
            for line in gzip.open("%s%d.jsonl.gz" % (json_path, i)):
                doc_dict = json.loads(line)
                id = doc_dict['id']

                vector = doc_dict['vector']

                length_t = 0
                for k in vector:
                    if vector[k] > thres:
                        length_t += 1

                        if k not in posting:
                            posting[k] = []

                        posting[k] += [id, vector[k]]
                
                length.append(length_t)
        except:
            print("end of file")
        print(len(length))
        if i == 0 and np.mean(length) > 310:
            print("current length", np.mean(length))
            print("thres", thres)
            if thres < upper:
                print("fix thred")
            else:
                thres += 5
                upper = thres
                i -= 1

        elif i == 0 and np.mean(length) < 290:
            print("current length", np.mean(length))
            print("thres", thres)
            if thres > lower:
                print("fix thred")
            else:
                thres -= 5
                lower = thres
                i -= 1
        elif i == 0:
            print("fixed thred to be %d" %thres)
        i += 1

    term_id = {}
    id = 0
    for k in posting:
        term_id[k] = id
        id += 1

    with open(sys.argv[2] + '.id', 'w') as f:
        json.dump(term_id, f)
        
    fout_docs = open(sys.argv[2] + ".docs", 'wb')
    fout_freqs = open(sys.argv[2] + ".freqs", 'wb')
    binarySequence([len(length)], fout_docs)


    for k in tqdm(posting):
        binarySequence(posting[k][::2], fout_docs) # docIDs
        binarySequence(posting[k][1::2], fout_freqs) # score instead of freq
    fout_docs.close()
    fout_freqs.close()
    fout_sizes = open(sys.argv[2] + ".sizes", 'wb')
    binarySequence(length, fout_sizes)

        # fout_raw.write(k + ' ' + str(term_id[k]) + ' '.join([str(i) for i in posting[k]]) + '\n')
        # fout_info.write(k + ' ' + str(term_id[k]) + ' ' + str(len(posting[k]) // 2))

        # for i in range(0, len(posting[k]), 2 * STEP):
        #     p_bytes = np.array(posting[k][i : i + 2 * STEP], dtype=np.uint32, order='C')
        #     deltaEncoding(p_bytes)
        #     size = len(p_bytes)
        #     encoded = np.zeros(size + 1024, dtype = np.uint32, order='C')
        #     encoded_size = codec.encodeArray(p_bytes, size, encoded, size + 1024)

        #     start_pos = fout_data.tell()
        #     encoded_bytes = bytes(encoded[:encoded_size])
        #     fout_data.write(encoded_bytes)

        #     fout_info.write(' ' + str(start_pos) + ' ' + str(encoded_size * 4))
            
        # fout_info.write('\n')
            
