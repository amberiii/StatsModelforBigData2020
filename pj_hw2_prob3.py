#############################################
#    SDS385 Homework Assignment 2.3 - PJ    #
#############################################


# Load packages and data
import numpy as np
import time
import nltk
import binascii
import matplotlib.pyplot as plt
nltk.download('stopwords')# only need run once
stop_words = nltk.corpus.stopwords.words('english')

def load_data():
    with open('articles-1000.txt', 'r') as f:
        contents = f.readlines()
    return contents



################################################ 
# 3.a) Split each article into hashed 2-shingles 
#################################################

def problem_part_a(contents):
    num_doc = len(contents)
    article_to_shingles = {}
    article_IDs = []

    for content in contents:
        splits = content.strip().split(" ")
        tID = splits[0]
        rest = splits[1:]
        article_IDs.append(tID)

        # remove stop words
        rest = [x for x in rest if x not in stop_words]


        # build shingle
        shingles = [rest[i] + ' ' + rest[i+1] for i in range(len(rest)-1)]
        shingle_IDs = [binascii.crc32(str.encode(shingle)) & 0xffffffff for shingle in shingles]

        article_to_shingles[tID] = shingle_IDs

    # total number of unique shingle
    shingles = []
    for x in article_to_shingles:
        shingles.extend(article_to_shingles[x])
    print('total unique shingle ', len(set(shingles)))

    # average number of shingles per article
    avg = len(shingles) / float(len(article_to_shingles))
    print('average single per article ', avg)

    # average number of unique shingles per article
    avg = len(set(shingles)) / float(len(article_to_shingles))
    print('average unique single per article ', avg)

    return article_IDs, article_to_shingles

##################################################################
# 3.b) MinHash signatures 
#       &
#      The article with largest Jaccard similarity to 1st article.
##################################################################
def jaccard(set0, set1):
    # compute jaccard similarity score of two sets
    return len(set0.intersection(set1))/float(len(set0.union(set1)))

def hash_fn(p, c1, c2, x):
    h = (c1 * x + c2) % p
    return h

def construct_minhash_signitures(article_IDs, article_to_shingles, n):
    signitures = []
    p = 4294967311

    # generate n random coefficients
    c1s = np.random.randint(0, p, n)
    c2s = np.random.randint(0, p, n)
    num_doc = len(article_IDs)

    # generate signature for all articles
    for i in range(num_doc):
        tID = article_IDs[i]
        signiture = np.zeros([n])
        for j in range(n):
            c1 = c1s[j]
            c2 = c2s[j]
            sig = p+1
            for shingle in article_to_shingles[tID]:
                val = hash_fn(p, c1,c2,shingle)
                if val < sig:
                    sig = val
            signiture[j] = sig
        signitures.append(signiture)
    return signitures

def problem_part_b(article_IDs, article_to_shingles):
    
    num_doc = len(article_IDs)

    signitures = construct_minhash_signitures(article_IDs, article_to_shingles, 10)

    # compare the first article's signiture to all others
    scores = np.zeros([num_doc])
    for i in range(1, num_doc):
        scores[i] = (signitures[0] == signitures[i]).mean()

    idx = np.argmax(scores)
    estimate_jaccard = scores[idx]

    true_jaccard = jaccard(set(article_to_shingles[article_IDs[0]]), set(article_to_shingles[article_IDs[idx]]))
    print(idx, estimate_jaccard, true_jaccard)
    return (idx, estimate_jaccard, true_jaccard)
    #print(contents[0], contents[idx])


################################################ 
# 3.c) Amplification
#################################################      

def problem_part_c(article_IDs, article_to_shingles):
    # first compute the true jaccard measure
    num_doc = len(article_IDs)
    simij = np.zeros([num_doc])
    for i in range(1, num_doc):
        simij[i] = jaccard(set(article_to_shingles[article_IDs[0]]),set(article_to_shingles[article_IDs[i]]))


    t = 0.8
    r = 2
    fp = []
    bs = [1,3,5,7,9]
    num_repeat=10
    for b in bs:
        print(f"b: {b}, r: {r}")
        n = b*r
        fp_tp = 0
        for repeat in range(num_repeat):
            signitures = construct_minhash_signitures(article_IDs, article_to_shingles, n)

            # find items similar to item[0] using Local Sentitivity Hashing (LSH)
            band_hash = {}
            for i in range(num_doc):
                for k in range(b):
                    sig = signitures[i][r*k:r*(k+1)]
                    key = f"band_{k}_{str(sig)}"
                    if key in band_hash:
                        band_hash[key].add(i)
                    else:
                        band_hash[key] = set()
                        band_hash[key].add(i)

            similar_set = set()
            for k in range(b):
                sig = signitures[0][r*k:r*(k+1)]
                key = f"band_{k}_{str(sig)}"
                similar_set.update(band_hash[key])
            cnt = 0
            for id in similar_set:
                if id == 0:
                    continue
                if simij[id] < t:
                    cnt+=1
            fp_tp += cnt / (len(similar_set)-1)
        fp_tp /= num_repeat
        fp.append(fp_tp)

    plt.clf()
    plt.plot(bs, fp)
    plt.xlabel('b')
    plt.ylabel('fp')
    plt.title('b v.s. fp with r == 2')
    plt.savefig('prob3_c1.png')

    t = 0.8
    b = 2
    fp = []
    rs = [1,3,5,7,9]
    num_repeat=10
    for r in rs:
        print(f"b: {b}, r: {r}")
        n = b*r
        fp_tp = 0
        for repeat in range(num_repeat):
            signitures = construct_minhash_signitures(article_IDs, article_to_shingles, n)

            # find items similar to item[0] using Local Sentitivity Hashing (LSH)
            band_hash = {}
            for i in range(num_doc):
                for k in range(b):
                    sig = signitures[i][r*k:r*(k+1)]
                    key = f"band_{k}_{str(sig)}"
                    if key in band_hash:
                        band_hash[key].add(i)
                    else:
                        band_hash[key] = set()
                        band_hash[key].add(i)

            similar_set = set()
            for k in range(b):
                sig = signitures[0][r*k:r*(k+1)]
                key = f"band_{k}_{str(sig)}"
                similar_set.update(band_hash[key])
            cnt = 0
            for id in similar_set:
                if id == 0:
                    continue
                if simij[id] < t:
                    cnt+=1
            fp_tp += cnt / (len(similar_set)-1)
        fp_tp /= num_repeat
        fp.append(fp_tp)

    plt.clf()
    plt.plot(bs, fp)
    plt.xlabel('r')
    plt.ylabel('fp')
    plt.title('r v.s. fp with b == 2')
    plt.savefig('prob3_c2.png')

################################################ 
# 3.d) Five Similar Pairs
#################################################  
def problem_part_d(article_IDs, article_to_shingles):

    # find 5 most similar items without approximation
    K = 5
    time_start = time.time()
    num_doc = len(article_IDs)
    simij = np.zeros([num_doc, num_doc])
    for i in range(num_doc):
        if i % 100 == 0:
            print(i, num_doc)
        for j in range(i+1, num_doc):
            simij[i,j] = jaccard(set(article_to_shingles[article_IDs[i]]),set(article_to_shingles[article_IDs[j]]))
    ids = np.argsort(simij.flatten())[-K:]
    id_target = ids % num_doc
    id_source = ids // num_doc
    print(f"time elapsed for finding top {K} similar pairs without any approximation: {time.time() - time_start}")

    # find 5 most similar items using LSH
    rs = [1,3,5,7,9]
    bs = [1,3,5,7,9]
    for b in bs:
        for r in rs:
            time_start = time.time()
            n = b*r
            fp_tp = 0
            signitures = construct_minhash_signitures(article_IDs, article_to_shingles, n)

            band_hash = {}
            for i in range(num_doc):
                for k in range(b):
                    sig = signitures[i][r*k:r*(k+1)]
                    key = f"band_{k}_{str(sig)}"
                    if key in band_hash:
                        band_hash[key].add(i)
                    else:
                        band_hash[key] = set()
                        band_hash[key].add(i)

            sims = []
            sim_ids = []
            for i in range(num_doc):
                similar_set = set()
                for k in range(b):
                    if k == i:
                        continue
                    sig = signitures[i][r*k:r*(k+1)]
                    key = f"band_{k}_{str(sig)}"
                    similar_set.update(band_hash[key])
                for id in similar_set:
                    sims.append(jaccard(set(article_to_shingles[article_IDs[i]]),set(article_to_shingles[article_IDs[id]])))
                    sim_ids.append([i, id])
            ids = np.argsort(sims)[-5:]
            print(f"time elapsed for finding top {K} similar pairs using LSH with b: {b}, r: {r}: {time.time() - time_start}")




if __name__ == '__main__':

    contents = load_data()

    print('\n\nproblem a: ')
    article_IDs, article_to_shingles = problem_part_a(contents)

    print('\n\nproblem b: ')
    problem_part_b(article_IDs, article_to_shingles)

    print('\n\nproblem c: ')
    problem_part_c(article_IDs, article_to_shingles)

    print('\n\nproblem d: ')
    problem_part_d(article_IDs, article_to_shingles)

    print('\n\n Thanks god it is done!')
