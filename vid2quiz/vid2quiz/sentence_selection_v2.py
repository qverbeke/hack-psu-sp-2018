import nltk, re

def compute_document_score(original_corpus, path='./BTM/output/model/k10.pz_d', path_clust='./BTM/output/model/k10.pz', k=10, topn=20, write=False):

    original_sentences = original_corpus
    with open(path) as f:
        corpus = f.readlines()
    with open(path_clust) as f:
        scores = f.readlines()

    scores = [list(map(float, x.strip().split())) for x in scores][0]
    corpus = [list(map(float, x.strip().split())) for x in corpus]
    sentence_scores0 = []
    sentence_scores1 = []

    for c, clust in enumerate(corpus):
        top3 = sorted(range(len(clust)), key=lambda i: clust[i], reverse=True)[:3]
        sentence_scores1.append(scores[top3[0]] * .5 + scores[top3[1]]* .3 + scores[top3[2]]* .2)

    #select the top 20 cadidate sentences
    candidate_indices = sorted(range(len(sentence_scores1)), key=lambda i: sentence_scores1[i], reverse=True)[:topn]
    candidate_sentences = [original_sentences[i] for i in candidate_indices]
    # candidate_sentences_text = ''.join(candidate_sentences)

    #write down the candidate sentences
    # if(write):
    #     corp_file = open( "/home/herobaby71/Vid2Quiz/data/output/candidate.txt", "w")
    #     corp_file.write(candidate_sentences_text)
    #     corp_file.close()

    print(len(candidate_sentences))

    return candidate_sentences

# path = './BTM/output/model/k5.pz_d'
# compute_document_score(path, topn = 5)
