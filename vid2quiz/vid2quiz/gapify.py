import retinasdk
import subprocess, re, json, random
from .saliencecall import getsalience
from .sentence_selection_v2 import compute_document_score #compute_document_score will return a list of candidate sentences
"""
{
    original_sentence: "original sentence",
    gap_sentence: "sentence with gap",
    distractors: [],
    answer: 0
}
"""

liteClient = retinasdk.LiteClient("f12af3f0-3a0d-11e8-9172-3ff24e827f76")


def gapify(sentences):
    #get the keywords for every sentence
    keywords = []
    for sentence in sentences:
        keywords.append(liteClient.getKeywords(sentence))


    #compute the salience for each candidate sentence (name, type, salience)
    # keywords_sal = []
    # for i,sentence in enumerate(sentences):
    #     keywords_sal.append(getsalience(sentence))

    #get all the common keywords from both keywords and keywords_sal
    # common_keywords = []
    # for i in range(len(sentences)):
    #     found = False
    #     for ks in keywords_sal[i]:
    #         for kk in keywords[i]:
    #             if(kk.lower() == ks.lower()):
    #                 common_keywords.append([ks])
    #                 found=True
    #                 break
    #         if(found):
    #             break
    #     if(not found):
    #         #check salience to see if there is a good noun
    #         # if()
    #         common_keywords.append([item for item in keywords[i][0:2]])

    # print("Sentences:", sentences[0:5])
    # print("keywords:",keywords[0:5])
    # print("keywords_sal:",keywords_sal[0:5])
    # print("common_keywords:",common_keywords[0:5])
    #get rid of gap and find selector
    gap_questions = []
    for i, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        sentence_tokens = sentence.strip().split()
        for k, key in enumerate(keywords[i]):
            if(k >2): break
            index_range = [0,0]
            index_range[0] = re.search(key, sentence_lower).start()
            index_range[1] = index_range[0]+len(key)

            #create gap sentence
            gap_sentence = ''.join((sentence[0:index_range[0]],"_________",sentence[index_range[1]:]))

            #get the distractors
            distractors = liteClient.getSimilarTerms(key)[0:4]
            random.shuffle(distractors)
            try:
                answer_index = distractors.index(key)
            except:
                break
            #create a new gap question
            new_gap_question = {}
            new_gap_question['original_sentence'] = sentence
            new_gap_question['gap_sentence'] = gap_sentence
            new_gap_question['answer'] = key
            new_gap_question['distractors'] = distractors
            gap_questions.append(new_gap_question)
    return gap_questions

def generate_gap_sentences(corpus, topn = 10):
    corpus = re.split(r'(?<=\.) ', corpus)
    path='/home/herobaby71/Vid2Quiz/BTM/output/model/k10.pz_d'
    path_clust='/home/herobaby71/Vid2Quiz/BTM/output/model/k10.pz'
    k=10

    candidate_sentences = compute_document_score(corpus, path=path, path_clust=path_clust, k=k, topn=topn)
    questions = gapify(candidate_sentences)
    return questions
