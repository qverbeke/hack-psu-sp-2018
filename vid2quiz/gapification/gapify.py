import subprocess, re, json, random
import retinasdk
from sentence_selection_v2 import compute_document_score
"""
[{
    original_sentence: "original sentence",
    gap_sentence: "sentence with gap",
    distractors: [],
    answer: 0
}]
"""

liteClient = retinasdk.LiteClient("f12af3f0-3a0d-11e8-9172-3ff24e827f76")

def distractor_selection():
    pass

def gapify(sentences):

    #get the keywords for every sentence
    keywords = []
    for sentence in sentences:
        keywords.append(liteClient.getKeywords(sentence))

    #get rid of gap and find selector
    gap_questions = []
    for i, sentence in enumerate(sentences):

        for k, key in enumerate(keywords[i]):
            gap_sentence = sentence.replace(key, '__________')
            distractors = liteClient.getSimilarTerms(key)[0:6]
            random.shuffle(distractors)
            answer_index = distractors.index(key)

            #create a new gap question
            new_gap_question = {}
            new_gap_question['original_sentence'] = sentence
            new_gap_question['gap_sentence'] = gap_sentence
            new_gap_question['answer'] = key
            new_gap_question['distractors'] = distractors
            gap_questions.append(new_gap_question)
    print(gap_questions)
    return gap_questions
def main():
    path='./BTM/output/model/k20.pz_d'
    path_clust='./BTM/output/model/k20.pz'
    k=20
    topn=10

    candidate_sentences = compute_document_score(path=path, path_clust=path_clust, k=k, topn=topn)
    questions = gapify(candidate_sentences)

main()
