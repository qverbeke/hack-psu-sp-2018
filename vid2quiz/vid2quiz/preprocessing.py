import numpy as np
import nltk,re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer

lemmatize = True
rm_stopwords = True
num_sentences = 20
stopwords = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
vectorizer = TfidfVectorizer()


def tffilter(tokens, tfeatures = []):
    tokens = [t.lower() for t in tokens]
    new_tokens = []
    if(lemmatize):
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        for i,w in enumerate(tokens):
            if(len(wn.synsets(w)) > 0 and wn.synsets(w)[0].pos() == 'v'):
                tokens[i] = lemmatizer.lemmatize(w, 'v')
    if(rm_stopwords):
        tokens = [t for t in tokens if t not in stopwords]
    for t in tokens:
        if(t in tfeatures):
            new_tokens.append(t);
    return new_tokens

def clean_sentence(tokens, tfilter = False, tfeatures = []):
	tokens = [t.lower() for t in tokens]
	if lemmatize:
		tokens = [lemmatizer.lemmatize(t) for t in tokens]
		for i,w in enumerate(tokens):
			if(len(wn.synsets(w)) > 0 and wn.synsets(w)[0].pos() == 'v'):
				tokens[i] = lemmatizer.lemmatize(w, 'v')
	if rm_stopwords:
		tokens = [t for t in tokens if t not in stopwords]
	return tokens

def preprocess(corpus, outpath='/home/herobaby71/Vid2Quiz/BTM/sample-data/doc_info.txt'):
    """
        corpus: array of sentences
        outpath: file to save the files to
    """
    corpus = re.split(r'(?<=\.) ', corpus)
    corpus = [nltk.word_tokenize(x.strip().replace('.',' .').replace("\"", "\" ").replace(",", ' ,').replace('(', '( ').replace(')', ') ')) for x in corpus]

    new_corpus = []
    for i,sentence in enumerate(corpus):
        new_corpus.append(' '.join(clean_sentence(sentence)))

    #get rid of trash features
    # X = vectorizer.fit_transform(new_corpus)
    # indices = np.argsort(vectorizer.idf_)[::-1]
    # features = vectorizer.get_feature_names()
    # top_n = 14500
    # tfidf_features = set([features[i] for i in indices[:top_n]])

    # new_new_corpus = []
    # for i, sentence in enumerate(corpus):
    #     new_new_corpus.append(' '.join(clean_sentence(sentence, True, tfidf_features)))

    new_new_corpus = new_corpus #remove if uncomment above

    #make a doc and output
    corpus_text = '\n'.join(new_new_corpus)

    corp_file = open(outpath, "w")
    corp_file.write(corpus_text)
    corp_file.close()

def main():
    path = './data/documents/docA.txt'
    preprocess(path)
# if __name__ == '__main__':
# 	main()
