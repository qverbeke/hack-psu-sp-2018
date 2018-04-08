import nltk, sys, glob
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import wordnet as wn

lemmatize = True
rm_stopwords = True
num_sentences = 20
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()



def get_probabilities(cluster, lemmatize, rm_stopwords):
	# Store word probabilities for this cluster
	word_ps = {}
	# Keep track of the number of tokens to calculate probabilities later
	token_count = 0.0
	# Gather counts for all words in all documents
	for path in cluster:
		with open(path) as f:
			tokens = clean_sentence(nltk.word_tokenize(f.read()))
			token_count += len(tokens)
			for token in tokens:
				if token not in word_ps:
					word_ps[token] = 1.0
				else:
					word_ps[token] += 1.0
	# Divide word counts by the number of tokens across all files
	for word_p in word_ps:
		word_ps[word_p] = word_ps[word_p]/float(token_count)
	return word_ps

def get_sentences(cluster):
	sentences = []
	for path in cluster:
		with open(path) as f:
			sentences += nltk.sent_tokenize(f.read())
	return sentences

def clean_sentence(tokens):
	tokens = [t.lower() for t in tokens]
	if lemmatize:
		tokens = [lemmatizer.lemmatize(t) for t in tokens]
		for i,w in enumerate(tokens):
			if(len(wn.synsets(w)) > 0 and wn.synsets(w)[0].pos() == 'v'):
				tokens[i] = lemmatizer.lemmatize(w, 'v')
	if rm_stopwords:
		tokens = [t for t in tokens if t not in stopwords]
	return tokens

def score_sentence(sentence, word_ps):
	score = 0.0
	num_tokens = 0.0
	sentence = nltk.word_tokenize(sentence)
	tokens = clean_sentence(sentence)
	for token in tokens:
		if token in word_ps:
			score += word_ps[token]
			num_tokens += 1.0
	return float(score)/float(num_tokens)

def max_sentence(sentences, word_ps, simplified):
	max_sentence = None
	max_score = None
	for sentence in sentences:
		score = score_sentence(sentence, word_ps)
		if  max_score is None or score > max_score:
			max_sentence = sentence
			max_score = score
	if not simplified: update_ps(max_sentence, word_ps)
	return max_sentence

def update_ps(max_sentence, word_ps):
	sentence = nltk.word_tokenize(max_sentence)
	sentence = clean_sentence(sentence)
	for word in sentence:
		word_ps[word] = word_ps[word]**2
	return True

def orig(cluster):
	cluster = glob.glob(cluster)
	word_ps = get_probabilities(cluster, lemmatize, rm_stopwords)
	sentences = get_sentences(cluster)
	tokenizer = RegexpTokenizer(r'\w+')
	summary = []
	corpus = []
	#save the entire preprocessed corpus
	for i in range(len(sentences)):
		sent = tokenizer.tokenize(sentences[i].lower())
		filtered_words = clean_sentence(sent)
		corpus.append(' '.join(filtered_words))
	print(corpus[50:60])
	#save top n sentences (summary)
	for i in range(num_sentences):
		sent = max_sentence(sentences, word_ps, False).lower()
		summary.append(' '.join(tokenizer.tokenize(sent)))

	ret_summary = "\n".join([item.replace('.', ' .').replace(',', ' ,').lower() for item in summary])
	ret_corpus = "\n".join([item.replace('.', ' .').replace(',', ' ,').lower() for item in corpus])
	return (ret_summary, ret_corpus)

def leading(cluster):
	cluster = glob.glob(cluster)
	sentences = get_sentences(cluster)
	summary = []
	for i in range(num_sentences):
		summary.append(sentences[i])
	return summary

def main():
	path_cluster = './data/documents/docA.txt'
	path_save = './data/output/'
	path_corpus = './BTM/sample-data/'
	summary, corpus = orig(path_cluster)

	summ_file = open(path_save + "summary.txt", "w")
	summ_file.write(summary)
	summ_file.close()

	corp_file = open(path_corpus + "doc_info.txt", "w")
	corp_file.write(corpus)
	corp_file.close()

if __name__ == '__main__':
	main()
