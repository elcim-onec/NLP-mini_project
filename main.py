import nltk
from nltk import sent_tokenize, word_tokenize, bigrams, ProbDistI
from nltk.probability import MLEProbDist, ELEProbDist

file = input("File directory: ")
inputFile = open(file).read()



tokenized_text = len(nltk.sent_tokenize(inputFile))
print(sent_tokenize(inputFile))
print("tokenized text: ", tokenized_text, "\n")


tokenized_text = nltk.word_tokenize(inputFile)
tokenized_text = [word.lower() for word in tokenized_text if word.isalpha()]
print("Lower cased text: ", tokenized_text)
print("Word Count: ", len(tokenized_text), "\n")


freq_dist_uni = nltk.FreqDist(tokenized_text)
print("Most common 10 unigram: ", freq_dist_uni.most_common(10), "\n", "least common 3 words: ",
      freq_dist_uni.most_common()[-3:], "\n")


prob_distArray = []
prob_dist_uni = MLEProbDist(freq_dist_uni)
for s in prob_dist_uni.samples():
    prob_distArray.append(prob_dist_uni.prob(s))
i = 0
for lim in freq_dist_uni.most_common(10):
    print(lim, prob_distArray[i])
    i += 1

elep = ELEProbDist(freq_dist_uni)
for s in elep.samples():
    prob_distArray.append(elep.prob(s))
i = 0
for lim in freq_dist_uni.most_common(10):
    print(lim, prob_distArray[i], "\n")
    i += 1


uniqueWords = len(set(tokenized_text))
print("Unique Words: ", uniqueWords, "\n")


bigram_count = bigrams(tokenized_text)
counts = nltk.Counter(bigram_count)
print("Bigram Count: ", counts, "\n", "Most Common 10 bigram: ", counts.most_common(10), "\n",
      "Least Common 3 words: ", counts.most_common()[-3:], "\n")

word_mapping = dict((w, w) if freq_dist_uni[w] > 1 else (w, 'UNK')
                    for w in tokenized_text)
print("less frequent unigrams to UNK: ", word_mapping)

word_mapping = dict((w, w) if counts[w] > 1 else (w, 'UNK')
                    for w in tokenized_text)
print("less frequent bigrams to UNK: ", word_mapping)
