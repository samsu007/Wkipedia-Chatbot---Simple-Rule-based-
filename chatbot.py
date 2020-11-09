# Importing Required Libraries

import nltk
nltk.download('punkt')
nltk.download('wordnet')
import numpy as np
import random
import string

import bs4 as bs
import urllib.request
import re

# Creating the Corpus



corpustitle = input('Enter your Loved Sports:')

url = 'https://en.wikipedia.org/wiki/{}'.format(corpustitle)

print(url)

# retrieves the wiki article
raw_html = urllib.request.urlopen(url)
raw_html = raw_html.read()


# extract all paragraphs from the article text
article_html = bs.BeautifulSoup(raw_html,"html.parser")
article_paragraphs = article_html.find_all('p')


article_text = ""

# Finally the text is converted into lower case for easier processing
for para in article_paragraphs:
    article_text += para.text

article_text = article_text.lower()

# we need to preprocess our text to remove the special characters and empty spaces from our text
article_text = re.sub(r'\[[0-9]*\]',' ',article_text)
article_text = re.sub(r'\s+',' ',article_text)

# divide our text into senences and words
article_sentences = nltk.sent_tokenize(article_text)
article_words = nltk.word_tokenize(article_text)


# Create helper functions that will remove the punctuation from the user input and will also lemmatize the text

# instantiate the WordNetLemmatizer from the NTLK library.
wnlemmatizer = nltk.stem.WordNetLemmatizer()

# perform_lemmatization, which takes a list of words as input and lemmatize the corresponding lemmatized list of words
def perform_lemmatization(tokens):
    return [wnlemmatizer.lemmatize(token) for token in tokens]

# punctuation_removal list removes the punctuation from the passed text.
punctuation_removal = dict((ord(punctuation),None) for punctuation in string.punctuation)

# get_processed_text method takes a sentence as input, tokenizes it, lemmatizes it, and then removes the punctuation from the sentence.
def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(punctuation_removal)))


greeting_inputs = ("hey", "good morning", "good evening", "morning", "evening", "hi", "whatsup")
greeting_responses = ["hey", "hey hows you?", "*nods*", "hello, how you doing", "hello", "Welcome, I am good and you"]

def generate_greeting_response(greeting):
    for token in greeting.split():
        if token.lower() in greeting_inputs:
            return random.choice(greeting_responses)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def generate_response(user_input):
    Robot_response = ''
    article_sentences.append(user_input)

    word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
    all_word_vectors = word_vectorizer.fit_transform(article_sentences)
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]

    if vector_matched == 0:
        Robot_response = Robot_response + "I am sorry, I could not understand you"
        return Robot_response
    else:
        Robot_response = Robot_response + article_sentences[similar_sentence_number]
        return Robot_response


continue_dialogue = True
print("Hello, I am your friend Robot. You can ask me any question regarding tennis:")
while(continue_dialogue == True):
    human_text = input("me:")
    human_text = human_text.lower()
    if human_text != 'bye':
        if human_text == 'thanks' or human_text == 'thank you very much' or human_text == 'thank you':
            continue_dialogue = False
            print("Robot: Most welcome")
        else:
            if generate_greeting_response(human_text) != None:
                print("Robot: " + generate_greeting_response(human_text))
            else:
                print("Robot: ", end="")
                print(generate_response(human_text))
                article_sentences.remove(human_text)
    else:
        continue_dialogue = False
        print("Robot: Good bye and take care of yourself...")