# 24 Getting Started with Natural Language Processing

<br>

## Content 
- **Getting Started with Natural Language Processing**
    - **Getting Started with Natural Language Processing**
        - [Intro to NLP](#intro-to-nlp)
        - [Text Preprocessing](#text-preprocessing)
        - [Parsing Text](#parsing-text)
        - [Language Models: Bag-of-Words](#language-models-bag-of-words)
        - [Language Models: N-Gram and NLM](#language-models-n-gram-and-nlm)
        - [Topic Models](#topic-models)
        - [Text Similarity](#text-similarity)
        - [Language Prediction & Text Generation](#language-prediction--text-generation)
        - [Advanced NLP Topics](#advanced-nlp-topics)
        - [Challenges and Considerations](#challenges-and-considerations)
        - [NLP Review](#nlp-review)

<br>

## Intro to NLP
- Look at the technologies around us:
    - Spellcheck and autocorrect
    - Auto-generated video captions
    - Virtual assistants like Amazon’s Alexa
    - Autocomplete
    - Your news site’s suggested articles
- What do they have in common?

<br>

- All of these handy technologies exist because of ***natural language processing!*** 
- Also known as ***NLP***, the field is at the intersection of linguistics artificial intelligence, and computer science. 
- The goal? Enabling computers to interpret, analyze, and approximate the generation of human languages (like English or Spanish).

<br>

- NLP got its start around 1950 with Alan Turing’s test for artificial intelligence evaluating whether a computer can use language to fool humans into believing it’s human.

<br>

- But approximating human speech is only one of a wide range of applications for NLP! 
- Applications from detecting spam emails or bias in tweets to improving accessibility for people with disabilities all rely heavily on natural language processing techniques.

<br>

- NLP can be conducted in several programming languages. 
- However, Python has some of the most extensive open-source NLP libraries, including the Natural Language Toolkit or ***NLTK***. 
- Because of this, you’ll be using Python to get your first taste of NLP.

<br>

## Text Preprocessing
*"You never know what you have... until you clean your data."*

- Cleaning and preparation are crucial for many tasks, and NLP is no exception. 
- ***Text preprocessing*** is usually the first step you’ll take when faced with an NLP task.
- Without preprocessing, your computer interprets` "the"`, `"The"`, and `"<p>The"` as entirely different words. 
- There is a LOT you can do here, depending on the formatting you need.
- Lucky for you, [Regex](https://en.wikipedia.org/wiki/Regular_expression) and NLTK will do most of it for you! Common tasks include:
    - **Noise removal** — stripping text of formatting (e.g., HTML tags).
    - **Tokenization** — breaking text into individual words.
    - **Normalization** — cleaning text data in any other way:
        - **Stemming** is a blunt axe to chop off word prefixes and suffixes. “booing” and “booed” become “boo”, but “computer” may become “comput” and “are” would remain “are.”
        - **Lemmatization** is a scalpel to bring words down to their root forms. For example, NLTK’s savvy lemmatizer knows “am” and “are” are related to “be.”
        - Other common tasks include lowercasing, stopwords removal, spelling correction, etc.

<br>

## Parsing Text
- You now have a preprocessed, clean list of words. 
- Now what? It may be helpful to know how the words relate to each other and the underlying syntax (grammar). 
- Parsing is an NLP process concerned with segmenting text based on syntax.

<br>

- You probably do not want to be doing any parsing by hand and NLTK has a few tricks up its sleeve to help you out:
    - ***Part-of-speech tagging (POS tagging)*** 
        - identifies parts of speech (verbs, nouns, adjectives, etc.). 
        - NLTK can do it faster (and maybe more accurately) than your grammar teacher.
    - ***Named entity recognition (NER)*** 
        - helps identify the proper nouns (e.g., “Natalia” or “Berlin”) in a text. 
        - This can be a clue as to the topic of the text and NLTK captures many for you.
    - ***Dependency grammar*** 
        - trees help you understand the relationship between the words in a sentence.
        - It can be a tedious task for a human, so the Python library `spaCy` is at your service, even if it isn’t always perfect.

<br>

- In English we leave a lot of ambiguity, so syntax can be tough, even for a computer program. 
- Take a look at the following sentence:

<img src="Images/parsing_syntactic_ambiguity.gif" width=500>

- ***Regex parsing*** 
    - using Python’s `re` library, allows for a bit more nuance. 
    - When coupled with POS tagging, you can identify specific phrase chunks.
    - On its own, it can find you addresses, emails, and many other common patterns within large chunks of text.

<br>

## Language Models: Bag-of-Words
- How can we help a machine make sense of a bunch of word tokens? 
- We can help computers make predictions about language by training a language model on a corpus (a bunch of example text).

<br>

- Language models are probabilistic computer models of language. 
- We build and use these models to figure out the likelihood that a given sound, letter, word, or phrase will be used. 
- Once a model has been trained, it can be tested out on new texts.

<br>

- One of the most common language models is the unigram model, a statistical language model commonly known as bag-of-words. 
- As its name suggests, bag-of-words does not have much order to its chaos! 
- What it does have is a tally count of each instance for each word. 
- Consider the following text example:

<img src="Images/bag-of-words.gif" width=500>

- Provided some initial preprocessing, bag-of-words would result in a mapping like:
    ```py
    {
        "the": 2, 
        "squid": 1, 
        "jump": 1, 
        "out": 1, 
        "of": 1, 
        "suitcase": 1
    }
    ```
- Now look at this sentence and mapping: “Why are your suitcases full of jumping squids?”
    ```py
    {
        "why": 1, 
        "are": 1, 
        "your": 1, 
        "suitcases": 1, 
        "full": 1, 
        "of": 1, 
        "jumping": 1, 
        "squids": 1
    }
    ```
- You can see how even with different word order and sentence structures, “jump,” “squid,” and “suitcase” are shared topics between the two examples.
- Bag-of-words can be an excellent way of looking at language when you want to make predictions concerning topic or sentiment of a text. 
- When grammar and word order are irrelevant, this is probably a good model to use.

<br>

## Language Models: N-Gram and NLM
- For parsing entire phrases or conducting language prediction, you will want to use a model that pays attention to each word’s neighbors. 
- Unlike bag-of-words, the ***n-gram*** model considers a sequence of some number (*n*) units and calculates the probability of each unit in a body of language given the preceding sequence of length *n*. 
- Because of this, *n-gram* probabilities with larger *n* values can be impressive at language prediction.

<br>

- Take a look at our revised squid example: “The squids jumped out of the suitcases. The squids were furious.”
- A bigram model (where *n* is 2) might give us the following count frequencies:
    ```py
    {
        ('', 'the'): 2, 
        ('the', 'squids'): 2, 
        ('squids', 'jumped'): 1, 
        ('jumped', 'out'): 1, 
        ('out', 'of'): 1, 
        ('of', 'the'): 1, 
        ('the', 'suitcases'): 1, 
        ('suitcases', ''): 1, 
        ('squids', 'were'): 1, 
        ('were', 'furious'): 1, 
        ('furious', ''): 1
    }
    ```
- There are a couple problems with the n gram model:
    1. **1.**
        - How can your language model make sense of the sentence “The cat fell asleep in the mailbox” if it’s never seen the word “mailbox” before? 
        - During training, your model will probably come across test words that it has never encountered before (this issue also pertains to bag of words). 
        - A tactic known as *language smoothing* can help adjust probabilities for unknown words, but it isn’t always ideal.
    2. **2.** 
        - For a model that more accurately predicts human language patterns, you want *n* (your sequence length) to be as large as possible. 
        - That way, you will have more natural sounding language, right? 
        - Well, as the sequence length grows, the number of examples of each sequence within your training corpus shrinks. 
        - With too few examples, you won’t have enough data to make many predictions.
- Enter ***neural language models (NLMs)***! 
- Much recent work within NLP has involved developing and training neural networks to approximate the approach our human brains take towards language.
- This deep learning approach allows computers a much more adaptive tack to processing human language. 
- Common NLMs include LSTMs and transformer models.

<br>

- **Example 1:**
    - We have following sentence: *The cat is asleep. The cat purrs.*
    - Then if we use bigram model (*n* = 2), we have following dictionary:
        ```py
        {
            ('the', 'cat'): 2,
            ('cat', 'is'): 1,
            ('is', 'asleep'): 1,
            ('asleep', 'the'): 1,
            ('cat', 'purrs'): 1
        }
        ```	
    - If we want to predict the probability of the word "cat" given "the", we can use the bigram model to calculate the probability as follows:
    $$ P(\text{"cat" | "the"}) = \frac{Frequency(\text{"the", "cat"})}{Frequency(\text{"the"})} = \frac{2}{2} = 1 $$
    - $Frequency(\text{"the", "cat"})$ can be found in the dictionary above.
    - $Frequency(\text{"the"})$ just counts the number of times the word "the" appears in the corpus / text, or you can just create a unigram model (*n* = 1) to calculate the frequency of the word "the".

<br>

- **Example 2:**
    - We are using the same sentence as in example 1, with the same dictionary and bigram model (*n* = 2).
    - If we want to predict the probability of the word "is" given "cat", we can use the bigram model to calculate the probability as follows:
    $$ P(\text{"is" | "cat"}) = \frac{Frequency(\text{"cat", "is"})}{Frequency(\text{"cat"})} = \frac{1}{2} = 0.5 $$


<br>

- **Example 3:**
    - *The cat is asleep. The cat purrs.*
    - We are using the same sentence as in example 1, with the same dictionary but with trigram model (*n* = 3).
    - Then we have following dictionary:
        ```py
        {
            ('the', 'cat', 'is'): 1,
            ('cat', 'is', 'asleep'): 1,
            ('is', 'asleep', 'the'): 1,
            ('asleep', 'the', 'cat'): 1,
            ('the', 'cat', 'purrs'): 1
        }
        ```
    - If we want to predict the probability of the word "asleep" given "cat is", we can use the trigram model to calculate the probability as follows:
    $$ P(\text{"asleep" | "cat", "is"}) = \frac{Frequency(\text{"cat", "is", "asleep"})}{Frequency(\text{"cat", "is"})} = \frac{1}{1} = 1 $$
    - $Frequency(\text{"cat", "is", "asleep"})$ can be found in the dictionary above.
    - $Frequency(\text{"cat", "is"})$ can we just create a dictionary for bigram model (*n* = 2) to calculate the frequency of the words "cat" and "is", or count the number of times the combination "cat is" appears in the corpus / text.

<br>

- **Example 4:**
    - We are using the same sentence as in example 3 and the same dictionary and the same trigram model (*n* = 3).
    - We want to predict the probability of the word "is" given "the cat", we can use the trigram model to calculate the probability as follows:
    $$ P(\text{"is" | "the", "cat"}) = \frac{Frequency(\text{"the", "cat", "is"})}{Frequency(\text{"the", "cat"})} = \frac{1}{2} = 0.5 $$
    - $Frequency(\text{"the", "cat", "is"})$ can be found in the dictionary above.
    - $Frequency(\text{"the", "cat"})$ can we just create a dictionary for bigram model (*n* = 2) to calculate the frequency of the words "the" and "cat", or count the number of times the combination "the cat" appears in the corpus / text.

<br>

## Topic Models
- We’ve touched on the idea of finding topics within a body of language. 
- But what if the text is long and the topics aren’t obvious?

<br>

- ***Topic modeling*** is an area of NLP dedicated to uncovering latent, or hidden, topics within a body of language. 
- For example, one Codecademy curriculum developer used topic modeling to discover patterns within Taylor Swift songs related to love and heartbreak over time.

<br>

- A common technique is to deprioritize the most common words and prioritize less frequently used terms as topics in a process known as ***term frequency-inverse document frequency (tf-idf)***. 
- Say what?! This may sound counter-intuitive at first. 
- Why would you want to give more priority to less-used words? 
- Well, when you’re working with a lot of text, it makes a bit of sense if you don’t want your topics filled with words like “the” and “is.” 
- The Python libraries `gensim` and `sklearn` have modules to handle tf-idf.

<br>

- Whether you use your plain bag of words (which will give you term frequency) or run it through tf-idf, the next step in your topic modeling journey is often ***latent Dirichlet allocation (LDA)***. 
- LDA is a statistical model that takes your documents and determines which words keep popping up together in the same contexts (i.e., documents). 
- We’ll use `sklearn` to tackle this for us.

<br>

- If you have any interest in visualizing your newly minted topics, ***word2vec*** is a great technique to have up your sleeve. 
- word2vec can map out your topic model results spatially as vectors so that similarly used words are closer together. 
- In the case of a language sample consisting of “The squids jumped out of the suitcases. 
- The squids were furious. 
- Why are your suitcases full of jumping squids?”, we might see that “suitcase”, “jump”, and “squid” were words used within similar contexts. 
- This word-to-vector mapping is known as a *word embedding*.

<br>

## Text Similarity
- Most of us have a good autocorrect story. 
- Our phone’s messenger quietly swaps one letter for another as we type and suddenly the meaning of our message has changed (to our horror or pleasure). 
- However, addressing ***text similarity*** — including spelling correction — is a major challenge within natural language processing.

<br>

- Addressing word similarity and misspelling for spellcheck or autocorrect often involves considering the ***Levenshtein distance*** or minimal edit distance between two words. 
- The distance is calculated through the minimum number of insertions, deletions, and substitutions that would need to occur for one word to become another. 
- For example, turning “bees” into “beans” would require one substitution (“a” for “e”) and one insertion (“n”), so the Levenshtein distance would be two.

<br>

- Phonetic similarity is also a major challenge within speech recognition. 
- English-speaking humans can easily tell from context whether someone said “euthanasia” or “youth in Asia,” but it’s a far more challenging task for a machine! 
- More advanced autocorrect and spelling correction technology additionally considers key distance on a keyboard and ***phonetic similarity*** (how much two words or phrases sound the same).

<br>

- It’s also helpful to find out if texts are the same to guard against plagiarism, which we can identify through ***lexical similarity*** (the degree to which texts use the same vocabulary and phrases). 
- Meanwhile, ***semantic similarity*** (the degree to which documents contain similar meaning or topics) is useful when you want to find (or recommend) an article or book similar to one you recently finished.

<br>

## Language Prediction & Text Generation
- How does your favorite search engine complete your search queries? 
- How does your phone’s keyboard know what you want to type next? 
- ***Language prediction*** is an application of NLP concerned with predicting text given preceding text. 
- Autosuggest, autocomplete, and suggested replies are common forms of language prediction.

<br>

- Your first step to language prediction is picking a language model. 
- Bag of words alone is generally not a great model for language prediction; no matter what the preceding word was, you will just get one of the most commonly used words from your training corpus.

<br>

- If you go the *n*-gram route, you will most likely rely on ***Markov chains*** to predict the statistical likelihood of each following word (or character) based on the training corpus. 
- Markov chains are memory-less and make statistical predictions based entirely on the current *n*-gram on hand.

<br>

- For example, let’s take a sentence beginning, “I ate so many grilled cheese”. Using a trigram model (where *n* is 3), a Markov chain would predict the following word as “sandwiches” based on the number of times the sequence “grilled cheese sandwiches” has appeared in the training data out of all the times “grilled cheese” has appeared in the training data.

<br>

- A more advanced approach, using a neural language model, is the Long Short Term Memory (LSTM) model. LSTM uses deep learning with a network of artificial “cells” that manage memory, making them better suited for text prediction than traditional neural networks.

<br>

## Advanced NLP Topics
- Believe it or not, you’ve just scratched the surface of natural language processing. 
- There are a slew of advanced topics and applications of NLP, many of which rely on deep learning and neural networks.

<br>

- ***Naive Bayes classifiers*** are supervised machine learning algorithms that leverage a probabilistic theorem to make predictions and classifications. 
- They are widely used for sentiment analysis (determining whether a given block of language expresses negative or positive feelings) and spam filtering.

<br>

- We’ve made enormous gains in ***machine translation***, but even the most advanced translation software using neural networks and LSTM still has far to go in accurately translating between languages.

<br>

- Some of the most life-altering applications of NLP are focused on improving ***language accessibility*** for people with disabilities. 
- Text-to-speech functionality and speech recognition have improved rapidly thanks to neural language models, making digital spaces far more accessible places.

<br>

- NLP can also be used to detect bias in writing and speech. 
- Feel like a political candidate, book, or news source is biased but can’t put your finger on exactly how? 
- Natural language processing can help you identify the language at issue.

<br>

## Challenges and Considerations
- As you’ve seen, there are a vast array of applications for NLP. 
- However, as they say, “with great language processing comes great responsibility” (or something along those lines). When working with NLP, we have several important considerations to take into account:
    - **Different NLP tasks** 
        - may be more or less difficult in different languages. 
        - Because so many NLP tools are built by and for English speakers, these tools may lag behind in processing other languages. 
        - The tools may also be programmed with cultural and linguistic biases specific to English speakers.
    - **What if your Amazon Alexa could only understand wealthy men from coastal areas of the United States?** 
        - English itself is not a homogeneous body. 
        - English varies by person, by dialect, and by many sociolinguistic factors. 
        - When we build and train NLP tools, are we only building them for one type of English speaker?
    - **You can have the best intentions and still inadvertently program a bigoted tool.** 
        - While NLP can limit bias, it can also propagate bias. 
        - As an NLP developer, it’s important to consider biases, both within your code and within the training corpus. 
        - A machine will learn the same biases you teach it, whether intentionally or unintentionally.
    - **As you become someone who builds tools with natural language processing, it’s vital to take into account your users’ privacy.** 
        - There are many powerful NLP tools that come head-to-head with privacy concerns.
        - Who is collecting your data? How much data is being collected and what do those companies plan to do with your data?

<br>

## NLP Review
- Natural language processing combines computer science, linguistics, and artificial intelligence to enable computers to process human languages.
- NLTK is a Python library used for NLP.
- Text preprocessing is a stage of NLP focused on cleaning and preparing text for other NLP tasks.
- Parsing is an NLP technique concerned with breaking up text based on syntax.
- Language models are probabilistic machine models of language use for NLP comprehension tasks. Common models include bag-of-words, n-gram models, and neural language modeling.
- Topic modeling is the NLP process by which hidden topics are identified given a body of text.
- Text similarity is a facet of NLP concerned with semblance between instances of language.
- Language prediction is an application of NLP concerned with predicting language given preceding language.
- There are many social and ethical considerations to take into account when designing NLP tools.