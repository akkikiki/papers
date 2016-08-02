# Deep Learning
[Long Short-Term Meomory Neural Networks, EMNLP 2015](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP141.pdf)
* Uses 3 NNs 1) character embeddings, 2) LSTM, and 3) tag inference.
* Takes 3 days to train on PKU dataset using CPU without any optimization.
* A code implemented by another person is [here](https://github.com/dalstonChen/CWS_LSTM) but I have not checked if this implementation is correct or not.

[Structured Training for Neural Network Transition-Based Parsing, ACL 2015](http://www.petrovi.de/data/acl15.pdf)
* Exploiting unlabeled data
* “tri-training”: using three different parsers and use the sentences for which two parsers output the same  result (Ambiguity-aware ensemble training for semisupervised dependency parsing, ACL 2014)
* NN-based parsing compared to other approaches

[Semi-supervised Sequence Learning, NIPS 2015](https://papers.nips.cc/paper/5949-semi-supervised-sequence-learning.pdf)
* Pretraining using a language model and a sequence autoencoder
* A sequence of autoencoder is unsupervised

[Pointing the Unknown Words, ACL 2016](https://arxiv.org/pdf/1603.08148.pdf)
* Dealing with unknown/rare words.
* "implement the copy-mechanism" ([A Neural Knowledge Language Model, (submitted to?)NIPS 2016](http://arxiv.org/pdf/1608.00318v1.pdf))

[BlackOut: Speeding up Recurrent Neural Network Language Models With Very Large Vocabularies, ICLR 2016](http://arxiv.org/pdf/1511.06909v7.pdf)
* Similar to negative sampling

# Autoencoders
[Pixel Recurrent Neural Networks, ICML 2016](https://arxiv.org/pdf/1601.06759v2.pdf)
* [Slides in Japanese](http://www.slideshare.net/beam2d/pixel-recurrent-neural-networks) created by Seiya Tokui
* "to cast it as a product of conditional distributions": autoregressive models 

# Word/Character Embeddings
[Charagram: Embedding Words and Sentences via Character n-grams, EMNLP 2016](https://arxiv.org/pdf/1607.02789v1.pdf)
* Comment: Does it still uses the distributional hypothesis?

# Topic Models
[Sparse Additive Generative Models of Text, ICML 2011](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Eisenstein_534.pdf)
* Require less variables. (E.g. switching variables in topic-perspective models in Ahmed & Xing 2010)

# Websites/Blogs/Survey Slides
* [2016 Paper Selection](http://anie.me/paper-compose-2016/)
* [Arxiv Sanity Preserver](http://www.arxiv-sanity.com/)
* [Recent Progress in RNN and NLP](http://www.slideshare.net/hytae/recent-progress-in-rnn-and-nlp-63762080)
