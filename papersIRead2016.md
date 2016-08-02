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

# Topic Models
[Sparse Additive Generative Models of Text, ICML 2011](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Eisenstein_534.pdf)
* Require less variables. (E.g. switching variables in topic-perspective models in Ahmed & Xing 2010)

# Websites/Blogs
* [2016 Paper Selection](http://anie.me/paper-compose-2016/)
* [Arxiv Sanity Preserver](http://www.arxiv-sanity.com/)
