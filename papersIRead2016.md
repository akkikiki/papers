I need to spend more time to understand and think deeply about these papers.

# Deep Learning
1. [Long Short-Term Meomory Neural Networks, EMNLP 2015](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP141.pdf)
 * Uses 3 NNs 1) character embeddings, 2) LSTM, and 3) tag inference.
 * Takes 3 days to train on PKU dataset using CPU without any optimization.
 * A code implemented by another person is [here](https://github.com/dalstonChen/CWS_LSTM) but I have not checked if this implementation is correct or not.

1. [Structured Training for Neural Network Transition-Based Parsing, ACL 2015](http://www.petrovi.de/data/acl15.pdf)
 * Exploiting unlabeled data
 * “tri-training”: using three different parsers and use the sentences for which two parsers output the same  result (Ambiguity-aware ensemble training for semisupervised dependency parsing, ACL 2014)
 * NN-based parsing compared to other approaches

1. [Semi-supervised Sequence Learning, NIPS 2015](https://papers.nips.cc/paper/5949-semi-supervised-sequence-learning.pdf)
 * Pretraining using a language model and a sequence autoencoder
 * A sequence of autoencoder is unsupervised

1. [Pointing the Unknown Words, ACL 2016](https://arxiv.org/pdf/1603.08148.pdf)
 * Dealing with unknown/rare words.
 * "implement the copy-mechanism" ([A Neural Knowledge Language Model, (submitted to?)NIPS 2016](http://arxiv.org/pdf/1608.00318v1.pdf))

1. [BlackOut: Speeding up Recurrent Neural Network Language Models With Very Large Vocabularies, ICLR 2016](http://arxiv.org/pdf/1511.06909v7.pdf)
 * Similar to negative sampling

1. [Generating Images from Captions with Attention, ICLR 2016](https://arxiv.org/pdf/1511.02793v2.pdf)

# Autoencoders
1. [Pixel Recurrent Neural Networks, ICML 2016](https://arxiv.org/pdf/1601.06759v2.pdf)
 * [Slides in Japanese](http://www.slideshare.net/beam2d/pixel-recurrent-neural-networks) created by Seiya Tokui
 * "to cast it as a product of conditional distributions": autoregressive models 

# Word/Character Embeddings
1. [Charagram: Embedding Words and Sentences via Character n-grams, EMNLP 2016](https://arxiv.org/pdf/1607.02789v1.pdf)
 * Comment: Does it still uses the distributional hypothesis?

1. [Tweet2Vec: Character-Based Distributed Representations for Social Media, ACL 2016](http://aclweb.org/anthology/P/P16/P16-2044.pdf)
 * Very interesting.
 * Uses bidirectional GRU
 * Dealing with rare/unknown words by character n-grams

1. [Enriching Word Vectors with Subword Information, ArXiv preprint 2016](https://arxiv.org/pdf/1607.04606v1.pdf)
 * Its implementation can be found at [fastText](https://github.com/facebookresearch/fastText)

1. [Character-Aware Neural Language Models, AAAI 2016](https://arxiv.org/abs/1508.06615)

1. [RAND-WALK: A Latent Variable Model Approach to Word Embeddings, TACL 2016](https://arxiv.org/abs/1502.03520)
 * Proposes a new generative model

1. [Multi-Task Cross-Lingual Sequence Tagging from Scratch, arXiv 2016](http://arxiv.org/pdf/1603.06270v2.pdf)
 * Also mentioned in [Learning Deep Generative Models @ Deep Learning Summer School](http://www.cs.toronto.edu/~rsalakhu/talk_Montreal_2016_Salakhutdinov.pdf)
 * "Using both word-level and character-level RNNs"

# Topic Models
1. [Sparse Additive Generative Models of Text, ICML 2011](http://machinelearning.wustl.edu/mlpapers/paper_files/ICML2011Eisenstein_534.pdf)
 * Require less variables. (E.g. switching variables in topic-perspective models in Ahmed & Xing 2010)

1. [Comparing Apples to Apple: The Effects of Stemmers on Topic Models, TACL 2016](http://aclweb.org/anthology/Q/Q16/Q16-1021.pdf)
 * rule-based stemmers and context-based methods
 * "Topic modeling is sensitive to preprocessing because of its dependence on a sparse vocabulary (Jockers and Mimno, 2013)"
 * LDA is in question and no other topic models are out of scope

1. [Multilingual Topic Models for Unaligned Text, UAI 2009](http://www.auai.org/uai2009/papers/UAI2009_0194_e9b915894f2228eb675c97f199bebe6d.pdf)
 * word level matching "from non-parallel but comparable corpora".

1. [Collective supervision of topic models for predicting surveys with social media](http://cmci.colorado.edu/~mpaul/files/aaai16_collective.pdf)
 * ``We define collective supervision as supervision in which labels are provided for groups or collections of documents''
 * What does it mean by ``adaptive'' version?
 * Upstream supervision: ``supervision influences the priors over topic distributions in documents''
 * assuming a lack of observed values in data (adaptive version).
 * ``SPRITE (Paul and Dredze 2015), which extends DMR to use log-linear priors in various ways''

# Topic Models and Word Embeddings
1. [Nonparametric Spherical Topic Modeling with Word Embeddings, ACL 2016](http://aclweb.org/anthology/P/P16/P16-2087.pdf)
 * Cosine similarity and von Mises-Fisher distribution (Incorporates directional statistics)

2. [A Latent Concept Topic Model for Robust Topic Inference Using Word Embeddings](http://aclweb.org/anthology/P/P16/P16-2062.pdf)

# Curriculum Learning
1. [Learning the Curriculum with Bayesian Optimization for Task-Specific Word Representation Learning, ACL 2016](http://aclweb.org/anthology/P/P16/P16-1013.pdf)
 * curriculum learning
 * How do you define a “complexity” of training data?
     * length of training sentences for grammar induction ([Spitkovsky et al., NAACL 2010](http://www.aclweb.org/anthology/N/N10/N10-1116.pdf))
     * frequency of a word (Bengio et al., 2009)
 * "we search for an optimal curriculum using Bayesian optimization"
 * Bayesian optimization:
     * approximate objective functions using a "surrogate model"
     * acquisition function
        * estimating the next set of parameters to explore

1. [From Baby Steps to Leapfrog: How “Less is More” in Unsupervised Dependency Parsing, NAACL 2010](http://www.aclweb.org/anthology/N/N10/N10-1116.pdf)


# Short texts
1. [Semi-supervised Clustering for Short Text via Deep Representation Learning, CoNLL 2016](http://aclweb.org/anthology/K/K16/K16-1004.pdf) 
 * Clustering algorithms often suffer from small number of unique words in short text.

1. [Bayesian Supervised Domain Adaptation for Short Text Similarity, NAACL 2016](http://www.umiacs.umd.edu/~jbg/docs/2016_naacl_sts.pdf)
 * Multi-task learning

# Interpretability
1. [“Why Should I Trust You?” Explaining the Predictions of Any Classifier, KDD 2016](http://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf)
 * [video on YouTube](https://www.youtube.com/watch?v=hUnRCxnydCc)
 * Figure 4 is a very good demonstration of the proposed method.
 

# Cross-lingual projection
1. [Learning when to trust distant supervision: An application to low-resource POS tagging using cross-lingual projection, CoNLL 2016](http://aclweb.org/anthology/K/K16/K16-1018.pdf)

1. [Cross-lingual projection for class-based language models, ACL 2016](http://www.aclweb.org/anthology/P/P16/P16-2014.pdf)

1. [Towards cross-lingual distributed representations without parallel text trained with adversarial autoencoders, Proceedings of the 1st Workshop on Representation Learning for NLP 2016](http://aclweb.org/anthology/W/W16/W16-1614.pdf)
 * Why adversarial autoencoders?
 * The goal is to genrate aritificial training data that cannot discriminate between the real training data and the artificial one.

# Others
1. [Generating Natural Language Descriptions for Semantic Representations of Human Brain Activity, ACL 2016](http://aclweb.org/anthology/P/P16/P16-3004.pdf)
 * "Nishida et al.(2015) demonstrated that skip-gram, employed in the framework of word2vec proposed by Mikolov (2013), is a more appropriate model than the conventional statistical models used for the quantitative analysis of semantic representation in human brain activity under the same experimental settings as the prior studies."
 * Caption generation

1. [Mixture Modeling of Individual Learning Curves, EDM 2015](http://www.educationaldatamining.org/EDM2015/uploads/papers/paper_133.pdf)

# Websites/Blogs/Survey Slides
* [2016 Paper Selection](http://anie.me/paper-compose-2016/)
* [Arxiv Sanity Preserver](http://www.arxiv-sanity.com/)
* [Recent Progress in RNN and NLP](http://www.slideshare.net/hytae/recent-progress-in-rnn-and-nlp-63762080)
 * Slides made by Sosuke Kobayashi
* [karpathy's notes on Google's wikireading paper](https://github.com/karpathy/paper-notes/blob/master/wikireading.md)
* [A Beginner's Guide to Variational Methods: Mean-Field Approximation](http://blog.evjang.com/2016/08/variational-bayes.html)
* [How to be a successful PhD](https://people.cs.umass.edu/~wallach/how_to_be_a_successful_phd_student.pdf)
* [Hal's blog post about ACL 2016](http://nlpers.blogspot.com/2016/08/some-papers-i-liked-at-acl-2016.html)
* [NLP Programming Tutorial 7 - Topic Model (In Japanese)](http://www.phontron.com/slides/nlp-programming-ja-07-topic.pdf)
