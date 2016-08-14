# Prerequisite
* Probability distribution can occur in two ways: 1) Probability density function (continuous) and 2) Probability mass function (discrete)

# Mean-field Variational Inference
* Rewrite statistical inference problem as an optimization poroblem.
* Posterior probability P(Z|X): Given the image, what is the probability that this is a cat?
 * z ~ P(Z|X) can be a binary classifier for a cat vs. non-cat ("encoding")
 * x ~ P(X|Z) can generate images of a cat and a non-cat ("decoding")
 * P(Z): prior probability. e.g., cats exist in 1/3 of all images 
* Let's perform inference on an easy distirbution Q_phi(Z|X), which is close to the actual distribution P.
* Minimizing reverse KL divergence "squeezes" the Q(Z) under P(Z)

# References
* [A Beginner's Guide to Variational Methods: Mean-Field Approximation](http://blog.evjang.com/2016/08/variational-bayes.html)
* [Learning Deep Generative Models @ Deep Learning Summer School](http://www.cs.toronto.edu/~rsalakhu/talk_Montreal_2016_Salakhutdinov.pdf)
