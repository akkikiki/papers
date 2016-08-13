# Mean-field Variational Inference
* Rewrite statistical inference problem as an optimization poroblem.
* Posterior probability P(Z|X): Given the image, what is the probability that this is a cat?
 * z ~ P(Z|X) can be a binary classifier for a cat vs. non-cat ("encoding")
 * x ~ P(X|Z) can generate images of a cat and a non-cat ("decoding")
 * P(Z): prior probability. e.g., cats exist in 1/3 of all images 

# References
* [A Beginner's Guide to Variational Methods: Mean-Field Approximation](http://blog.evjang.com/2016/08/variational-bayes.html)
