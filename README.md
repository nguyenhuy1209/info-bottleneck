# Implementation of the Information Bottleneck curve

This is an IMCOMPLETE implementation of the Information Bottleneck curve introduced in [1] and [2] (Figure 6). We are implementing the most basic iterative algorithm only (iIB in Section 3.1 of [3])

### Definitions

We define probability distributions as follow:
- P(X): an array of N values, with N is the number of distinct values of X. Each item in the array is the probability of a variable of X.
- P(X|Y): a matrix of shape N-by-M, with N is the number of distinct values of X, M is the number of distinct values of Y. Each value at row `n` and column `m` is `P(X_n | Y_m)`.

### Problem

The value of the KL Divergence of P(y|x) and P(y|t) is resulting in a nan value, which making the calculation of P(T|X) incorrect.

## Reference

[1]: [The information bottleneck method](https://arxiv.org/abs/physics/0004057).

[2]: [Opening the Black Box of Deep Neural Networks via Information](https://arxiv.org/abs/1703.00810).

[3]: [The Information Bottleneck: Theory and Applications](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=f9357064ef06a30f4533901cbc956bb25af646ad).

