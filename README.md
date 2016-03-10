# Potapov Interpolation
Given a meromorphic matrix-valued function $T(z)$ bounded on $mathbf{C}^+$ and unitary for $z \in i \mathbb{R}$, we wish to construct an approximate function $\tilde{T}(z)$ using a zero-pole interpolation procedure based on Blaschke-Potapov factors.

The code in Potapov.py implements the procedure for finding Blaschke-Potapov products to approximate given functions near poles. Please see section 6.2 in our manuscript for details: http://arxiv.org/abs/1510.08942 or http://www.epjquantumtechnology.com/content/3/1/3.

We provide several examples in Examples.py, including those that appear in our paper. 


## State-space representation
Given a rational matrix-valued function, we also construct the matrices ABCD that give the state-space representation of the system.
