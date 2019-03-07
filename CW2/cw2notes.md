<script type="text/javascript"
        src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML"></script>

# GAN PDF
#### Up-sampling
page 6-8: moving the data into a larger space to analyse convolution between data points and decoding on the other end.

#### generative models
- Use GAN's to create new data to train with. Train on examples and create more example form the same distribution.
- provide predictions on inputs that are missing data

#### How they work
###### Maximum Likelihood Estimation
- Define a model that provides an estimate of the real probability distribution paramaterised by $\theta$

- Likelihood is the probability that the model assigns to the training data for a dataset containing $m$ training samples $x^{(i)}$ :
\[
\theta^* = argmax_\theta \prod^{m}_{i=1} p_{model}(x^{(i)}; \theta)
\\
= argmax_\theta \sum^{m}_{i=1} log \ p_{model}(x^{(i)}; \theta)
\]

- chooses the parameters for the moel that maximises the Likelihood of the training data.

#### Deep Generative Models

1) **Explicit Density Models:**
    - Define an explicit density function $p_{model}(x^{(i)})$.
    - Likelihood maximisation is straight forward: density function is plugged into above eqn and uphill gradient followed. (simple gaussian)

      **Main challenge**:
    - Design a model that can maintain data complexity when generating new data and still be computationally tractable.
        - Tractable explicit models
        - models that admit tractable approximations to $\theta$ and it's gradient

2) **Tractable Explicit Models:**
    ###### Fully Visible Belief Networks (FVBNs)
- Use chain rule of probability, decompse over n-dimensional vector $x$ into a product of 1D PDFs:
\[
p_{model}(x) = \prod^{n}_{i=1} p_{model}(x_i|x_1, ..., x_{i-1})
\]

**Drawbacks**
- $0(n)$ complexity for generating samples, they must be done one at a time.
- computation over each $x_i$ is done by a deep neural net
- Can not be parallelised and require a lot of computation
- One second of audio synthesised in two mins

    ###### Nonlinear Independant Component Analysis (ICA)

    - define continuous, nonlinear tranforms between two spaces
    - given a vector of variables $z$ and *continuous, differeitiable invertable* tranformation $g$.
    - $g(z)$ yeilds a sample from the model in $x$ space so:
    - Density $p_x$ is tractable if $p_z$ and the determinant of the jacobian $g^{-1} is tractable.

\[
p_x - p_z(g^{-1}(x)) \ \bigg| \ det\bigg (\frac{\delta g^{-1}(x)}{\delta x} \bigg) \bigg|
\]

**Drawbacks**
- restrictions on choice of $g$ (needs to be invertible)
- means $x$ and $z$ need to have the same dimensions

2) **Explicit Models Requiring Approximation:**
      ###### Variational Autoencoder (VAE)
      - Deterministic approx to define a lower bound.
      - max $L$ guarantees to obtain at least the value of the log-Likelihood as it does of $L$.
      - can define an $L$ that is computationally tractable even when log-Likelihood is not.
\[
  L(x; \theta) \leq log \ p_{model}(x; \theta)
\]

**Drawbacks**
- When a weak approx posterior distribution or prior distribution is used, the gap between $L$ and true Likelihood means the learned $p_{model} \neq p_{data}$
- Hard to optomise
- Practically get good Likelihood but get shitty quality samples.
      ###### Variational Autoencoder (VAE)

# LINKS
- https://github.com/soumith/ganhacks
- https://medium.com/@jonathan_hui/gan-why-it-is-so-hard-to-train-generative-advisory-networks-819a86b3750b
- https://github.com/znxlwm/tensorflow-MNIST-cGAN-cDCGAN
