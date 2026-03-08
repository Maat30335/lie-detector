# Abstract

Contextual hallucinations — statements unsupported by given context — remain a significant challenge in AI. We demonstrate a practical interpretability insight: a generator-agnostic observer model detects hallucinations via a single forward pass and a linear probe on its residual stream. This probe isolates a single, transferable linear direction separating hallucinated from faithful text, outperforming baselines by 5–27 points and showing robust mid-layer performance across Gemma-2 models (2B→27B). Gradient-times-activation localises this signal to sparse, late-layer MLP activity. Critically, manipulating this direction causally steers generator hallucination rates, proving its actionability. Our results offer novel evidence of internal, low-dimensional hallucination tracking linked to specific MLP sub-circuits, exploitable for detection and mitigation. We release the 2000-example CONTRATALES benchmark for realistic assessment of such solutions.

---

## 3.2 Residual-Stream Linear Probe Methodology

### Evaluation Protocol

Unless otherwise specified, all detection methods, including our proposed probe and the baselines, are evaluated using a logistic regression classifier trained to distinguish between factual and hallucinated continuations. Performance is assessed via 5-fold cross-validation. A fixed random seed was used for data splitting and sampling across all experiments to ensure reproducibility.

### Residual-stream linear probe

Given a frozen observer transformer $\mathcal{F}$ with $L$ decoder blocks and model dimension $d$, let $r_t^{(\ell)} \in \mathbb{R}^d$ denote the post-layer-norm residual stream at token position $t$ after block $\ell$. For each example, we concatenate the source context $X$ and its candidate continuation $Y$, feed the resulting sequence $(x_{0:T-1})$ through $\mathcal{F}$, and identify the index of the final token (typically a full stop) of the last sentence in the continuation, $t^* = T - 1$.

From a specific layer $\ell^*$ (selected via inner-fold validation on the training set), we extract the activation $\mathbf{h} = r_{t^*}^{(\ell^*)}$. A logistic probe, parametrised by weights $\mathbf{w} \in \mathbb{R}^d$ and bias $b \in \mathbb{R}$, then predicts the probability of hallucination:

$$
\hat{y} = \sigma(\mathbf{w}^\top \mathbf{h} + b),
\qquad
\text{where}
\qquad
\sigma(z) = \frac{1}{1 + e^{-z}}.
$$

The probe is trained using binary cross-entropy loss with $L_2$ regularisation on $\mathbf{w}$. At test time, the logit $s = \mathbf{w}^\top \mathbf{h} + b$ serves as the input for a hard decision threshold (typically $s > 0$ for hallucination) and as a continuous hallucination score.




