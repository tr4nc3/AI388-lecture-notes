

# Comprehensive Reference Guide for Natural Language Processing Examination

## Part I. Foundations and Lexical Classification

### Chapter 1: Goals, Representations, and Feature Engineering

#### 1.1 The Standard NLP Pipeline and Objectives

The overarching goal of Natural Language Processing (NLP) is to construct systems capable of exhibiting a profound comprehension of human text, enabling robust interaction and effective information processing.[1] Core applications include dialogue systems (e.g., Siri), automated machine translation, document summarization, information extraction, and question answering systems.[1]

The fundamental operational structure is the NLP pipeline, where raw text undergoes a series of text analysis components to produce structural annotations. These annotations—such as syntactic parses, coreference resolution, entity disambiguation, and discourse analysis—are then fed into downstream applications for tasks like sentiment identification or information extraction.[1] A crucial aspect underlying all these components is their reliance on statistical approaches utilizing machine learning to model complex linguistic phenomena.[1]

The primary challenge encountered in this process is the inherent complexity and ambiguity of human language. Ambiguities manifest on multiple levels, ranging from word sense disambiguation (polysemy) to structural ambiguities, such as Prepositional Phrase (PP) attachment ambiguities (e.g., "I eat cake with icing").[1] Statistical models address this difficulty by learning probability distributions that map text representations to linguistic structures, allowing the system to resolve multiple possibilities based on data evidence.[1]

#### 1.2 Text Representation and Feature Extraction

To make text machine-tractable, it must be converted into numerical feature vectors $f(\overline{x})$. The foundational method is the Bag-of-Words (BoW) model, which represents a text as a vector recording the count or binary presence (0/1) of words within a predefined vocabulary (which may contain up to 18,000 unique words or more).[1]

Feature richness can be augmented using N-gram features, which capture sequences of $n$ consecutive words (e.g., "the movie," "movie was"). Bigrams are essential for capturing local word order dependencies and handling crucial negation structures (e.g., extracting the feature "not good") that simple unigram BoW models often miss.[1] For feature weighting, TF-IDF is commonly employed, which weights the term frequency ($tf$) by its inverse document frequency ($idf$), calculated as $\log \frac{N}{\{D: w \in D\}}$, where $N$ is the total number of documents and the denominator is the count of documents containing word $w$. This emphasizes terms highly relevant to a specific document rather than those universally common (e.g., stopwords).[1]

Preprocessing is necessary before feature extraction:
1.  **Tokenization:** Breaking text into discrete units (tokens), which must correctly handle contractions (e.g., "wasn't" $\rightarrow$ "was," "n't") and punctuation.[1]
2.  **Normalization:** Includes lowercasing, truecasing, optional stopword removal, and handling out-of-vocabulary words (UNK).[1]

An analysis of performance metrics reveals an important distinction concerning the complexity of the feature set required. For basic sentiment analysis tasks, relatively simple features—such as unigrams coupled with presence/absence scoring—can achieve highly competitive results, with certain configurations yielding accuracy around $82.9\%$.[1] However, attempting to solve tasks that demand deep semantic understanding or cross-sentence alignment, such as Textual Entailment or Entity Disambiguation (which may involve classifying entities into $4.5$ million possible classes, like Wikipedia articles), mandates a radical departure from simple N-gram representations.[1] This suggests that while linear models utilizing basic N-gram features provide strong, interpretable baselines for many categorization problems, transitioning to sophisticated linguistic tasks requires rich, structured, or learned contextual representations (such as those provided by modern deep neural networks) to capture the necessary hierarchical and relational meanings.[1]

#### 1.3 Classification Applications and Fairness

Classification principles are applied to various NLP tasks, including Authorship Attribution. Early methods relied on handcrafted heuristics like stopword frequencies, but modern approaches employ complex feature sets, such as character 4-grams and word 2-grams through 5-grams, which are fed into a Support Vector Machine (SVM) classifier.[1] The effectiveness of these models is often attributed to the capture of unique linguistic markers, sometimes called "k-signatures"—N-grams that appear only in a specific percentage $k$ of one author’s texts but not in others.[1]

A critical consideration for deploying classifiers in the real world is fairness, which must be evaluated beyond simple accuracy metrics.[1] Several criteria for evaluating fairness have been proposed:
*   **Cleary Criterion (1966-1968):** States that a test is biased if prediction errors consistently result in non-zero errors for a specific subgroup ($\pi_2$) when compared to the aggregate population.[1]
*   **Thorndike/Petersen/Novik Criterion (1971, 1976):** Stipulates that for fairness, the ratio of predicted positives to ground truth positives must be approximately consistent across all demographic groups.[1] A classifier that predicts $50\%$ positive reviews for Group 1 (actual $50\%$ positive) and $50\%$ positive reviews for Group 2 (actual $60\%$ positive) is considered unfair, even if its overall accuracy appears high.[1]

The difficulty in ensuring fairness lies in the fact that classifiers can discriminate unintentionally. Features that seem neutral, such as ZIP codes or specific Bag-of-Words features highly correlated with dialects (e.g., AAVE or code-switching), can act as proxies for protected classes, leading to biased outcomes.[1] For instance, certain features associated with women’s colleges or organizations were observed to carry negative weights in a defunct AI recruiting tool, illustrating that high accuracy alone does not guarantee non-discrimination.[1] Therefore, system designers must proactively identify minority groups that could be overlooked and rigorously check the system's features for potential correlations with protected attributes.[1]

### Chapter 2: Linear Models and Optimization Principles

#### 2.1 Linear Binary Classification: Perceptron

The Perceptron is a fundamental linear binary classifier designed to separate input data $\overline{x} \in \mathbb{R}^d$ into two classes, $y \in \{-1, +1\}$.[1] The classification decision is based on whether the weighted sum of features $\overline{w}^T f(\overline{x})$ exceeds a threshold (usually absorbed into a bias term $b$).[1, 2] The decision rule is $y_{pred} = \text{sign}(\overline{w}^T f(\overline{x}))$.[1, 2]

Training relies on a mistake-driven learning approach using Stochastic Gradient Descent (SGD) with a learning rate $\alpha$. The weights are updated only when a misclassification occurs. For a misclassified example $(\overline{x}^{(i)}, y^{(i)})$, the update rule is:
$$\overline{w} \leftarrow \overline{w} + \alpha f(\overline{x}^{(i)}) \quad \text{if } y^{(i)}=+1$$
$$\overline{w} \leftarrow \overline{w} - \alpha f(\overline{x}^{(i)}) \quad \text{if } y^{(i)}=-1$$
This update is equivalent to taking a step proportional to the subgradient of the Perceptron loss function, $L(\overline{w}) = \max(0, -y^{(i)} \overline{w}^T f(\overline{x}^{(i)}))$.[2, 3] A critical theoretical property of the Perceptron is that its convergence is only guaranteed if the training data is perfectly linearly separable.[3]

#### 2.2 Logistic Regression (LR) and Probabilistic Output

Logistic Regression (LR) is a discriminative probabilistic model that directly estimates the class probability $P(y|\overline{x})$.[1] It utilizes the logistic (or sigmoid) function to map the linear score $\overline{w}^T f(\overline{x})$ to a probability in $(0, 1)$:
$$P(y=+1|\overline{x})=\frac{e^{\overline{w}^{T}f(\overline{x})}}{1+e^{\overline{w}^{T}f(\overline{x})}}$$
The decision boundary remains the same as the Perceptron's, defined by $\overline{w}^T f(\overline{x}) > 0$ (where $P(y=+1|\overline{x}) > 0.5$).[1]

Training LR involves maximizing the likelihood of the observed dataset, which is equivalent to minimizing the Negative Log-Likelihood (NLL) loss, a convex and differentiable objective function [1, 1]. The NLL is given by $\sum_{i=1}^{D} -\log P(y^{(i)}|\overline{x}^{(i)})$. The differentiability of this loss allows for training via standard gradient descent methods.

The mechanism of the LR gradient update provides improved stability compared to the Perceptron. For a positive example ($y^{(i)}=+1$), the gradient is derived as:
$$\frac{\partial L}{\partial \overline{w}} = f(\overline{x})[P(y=+1|\overline{x}) - 1]$$
The resulting weight update step is proportional to the error term $(1 - P(y=+1|\overline{x}))$.[1] This means that if the model is highly confident and correct ($P(y) \approx 1$), the update magnitude approaches zero, leading to convergence stability. Conversely, the Perceptron always applies an update of constant magnitude $\alpha f(\overline{x})$ upon a mistake, regardless of how close the current predicted score was to the decision boundary.[1] This difference in optimization means that Logistic Regression is more robust when dealing with non-separable data and generally reaches a global minimum loss, whereas the Perceptron only converges to a functional decision boundary if the data is perfectly separable.

Table I summarizes the core properties and trade-offs of these two linear classifiers.

Table I: Linear Classifier Properties and Tradeoffs

| Classifier | Objective Function | Output Type | Loss Property | Key Update Factor | Convergence Guarantee |
|---|---|---|---|---|---|
| Perceptron | $L = \max(0, -y \overline{w}^T f(\overline{x}))$ | Binary ($\pm 1$) | Non-differentiable (Subgradient) | $\pm \alpha f(\overline{x})$ (Constant magnitude) | Yes, if linearly separable.[4] |
| Logistic Regression | $L = -\log P(y|\overline{x})$ (NLL) | Probabilistic $(0, 1)$ | Convex, Differentiable | Proportional to error magnitude ($1-P(y|\overline{x})$).[1] | No, but reaches minimum loss/maximum likelihood. |

#### 2.3 Multiclass Classification and Softmax

Multiclass classification problems, such as topic classification (e.g., classifying text into categories like Health or Sports), require an extension of the binary linear models to output probabilities over a set of classes $\mathcal{Y}=\{1, 2, 3, \ldots\}$.[1]

Two primary feature/weight architectures exist:
1.  **Different Weights (DW):** Assigns a separate weight vector $\overline{w}_y$ for each class $y$. The prediction is made by selecting the class with the highest score: $\hat{y} = \arg\max_{y} \overline{w}_y^T f(\overline{x})$.[1]
2.  **Different Features (DF):** Uses a single weight vector $\overline{w}$ operating on an augmented feature vector $f(\overline{x}, y)$, which includes features specific to the hypothesized class $y$.[1]

For multiclass LR, the probabilities are normalized across all classes using the **Softmax** function:
$$P(y=\hat{y}|\overline{x})=\frac{e^{\overline{w}^{T}f(\overline{x},\hat{y})}}{\sum_{y^{\prime}\in \mathcal{Y}}e^{\overline{w}^{T}f(\overline{x},y^{\prime})}}$$
The Softmax operation ensures that the output probabilities sum to 1 over all possible classes $\mathcal{Y}$.[1]

The Multiclass Perceptron update rule, typically using the DF approach, adjusts weights when the predicted class $y_{pred}$ is incorrect. It attempts to increase the score of the correct class $y^{(i)}$ while simultaneously decreasing the score of the incorrect prediction $y_{pred}$:
$$\overline{w} \leftarrow \overline{w} + \alpha f(\overline{x}, y^{(i)}) - \alpha f(\overline{x}, y_{pred})$$
This is generalized in Multiclass LR, where the gradient involves subtracting the correct class features $f(\overline{x}, y^{(i)})$ and adding an expectation over all features weighted by their predicted probabilities $\sum_{y^{\prime}\in \mathcal{Y}} P(y^{\prime}|\overline{x})f(\overline{x}, y^{\prime})$.[1]

#### 2.4 Optimization Techniques and Practices

Optimization involves searching for the optimal parameter vector $\overline{w}$ that minimizes the objective loss function $\mathcal{L}(\overline{w})$.[1] Stochastic Gradient Descent (SGD) is the standard method, iteratively updating the weights based on the gradient computed from a single example or a small batch:
$$\overline{w}\leftarrow\overline{w}-\alpha\frac{\partial}{\partial\overline{w}}\mathcal{L}(i,\overline{w})$$
where $\alpha$ is the step size.[1]

The choice of step size $\alpha$ is crucial. A value that is too large can cause the algorithm to oscillate around the minimum, failing to converge, while a value that is too small results in slow progress.[1] Standard practice often involves using a fixed schedule, such as decreasing the learning rate over time (e.g., $\alpha=1/t$ for epoch $t$), or decreasing it only when performance on held-out development data stagnates.[1]

Modern optimization relies heavily on adaptive methods like Adam, Adagrad, and Adadelta.[1] These techniques calculate a unique, per-parameter learning rate based on historical gradient information. They function as efficient approximations of Newton’s method, which is computationally infeasible for high-dimensional deep learning problems because it requires calculating and inverting the Hessian matrix ($\frac{\partial^2}{\partial\overline{w}}h$).[1] While adaptive methods often accelerate convergence, analysis suggests that meticulously tuned SGD sometimes achieves better generalization performance on the final test set.[1]

### Chapter 3: Word Embeddings and Contextual Vectors

#### 3.1 Distributional Semantics and Word2Vec

Word embeddings represent words as low-dimensional, continuous vectors (typically 50 to 300 dimensions) that capture semantic similarity.[1] This approach is rooted in the distributional hypothesis, formulated by J.R. Firth in 1957, which posits that a word's meaning is derived from the context in which it appears ("You shall know a word by the company it keeps").[1]

Word2Vec, developed by Mikolov et al. in 2013, introduced efficient neural architectures to learn these vectors by predicting word-context co-occurrence relations within large text corpora.[1, 5]

#### 3.2 Skip-Gram vs. CBOW

Word2Vec models operate under two primary architectural paradigms, which are essentially mirrored prediction tasks [1, 6]:

*   **Continuous Bag-of-Words (CBOW):** The model takes the sum or average of context word embeddings (within a fixed window size $K$) to predict the center (target) word.[1]
*   **Skip-Gram (SG):** The model uses the center word vector $\overline{v}_x$ to predict each individual context word $\overline{c}_y$ within the window $K$.[1] The probability of predicting a context word $y$ given the word $x$ is defined using the dot product similarity and Softmax normalization:
    $$P(context=y|word=x)=\frac{\exp(\overline{v}_{x}\cdot \overline{c}_{y})}{\sum_{y^{\prime}\in \mathcal{V}}\exp(\overline{v}_{x}\cdot \overline{c}_{y^{\prime}})}$$

The choice between the two architectures depends on the priorities of the task, as outlined in Table II [5, 7]: SG is generally better at capturing nuanced semantic relationships, particularly for rare words, because it trains multiple pairs for each context window. CBOW is generally faster to train and performs better for high-frequency words and syntactic relationships because it only makes one prediction per context window.[6]

Table II: Word Embedding Model Comparison

| Model | Prediction Direction | Primary Strength | Primary Weakness/Cost | Efficiency Technique |
|---|---|---|---|---|
| CBOW | Context $\rightarrow$ Target Word | Faster training; frequent words/syntax.[5] | Averages context; struggles with rare words.[8] | N/A (often uses HS/NS) |
| Skip-Gram (SG) | Target Word $\rightarrow$ Context Words | Better for rare words; semantics.[6] | Slower training (multiple predictions); initial $O(\|\mathcal{V}\|d)$ cost. | Negative Sampling or Hierarchical Softmax.[1] |
| GloVe | Matrix Factorization | Global context, highly effective representations.[1] | Requires full pre-computation of co-occurrence matrix. | N/A |

#### 3.3 Efficiency Techniques

A major computational bottleneck for both CBOW and Skip-Gram is the necessity of calculating the Softmax denominator, which requires a sum over the entire vocabulary $\mathcal{V}$ for every prediction, resulting in an $O(|\mathcal{V}|d)$ complexity per training step, where $d$ is the embedding dimension [1, 1].

To circumvent this cost, two primary optimization techniques are used:
1.  **Hierarchical Softmax (HS):** Replaces the flat Softmax layer with a binary tree structure, typically organized via Huffman encoding. Probability calculation involves navigating this tree, reducing the complexity from $|\mathcal{V}|$ dot products to $O(\log |\mathcal{V}|)$ sequential decisions.[1]
2.  **Skip-Gram with Negative Sampling (SGNS):** Replaces the full prediction task with a binary classification task. The model is trained to classify observed (word, context) pairs as "real" (+1) while discriminating against $k$ randomly sampled "negative" (fake) pairs.
    $$P(y=1|w,c)=\frac{e^{w \cdot c}}{e^{w \cdot c}+1}$$
    The training objective maximizes the likelihood of real pairs and minimizes the likelihood of negative pairs.[1] This technique reduces the complexity per step significantly to $O(k \cdot d)$, independent of the total vocabulary size $|\mathcal{V}|$.[1] SGNS is mathematically proven to be equivalent to factoring a word-context co-occurrence matrix adjusted by Pointwise Mutual Information (PMI).[1]

**GloVe (Global Vectors):** An alternative, non-predictive approach that directly leverages global co-occurrence statistics. GloVe minimizes a weighted least-squares objective on the logarithm of the co-occurrence counts, thereby directly factoring the resulting co-occurrence matrix.[1] This method is computationally efficient because it is constant relative to the dataset size (only requiring counts) and quadratic in the vocabulary size, making it one of the most widely used methods today.[1]

**fastText:** An extension of SGNS that addresses morphological variation and Out-of-Vocabulary (OOV) words. Words are represented not by a single vector, but by the sum of their constituent character n-gram vectors (e.g., 3-grams to 6-grams), allowing the model to generalize based on shared subword structure.[1]

#### 3.4 Bias and Debiasing in Embeddings

Word embeddings, trained on vast corpora reflecting real-world discourse, inherently encode societal biases.[1] For example, studies have shown correlations between vector differences and gender, such that projections along the "she-he" axis reveal clusters for occupations like "homemaker" (extreme she) versus "maestro" (extreme he).[1]

The geometric projection method (Bolukbasi et al.) attempts to neutralize this bias by identifying a gender subspace, usually defined by the difference vector between gendered word pairs (e.g., man-woman). The technique projects the vector of a target word (e.g., "homemaker") onto this subspace and then subtracts that component, moving the resulting vector ("homemaker'") closer to the neutral plane [1, 1].

However, the efficacy of this simple geometric debiasing technique is limited. Analysis has shown that removing the directional component is often insufficient because bias is not restricted to a single axis but is deeply infused throughout the entire embedding space structure.[1] Consequently, simple neutralization does not necessarily prevent traditionally male- or female-associated words from remaining clustered together, suggesting that more complex, distributed interventions are required to ensure genuine neutrality.[1]

## Part II. Structured Prediction and Syntactic Analysis

### Chapter 4: Structured Prediction for Sequence Labeling

#### 4.1 The Tagging Problem and Feature-Based Approaches

Sequence labeling involves mapping an input word sequence $\overline{X}=(x_1, \ldots, x_n)$ to an output tag sequence $\overline{y}=(y_1, \ldots, y_n)$, where a prediction is made for every input word.[1] Part-of-Speech (POS) tagging, which uses a tagset of around 44 tags (e.g., the Penn Treebank tags for open and closed classes), is the canonical sequence labeling problem.[1]

Local classifiers attempt to predict each tag $y_i$ independently using features centered at word $x_i$, including the word itself, its POS, and context derived from surrounding words (e.g., previous word, next word). These models often rely on complex feature conjunctions, such as indicator functions that activate if $\text{current word} = \text{'interest'} \wedge \text{tag} = \text{NN}$.[1]

A major shortcoming of independent classification, however, is the production of linguistically incoherent sequences where predicted tag transitions violate grammatical constraints (e.g., predicting an ungrammatical sequence of tags like VBZ $\rightarrow$ NNS $\rightarrow$ NN). This necessitates the use of structured prediction models that incorporate dependencies between adjacent tags.[1]

#### 4.2 Hidden Markov Models (HMMs) and Inference

Hidden Markov Models (HMMs) provide a generative sequence model used for tagging. HMMs factor the joint probability $P(\overline{x}, \overline{y})$ based on two core conditional independence assumptions [1]:
1.  **First-Order Markov Property (Transition):** The probability of the current tag $P(y_i|y_{i-1})$ depends only on the immediate preceding tag.
2.  **Emission Independence:** The probability of observing the current word $P(x_i|y_i)$ depends only on the current tag.
The joint probability is thus decomposed as:
$$P(\overline{x}, \overline{y}) = P(y_1) P(x_1|y_1) \prod_{i=2}^{n} P(y_i|y_{i-1}) P(x_i|y_i)$$
Parameter estimation for HMMs is straightforward, often achieved via Maximum Likelihood Estimation (MLE) through normalized frequency counts from labeled data.[1]

**The Viterbi Algorithm:** This dynamic programming technique is used for inference, specifically to find the single most likely tag sequence $\hat{\overline{y}}$ given the observations $\overline{x}$.[1, 9] The recurrence relation computes $\nu_i(y)$, the score of the best path ending in tag $y$ at time $i$:
$$\nu_i(y) = \log P(x_i|y) + \max_{y_{prev} \in \mathcal{T}} [\log P(y|y_{prev}) + \nu_{i-1}(y_{prev})]$$
The maximum operation ($\max$) distinguishes Viterbi from the Forward algorithm (which uses $\sum$ and computes the total probability of $\overline{x}$).[1, 10] The computational complexity of Viterbi is $O(nk^2)$, where $n$ is the sequence length and $k$ is the number of possible tags. The quadratic term arises from evaluating all possible $k \times k$ transitions at each timestep.[1]

When comparing HMMs with modern neural taggers, a critical difference emerges in how features are employed. In HMMs, the model strictly separates the probabilities governing tag sequences (transitions) from the probabilities governing word choice given a tag (emissions).[1] This contrasts sharply with modern Feedforward Networks (FFNNs) applied to tagging, where the hidden layers learn dense, non-linear mixing functions that can implicitly capture complex conjunctions of all input features simultaneously (e.g., combining information about the current word, previous word, and hypothesized tag transitions).[1] This inherent capability of neural models to learn complex, non-linear feature interactions allows them to surpass the performance ceilings imposed by the strict independence assumptions of HMMs.

### Chapter 5: Formal Grammars and Constituency Parsing

#### 5.1 Constituency Trees and PCFGs

Constituency parsing generates hierarchical, phrase-structure trees that decompose sentences into constituents like Sentence (S), Noun Phrase (NP), Verb Phrase (VP), etc..[1] These trees capture key linguistic structures, such as verb-argument relations, and can help resolve lexical ambiguity (e.g., deciding if "raises" is a verb or a plural noun).[1]

A Probabilistic Context-Free Grammar (PCFG) extends a basic Context-Free Grammar (CFG) by assigning a probability $P(R | \text{Parent})$ to each rule $R$ rooted at a specific non-terminal parent symbol.[1] The probability of a complete parse tree $T$ is the product of the probabilities of all rules used in its derivation.[1]

#### 5.2 The CKY Algorithm

The CKY (Cocke-Kasami-Younger) algorithm is a dynamic programming approach used for parsing sentences according to a PCFG. A prerequisite for CKY is that the grammar rules must be in Chomsky Normal Form (CNF), meaning all internal rules must be binary ($X \rightarrow YZ$) and all pre-terminal rules must be unary ($X \rightarrow w$).[1] Practical grammars often require lossless binarization to meet this requirement.[1]

CKY uses a chart $t[i, j, X]$ to store the probability (or log probability, for sums) of the best parse of non-terminal $X$ spanning the segment of words from index $i$ to $j$.[1] The runtime complexity of the CKY algorithm is $O(n^3 G)$, where $n$ is the sentence length and $G$ is the size of the grammar.[1] The cubic dependence on $n$ stems from the iterative procedure of considering all possible starting points $i$, ending points $j$, and internal split points $k$ within the span $(i, j)$.[1]

#### 5.3 Context Enhancement Techniques

A fundamental limitation of PCFGs is the independence assumption, which treats all instances of a non-terminal (e.g., all NPs) as identical, regardless of their surrounding context. This leads to inaccurate probability modeling, as NPs under an S node behave differently than NPs under a VP node.[1]

To improve contextual sensitivity, three main techniques are applied:
1.  **Vertical Markovization (Parent Annotation):** Augments each non-terminal symbol $X$ with its parent category, creating new symbols like $X^{\text{Parent}}$.[1] This allows the model to differentiate rules based on the immediate structural context above them. This operation inherently makes the grammar rules more restrictive, meaning the set of valid sentences generated by the Markovized grammar is a **subset** of the language generated by the original grammar.[1]
2.  **Horizontal Markovization (History Annotation):** Annotates a non-terminal expansion with the history of symbols already generated on the left-hand side of the rule, thereby retaining context within the expansion.[1]
3.  **Lexicalization (Head Words):** This is the most crucial enhancement, achieved by annotating each phrasal node with its "head word"—the most important word within that constituent (e.g., annotating $NP(\text{dog})$). Lexicalized grammars (like those used by Collins and Charniak in the late 1990s) are necessary to capture specific lexical dependencies, which are vital for resolving persistent structural ambiguities such as Prepositional Phrase (PP) attachment (e.g., distinguishing "I saw the man [with a telescope]" from "I saw the man [with a beard]").[1]

Table IV summarizes the effects of structural modifications in constituency parsing.

Table IV: Constituency Parsing Adjustments

| Technique | Rule Annotation | Goal/Linguistic Effect | Subset/Superset? |
|---|---|---|---|
| Vertical Markovization | $X^{\text{Parent}}$ | Constrain rule application based on parent category. | Subset (more restrictive).[1] |
| Lexicalization | $X(\text{Head Word})$ | Model lexical dependencies (e.g., PP attachment). | Greatly increases grammar size. |

### Chapter 6: Dependency Parsing and Transition Models

#### 6.1 Dependency Syntax and Structure

Dependency parsing offers an alternative syntactic representation where structure is defined by directed, asymmetrical arcs between a head (governor) and a dependent (modifier).[1] Every word in the sentence, except the virtual ROOT node, must have exactly one head. Dependencies can be labeled according to syntactic function (e.g., $nsubj$ for nominal subject, $det$ for determiner).[1]

Dependency formalisms are often preferred for languages with highly flexible word order because they are more portable cross-lingually (Universal Dependencies, UD) and do not impose the rigid phrase constraints necessary for constituency grammars.[1] A critical concept is **projectivity**: a dependency tree is projective if it can be drawn without any crossing arcs.[1] Non-projective structures, common in languages like Swiss German, represent discontiguous dependencies that constituency parsers cannot easily handle.[1]

#### 6.2 Transition-Based (Shift-Reduce) Parsing

Transition-based parsing (also known as shift-reduce parsing) offers an efficient, alternative method to the chart-based dynamic programming approach.[1] It builds the dependency tree incrementally by applying a sequence of local decisions (transitions).

The state of the parser is defined by two data structures: the **Stack** (containing words that have been processed and are waiting for potential right dependents) and the **Buffer** (containing the remaining input sentence).[1]

The Arc-Standard system uses three core transitions [1]:
1.  **Shift (S):** Moves the next word from the Buffer onto the Stack.
2.  **Left-Arc (LA):** Creates a dependency arc $w_2 \rightarrow w_1$ (where $w_1$ is the dependent child and $w_2$ is the head parent) between the top two items on the stack ($\dots w_2, w_1$). The dependent $w_1$ is popped.
3.  **Right-Arc (RA):** Creates a dependency arc $w_1 \rightarrow w_2$. The dependent $w_2$ is popped.

Since a sentence of length $n$ requires approximately $2n$ transitions, transition-based parsing achieves a highly efficient $O(n)$ time complexity.[1] Training is performed by building an oracle (which translates a gold tree into a sequence of correct transitions) and using the resulting state-transition pairs as labeled examples for a multiclass classifier.[1] The classifier predicts the optimal next action ($\arg\max_{a \in \{S, LA, RA\}} \overline{w}^T f(\text{stack, buffer}, a)$) based on features derived from the top elements and their dependents on the stack and buffer.[1]

However, the speed of transition-based parsing comes at the cost of robustness. Because it is a greedy, sequential process, the training procedure assumes all previous decisions were optimal. If the classifier makes an early error, this mistake propagates forward, potentially leading the parser into an unrecoverable, non-gold state.[1] This reliance on perfect local decisions makes the method fragile compared to global, search-based methods like CKY.

## Part III. Deep Learning Architectures and Modern NLP

### Chapter 7: Neural Network Architectures

#### 7.1 Feedforward Networks (FFNNs)

A Feedforward Neural Network transforms input features $f(\overline{x})$ into a latent feature space $\overline{z}$ via sequential layers.[1] Each layer performs a linear transformation followed by a non-linear activation function $g$: $\overline{z} = g(V f(\overline{x}))$.[1] Final classification is achieved by applying a weight vector $\overline{w}$ to the final hidden layer $\overline{z}$.[1]

Common activation functions include:
*   **Tanh:** Squashes outputs into the range $[-1, 1]$. It suffers from the vanishing gradient problem if activation inputs are too large in magnitude.[1]
*   **ReLU (Rectified Linear Unit):** $\max(0, z)$. It helps mitigate vanishing gradients for positive inputs by having a linear dynamic range but can suffer from "dying ReLUs" if the input remains strongly negative.[1]

A conceptually important architecture is the Deep Averaging Network (DAN), a simple model where the input to the FFNN is the average of all word embeddings in the sentence ($\text{av} = \frac{1}{n} \sum c_i$).[1] Empirical results show that this simple averaging technique often performs surprisingly well, achieving accuracy comparable to or better than complex tree-structured neural networks on sentiment tasks.[1] This evidence demonstrates that, for some classification problems, sophisticated syntactic composition may be less critical than merely creating a robust, distributed representation of the lexical content.[1]

#### 7.2 Training FFNNs and Backpropagation

The training process maximizes the log likelihood of the training data, typically by minimizing the Negative Log-Likelihood (NLL).[1] **Backpropagation** is the efficient algorithm used to compute the gradients of the loss with respect to all parameters (weights $V$ and $W$) by applying the chain rule sequentially backwards through the network's computation graph.[1] Modern deep learning frameworks (e.g., PyTorch) automate this differentiation process.[1]

Successful training depends on several practical considerations:
*   **Initialization:** Weights must be initialized to small, non-zero random values (e.g., using the Glorot initializer). Initializing all weights to zero is catastrophic, as all hidden units would be identical and receive zero gradients, preventing any learning.[1]
*   **Regularization (Dropout):** Dropout is a powerful form of stochastic regularization applied during training.[1] It probabilistically sets a portion of the hidden unit activations to zero. This practice forces the network to become robust to missing signals, promoting redundancy among neurons, and significantly preventing overfitting.[1]

### Chapter 8: The Transformer and Contextualized Models

#### 8.1 Attention Mechanism Fundamentals

The attention mechanism is the cornerstone of the Transformer architecture, replacing the sequential processing of RNNs with parallel computation.[1] Attention relies on projecting input vectors into three distinct matrices [11]:

1.  **Query ($Q$):** Represents what the current token is seeking (via weight matrix $W_Q$).
2.  **Key ($K$):** Represents what information is available at other tokens (via $W_K$).
3.  **Value ($V$):** Represents the content to be aggregated if selected (via $W_V$).

The output is computed using the **Scaled Dot-Product Attention** formula [1]:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
The inner term $QK^T$ calculates similarity scores between all queries and keys. The subsequent division by $\sqrt{d_k}$ (the dimension of keys) is crucial: since the variance of the dot product grows linearly with $d_k$, this scaling stabilizes the inputs to the softmax function. Without scaling, the softmax tends to saturate, leading to extremely "peaky" distributions where attention focuses on only one or two tokens, hindering diverse information integration.[1, 11]

The Self-Attention mechanism provides an $O(1)$ path length for information flow between any two tokens, regardless of sequence length $n$.[1] This architectural advantage fundamentally solves the long-range dependency problem that plagues RNNs.

#### 8.2 Transformer Architecture Details

The full Transformer utilizes **Multi-Head Attention (MHA)**. MHA runs several independent attention calculations in parallel, each initialized with different projection matrices ($W_Q, W_K, W_V$). The outputs of these "heads" are concatenated and passed through a final linear layer.[1] MHA allows the model to capture a variety of relational information simultaneously—for example, one head might specialize in positional dependencies while another captures syntactic relationships.[1]

Because the dot-product self-attention operation is permutation invariant, sequential information must be explicitly injected using **Positional Encoding (PE)**.[1] PE vectors (either learned or sinusoidal functions) are added to the input word embeddings before they enter the attention layers.[1]

The Transformer architecture consists of stacked blocks, each containing an MHA layer and a Feedforward Network (FFN) layer, linked by **Residual Connections** (known as Add \& Norm).[1] The FFN layers, which perform linear transformations and activation functions over each token vector independently, contain the majority of the model parameters.[1]

The computational complexity of the core Transformer block is constrained primarily by the self-attention mechanism, which is $O(n^2d)$ for sequence length $n$ and embedding dimension $d$.[1] This quadratic dependence limits the maximum context size that can be processed efficiently.

Table III contrasts sequential models with the parallel Transformer architecture.

Table III: Sequence Model Architecture Tradeoffs

| Architecture | Model Type | Context Path Length | Parallelization | Primary Limitation/Cost |
|---|---|---|---|---|
| HMM (Inference) | Generative DP | $O(n)$ (sequential) | None | Strong Markov/Independence assumptions; $O(nk^2)$ cost. |
| RNN/LSTM | Recurrent/Sequential | $O(n)$ (sequential steps) | Low (must wait for $h_{t-1}$) | Vanishing/Exploding Gradient [1]; slow training.[12] |
| Transformer (Self-Attention) | Attention/Parallel | $O(1)$ (direct) | High (Matrix Ops) | $O(n^2d)$ complexity in attention layer [1]; needs positional encoding. |

#### 8.3 Seq2Seq Models and Cross-Attention

The Encoder-Decoder framework is used for sequence-to-sequence (Seq2Seq) tasks such as machine translation, where an input sequence $x$ is mapped to an output sequence $y$.[1]

In the Transformer Seq2Seq architecture, the Decoder contains two key attention components [1]:
1.  **Causal Self-Attention:** A masked self-attention mechanism ensures that token $y_i$ only attends to previous output tokens $y_1, \dots, y_{i-1}$, preserving the autoregressive property required for generation.[1]
2.  **Encoder-Decoder Cross-Attention:** This mechanism allows the decoder to align its generation with the input. The Query ($Q$) comes from the current decoder state, while the Key ($K$) and Value ($V$) come from the entire, pre-computed output of the Encoder. This allows the decoder to selectively weight and extract relevant information from the source text during the generation of each target word.[1]

### Chapter 9: Pre-trained Language Models

#### 9.1 Contextualization: From ELMo to BERT

The development of contextualized embeddings marked a significant advance in NLP. Early models like ELMo used concatenated outputs of two independent, unidirectional LSTMs (left-to-right and right-to-left) to capture context.[1] However, this approach is limited because it models context in isolation.

**BERT (Bidirectional Encoder Representations from Transformers)** addressed this by using the Transformer Encoder to learn deeply bidirectional representations simultaneously.[1] To prevent the model from "cheating" by directly accessing the token it is trying to predict, BERT is trained using two primary unsupervised objectives [1]:
1.  **Masked Language Modeling (MLM):** Randomly masking $15\%$ of input tokens and training the model to predict the original masked tokens using the full, unrestricted context. This forces the model to integrate bidirectional information deeply.[1]
2.  **Next Sentence Prediction (NSP):** Predicting whether two input text segments are contiguous in the training corpus, helping to model relationships between sentences.[1]

BERT is architecturally optimized for analysis tasks (e.g., classification, tagging). For classification tasks, the contextualized embedding of the special `` token is used as the representation for the entire sequence, which is then fed into a linear layer and softmax.[1] Importantly, BERT, as a masked model, cannot inherently generate text autoregressively (left-to-right).[1]

#### 9.2 Encoder-Decoder Pre-training

For generative Seq2Seq tasks, models like BART and T5 introduced pre-training schemes targeting generation:

*   **BART (Bidirectional Auto-Regressive Transformer):** An encoder-decoder model pre-trained as a denoising autoencoder. It corrupts the input sequence using various noise schemes (e.g., token deletion, text infilling of long spans) and is tasked with reconstructing the original, clean sequence autoregressively.[1] BART is highly effective for summarization tasks when fine-tuned.[1]
*   **T5 (Text-to-Text Transfer Transformer):** Frames *all* NLP tasks, including QA, summarization, and translation, as a unified text-to-text generation task.[1] T5 utilizes a massive corpus (Colossal Cleaned Common Crawl, C4) and similar denoising pre-training objectives.[1] The UnifiedQA project demonstrated T5's versatility by showing that a single, large model, when fine-tuned, could successfully handle heterogeneous QA formats (span extraction, multiple choice, abstractive generation) within the same architecture.[1]

Performance gains across modern NLP have been driven by an emphasis on scale and robust optimization. The superior performance of models like RoBERTa (Robustly optimized BERT) confirmed that maximizing data quantity (scaling from 16GB to 160GB) and implementing dynamic masking (recomputing masks per epoch) are critical factors.[1] The massive leaps in model accuracy, such as the $90\%$ to $97\%$ accuracy gains on standard benchmarks achieved in a short time frame, confirm the principle that maximizing the capacity of the model via sheer scale (data and parameters) is currently the dominant factor in achieving state-of-the-art results.[1]

### Chapter 10: Generation and Decoding Strategies

#### 10.1 Decoding Strategies

Language models (LMs) and Seq2Seq models output a probability distribution $P(y_i|context)$ over the vocabulary for the next token.[1] Decoding strategies determine how to select the sequence of tokens from these distributions:

*   **Greedy Decoding:** Selects the highest probability token locally at each step ($\arg\max P(y_i|context)$). It is the fastest method but frequently results in globally suboptimal sequences.[1]
*   **Beam Search:** Maintains $B$ competing hypotheses (the "beam") at each timestep, expanding them and keeping the $B$ sequences with the highest total probability. It is commonly used for constrained tasks like machine translation to find the most probable output sequence.[1]

For open-ended generation tasks (like story writing), beam search frequently fails, leading to repetitive or "degenerate" outputs.[1] This occurs because, even if a token starts a repeating loop, the locally normalized probability of that token may still be high given its immediate predecessor, causing the beam to collapse onto this local maximum.[1]

#### 10.2 Nucleus Sampling

To generate diverse and high-quality text, pure sampling (drawing tokens proportionally to their probability) is often avoided because it frequently draws from the low-probability "long tail" of the distribution, resulting in grammatical errors and incoherence.[1]

**Nucleus Sampling** (p-sampling) is a superior strategy that dynamically filters the vocabulary.[1] It defines a probability mass threshold $p$ and selects only the smallest set of most probable tokens whose cumulative probability sum exceeds $p$ (the "nucleus"). Tokens outside this nucleus are discarded, and the probabilities of the remaining tokens are renormalized before sampling occurs.[1] This method effectively cuts off the noisy long tail while preventing the repetition seen in beam search, resulting in outputs that exhibit better perplexity and higher human evaluations (HUSE) for naturalness.[1]

#### 10.3 Language Model Evaluation

Standard accuracy is inappropriate for LMs since predicting the exact next word is generally impossible.[1] Instead, LMs are evaluated intrinsically based on their ability to model held-out data.

**Perplexity (PPL):** The standard metric, defined as the exponentiated average Negative Log-Likelihood (NLL) of the held-out text.[1]
$$\text{PPL} = \exp\left(-\frac{1}{n} \sum_{i=1}^{n} \log P(w_i|w_1, \ldots, w_{i-1})\right)$$
Lower perplexity scores indicate that the model assigns higher probability to the observed test data, reflecting a better fit and better generalization capacity. PPL scores typically range from $10$ to $200$, depending on the model quality and the complexity of the dataset.[1]

## Part IV. Appendix and Comparative Reference

### Appendix A: Core Formulas and Mathematical Relations

#### A.1 Classification Formulas (Binary, Multiclass, Gradients)

| Name | Formula / Definition | Context |
|---|---|---|
| Binary LR Probability | $P(y=+1\|\overline{x})=\frac{e^{\overline{w}^{T}f(\overline{x})}}{1+e^{\overline{w}^{T}f(\overline{x})}}$ | Sigmoid activation function. |
| Multiclass Softmax | $P(y=\hat{y}\|\overline{x})=\frac{e^{\overline{w}^{T}f(\overline{x},\hat{y})}}{\sum_{y^{\prime}\in \mathcal{Y}}e^{\overline{w}^{T}f(\overline{x},y^{\prime})}}$ | Logits normalized into probabilities. |
| Perceptron Loss (Hinge) | $L(\overline{w}) = \max(0, -y^{(i)} \overline{w}^T f(\overline{x}^{(i)}))$ | Used to derive the Perceptron update rule. |
| LR Gradient (for $y=+1$) | $\frac{\partial L}{\partial \overline{w}} = f(\overline{x})[P(y=+1\|\overline{x}) - 1]$ | Determines direction and magnitude of weight update. |

#### A.2 Word Embedding Objectives (Skip-Gram with Negative Sampling, GloVe)

| Name | Formula / Definition | Context |
|---|---|---|
| SGNS Binary Probability | $P(y=1\|w,c) = \frac{e^{w \cdot c}}{e^{w \cdot c}+1}$ | Likelihood of a (word, context) pair being real. |
| SGNS Training Objective | $\max \left( \log P(y=1\|w, c) + \frac{1}{k} \sum_{i=1}^{k} \log P(y=0\|w_{i}, c) \right)$ | Maximizing likelihood of real pairs, minimizing likelihood of $k$ negative pairs. |
| GloVe Objective | $\min \sum_{i,j} f(\text{count}(w_i, c_j)) (w_i^T c_j + a_i + b_j - \log \text{count}(w_i, c_j))^2$ | Weighted sum of squared errors on log co-occurrence. |

#### A.3 Hidden Markov Model Formulas (Initialization, Recurrence, Parameter Estimation)

| Name | Formula / Definition | Context |
|---|---|---|
| HMM Joint Probability | $P(\overline{x}, \overline{y}) = P(y_1) P(x_1\|y_1) \prod_{i=2}^{n} P(y_i\|y_{i-1}) P(x_i\|y_i)$ | Decomposition based on Markov and independence assumptions. |
| Viterbi Recurrence (Log) | $\nu_i(y) = \log P(x_i\|y) + \max_{y_{prev}} [\log P(y\|y_{prev}) + \nu_{i-1}(y_{prev})]$ | Dynamic programming step, using $\max$ operation. |
| Viterbi Runtime | $O(nk^2)$ | Computational cost for optimal sequence inference. |

#### A.4 Deep Learning Primitives (Softmax, Attention)

| Name | Formula / Definition | Context |
|---|---|---|
| Perplexity (PPL) | $\text{PPL} = \exp\left(-\frac{1}{n} \sum_{i=1}^{n} \log P(w_i\|w_1, \ldots, w_{i-1})\right)$ | Standard metric for language model evaluation. |
| Scaled Dot-Product Attention | $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$ | Transformer mechanism, scaled by $\sqrt{d_k}$ for stability. |

### Appendix B: Comparative Analysis Tables

Table V: Decoding Strategy Tradeoffs

| Strategy | Decision Rule | Primary Goal | Open-Ended Generation | Issue/Mitigation |
|---|---|---|---|---|
| Greedy Search | Local $\arg\max$ | Speed. | Suboptimal sequence quality. | None. |
| Beam Search | Top $B$ sequences by total $\log P$ | Global likelihood maximization. | Degeneration (repetition).[1] | Length normalization needed. |
| Nucleus Sampling | Sample from Top $p\%$ mass | High naturalness/diversity. | Avoids noisy long-tail, prevents degeneration.[1] | Requires selecting optimal $p$. |