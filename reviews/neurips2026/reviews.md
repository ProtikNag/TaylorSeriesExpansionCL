# NeurIPS 2026 Reviews, Submission 30395

Verbatim transcription of the OpenReview decision and official reviews.
Original screenshots are in [screenshots/](screenshots/).

| Reviewer | Rating | Confidence | Quality | Clarity | Significance | Originality |
|---|---|---|---|---|---|---|
| cfM1 | 2 Reject | 3 | 2 | 3 | 3 | 2 |
| tHSB | 2 Reject | 4 | 1 | 2 | 2 | 2 |
| SA31 | 3 Borderline reject | 3 | 2 | 2 | 3 | 3 |

**Decision:** Reject (Program Chairs, 24 Sept 2026)

---

## Metareview (Program Chairs)

The paper proposes HTCL, a continual-learning wrapper that searches over task orders within small groups and consolidates the selected local models using hierarchical curvature-aware updates. Task-order sensitivity is an important problem, and the evaluation spans several continual-learning methods, modalities, architectures, and metrics. The reported reductions in variance and forgetting are potentially useful.

The reviewers are nevertheless broadly aligned on several substantive concerns. First, HTCL trains multiple permutations, selects among them, and uses substantially more computation than the underlying baselines, while the data used for this selection are not clearly specified. It is therefore unclear whether the gains arise from the proposed consolidation rather than additional training and model selection. Second, the theoretical claims are overstated: Equations (1) and (2) are themselves permutation-invariant, Theorem A.3 concerns only selection of the best local permutation, and Proposition A.4 relies on a strong coverage assumption. These results do not establish the claimed performance guarantee for the final hierarchical model. The novelty of the consolidation update relative to standard quadratic or proximal updates and prior order-robust continual-learning work also needs clearer positioning.

The rebuttal should address four points:

1. Specify exactly what data Eval uses, how validation sets are constructed, and whether test data are excluded from all selection and tuning. Also clarify what access HTCL requires to tasks within a group and how this fits the claimed continual-learning setting.
2. Provide a decisive control separating permutation search, extra compute, and second-order consolidation. A suitable comparison would hold the trained permutations fixed while comparing selected-order consolidation, random-order consolidation, and selected-order training without the proposed update. Existing compute-matched evidence would also be relevant.
3. Correctly delimit the theory. Either establish a result for the final consolidated model or narrow the theorem and associated claims to the intra-group selection rule. Please also clarify the novelty of the update relative to the closest proximal, curvature-based, and order-robust methods.
4. Clarify the ImageNet-100 class-incremental protocol. The appendix describes a task-local 10-output classifier with remapped labels, which does not by itself define standard 100-way class-incremental evaluation. State the actual output space and whether task identity is available at inference.

---

## Reviewer cfM1

**Summary.** The paper proposes HTCL, a hierarchical continual learning wrapper that reduces task order sensitivity. It searches over small intra group task permutations, trains a local model on the best order, then consolidates it into a global model using a second order Taylor style update with Hessian or diagonal Hessian information. The method is tested across image, text, and graph benchmarks.

**Strengths and Weaknesses.** The paper studies an important and under discussed issue in continual learning: final performance can vary a lot depending on task order. The idea of using small group permutation search plus hierarchical consolidation is intuitive, and the experiments cover several datasets and baselines. The reported reductions in variance and forgetting are also encouraging.

That said, I am not fully convinced by the contribution. The core method is expensive and partly sidesteps the problem by trying multiple task orders within each group, which may not be realistic in strict online CL where task order is not controllable and repeated training on permutations is costly. The theoretical guarantee is weak: selecting the best among several permutations is naturally no worse than a random choice in that group, but this does not say much about global order robustness after consolidation. The Taylor update also relies on Hessian approximations and replay buffers, so the method is not as clean or model agnostic as claimed. The empirical gains are mostly shown with small groups and relatively standard benchmarks, while the compute overhead is large, about 4x in the main example. I also think the comparison needs stronger baselines that use the same extra compute budget, for example repeated training, model averaging, checkpoint selection, or hyperparameter tuning with equal wall clock cost. Without this, it is hard to know whether HTCL is better because of the proposed consolidation or simply because it spends much more computation exploring local orderings.

**Questions.** How would HTCL work in a truly online setting where tasks cannot be revisited or reordered within a group? What is the performance under an equal compute budget baseline? How sensitive are results to the diagonal Hessian approximation and lambda? How much of the gain comes from permutation search versus second order consolidation? Can the authors report results with no replay buffer or much smaller buffers across all datasets?

**Limitations.** The limitations should be stronger. The paper should more directly discuss the extra compute, dependence on replay, weak global guarantee, scalability of permutation search, and the gap between grouped permutation search and real uncontrolled task streams.

**Scores.** Quality 2, Clarity 3, Significance 3, Originality 2. Rating 2 (Reject). Confidence 3. Ethics: none. Formatting: none.

---

## Reviewer tHSB

**Summary.** This paper studies task-order sensitivity and forgetting in continual learning. The authors propose HTCL, a framework that exhaustively explores all task orderings within small task groups and consolidates the selected local models through a hierarchical, curvature-aware Taylor update. Experiments across multiple modalities and continual learning baselines reportedly demonstrate higher average accuracy, reduced forgetting, and lower order-induced performance variance.

**Strengths.**
- The local-search and hierarchical-consolidation workflow is clearly presented, with explicit update rules and a helpful algorithmic illustration.
- The evaluation covers multiple continual learning methods, data modalities, architectures, and metrics, including accuracy, forgetting, variance, and runtime.

**Weaknesses.**
- The formulations in Equations (1) and (2) do not formally capture task-order sensitivity. The objective itself is invariant to any permutation of tasks. Task-order effects arise from the sequential optimization trajectory rather than from the static summation objective presented in the paper.
- Theorem A.3 only shows that selecting the best local ordering within a fixed group is at least as good as uniformly sampling one ordering from that group. It does not provide any guarantee on the final consolidated model or any bounds on the global task-ordering problem.
- Proposition A.4 similarly treats the $m(k!)$ local evaluations as providing coverage of the $t!$ global permutation space, but this interpretation is not adequately justified.
- The closed-form update follows directly from a standard quadratic Taylor/proximal optimization formulation. As a result, the novelty of the consolidation step is not entirely clear.
- The paper does not sufficiently disentangle its contribution from recent prior work on order-robust continual learning.

**Questions.** None. **Limitations.** See above.

**Scores.** Quality 1, Clarity 2, Significance 2, Originality 2. Rating 2 (Reject). Confidence 4. Ethics: none. Formatting: none.

---

## Reviewer SA31

**Summary.** This paper studies task-order sensitivity in continual learning and proposes Hierarchical Taylor Series-based Continual Learning (HTCL). The method partitions the incoming task stream into small groups of size (k), trains all intra-group task permutations, selects the best local model, and then consolidates this local model into one or more hierarchical global models through a Hessian-regularized second-order Taylor update. The framework is also extended to multiple hierarchy levels for longer task sequences. Experiments on SplitMNIST, CIFAR-100, Cora, 20 Newsgroups, and ImageNet-100 add HTCL on top of several continual-learning baselines and report improvements in mean accuracy, task-order variance, and forgetting.

**Strengths and Weaknesses.** I think the paper targets an important issue. Many continual-learning papers report results under a fixed task order or a small number of random orders, but in deployment the order of incoming tasks may be outside the practitioner's control. Measuring and reducing order-induced performance variance is therefore a useful goal. The paper's idea of combining local order search with conservative global second-order consolidation is also a potentially useful outer-loop wrapper around existing continual-learning methods.

Strengths:
- The motivation is clear. Task-order sensitivity is an under-emphasized issue in continual learning, and evaluating mean performance, variance, and forgetting together is appropriate.
- The method has an intuitive structure: small intra-group order search provides local adaptability, while hierarchical Taylor/Hessian updates provide longer-term stability. This decomposition is easy to understand and naturally extends to multiple hierarchy levels.
- The empirical coverage is broad, with several continual-learning baselines and multiple modalities, including image, graph, and text tasks. The paper reports order variance and forgetting rather than only final average accuracy.
- The appendix includes useful ablations, such as first-order consolidation and a DiLoCo-style averaging variant, suggesting that the second-order component may be important.

Weaknesses:
- My main concern is the experimental protocol and fairness of the comparison. HTCL trains all (k!) task permutations inside each task group and selects the best local model. This is not the usual online continual-learning protocol: it introduces intra-group lookahead, extra training budget, and a model-selection step. The algorithm's `Eval` operation is not specified clearly enough. If it uses the test set, or data strongly overlapping with final evaluation, the results would suffer from selection bias. Even if it uses a validation set, HTCL receives extra data access, latency, and model-selection opportunities compared with the baselines.
- The current ablations do not cleanly attribute the gains. Improvements may come from enumerating and selecting the best intra-group order, from the second-order consolidation, or from the extra compute budget itself. The paper compares first-order consolidation and DiLoCo-style aggregation, but it does not sufficiently report controls such as best intra-group order search without second-order consolidation, random intra-group order plus second-order consolidation, compute-matched repeated baseline training with best/average selection, or simple ensembling/checkpoint selection using the same number of local models.
- The theoretical guarantee is weak relative to how it is presented. Theorem A.3 essentially proves that $\max_\sigma A(\sigma) \geq \mathbb{E}_\sigma A(\sigma)$, i.e., choosing the best among evaluated permutations is no worse than a random choice on that group-level metric. This statement is true but close to a direct property of the selection rule. It does not show that the full hierarchical algorithm remains no worse than random task ordering under nonconvex neural training, finite validation noise, approximate Hessians, and interactions across groups. Proposition A.4 additionally relies on a linear-coverage assumption for the residual performance gap, which is not well justified.

**Questions.**
1. What data does `Eval` use when selecting the best intra-group task order? Is it an independent validation set, the training set, the replay buffer, or the test set? If it is not a strictly independent validation set, please explain how selection bias is avoided. If it is a validation set, please describe how it is constructed for every task and how validation selection affects final test performance.
2. Can the authors add ablations that separate intra-group search, extra compute, and second-order consolidation? Important controls include best intra-group order search without second-order consolidation, random intra-group order plus second-order consolidation, repeated baseline training under the same compute budget with best/average selection, simple averaging or ensembling with the same number of local models, and wall-clock-matched baselines.
3. Should Theorem A.3 be stated more cautiously as a guarantee for the group-level selection rule under accurate evaluation of all intra-group permutations? The main-text statement that HTCL is never worse in expectation than random task ordering seems not to cover approximate Hessian consolidation, finite validation noise, nonconvex training randomness, and cross-group interactions. Please clarify exactly which part of the algorithm is covered by the theory.

**Limitations.** Yes.

**Scores.** Quality 2, Clarity 2, Significance 3, Originality 3. Rating 3 (Borderline reject). Confidence 3. Ethics: none. Formatting: no.
