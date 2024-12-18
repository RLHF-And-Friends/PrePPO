## Federated RLHF - Federated PPO for Cooperative Policy Optimization

### Motivation and Background

Why is this important?

- Federated Learning (FL): A decentralized approach for privacy-preserving training.
- Reinforcement Learning with Human Feedback (RLHF): Improves performance via human guidance.

Key Motivation:

- As highlighted in the OpenFedLLM paper by Rui Ye et al.:
  - "Trained on massive publicly available data, large language models (LLMs) have demonstrated tremendous success across various fields. While more data contributes to better performance, a disconcerting reality is that high-quality public data will be exhausted in a few years."
  - To address this, collaborative and privacy-preserving LLM training on underutilized distributed private data via federated learning (FL) offers a promising solution.

Key Challenge:

- How can isolated LLMs share knowledge privately and effectively to improve their performance?

Goal:

- Combine Federated Learning and RLHF to enable secure, decentralized cooperation.

---

### Related Work

Federated Learning (FL):

- Classical FL approaches like weight averaging (FedAvg - [McMahan et al., 2017](https://arxiv.org/abs/1602.05629)) and policy gradient sharing ([Geyer et al., 2017](https://arxiv.org/abs/1712.07557)) are computationally heavy for LLM pipelines and introduce potential privacy risks.

Reinforcement Learning with Human Feedback (RLHF):

- RLHF ([Christiano et al., 2017](https://arxiv.org/abs/1706.03741); [Ziegler et al., 2019](https://arxiv.org/abs/1909.08593)) aligns models with human preferences, improving performance through methods like PPO ([Schulman et al., 2017](https://arxiv.org/abs/1707.06347)) and DPO ([Rafailov et al., 2023](https://arxiv.org/abs/2305.18290)).
- RLHF remains predominantly applied in centralized, non-federated settings.

Federated LLM Fine-Tuning:

- Recent works combine Parameter-Efficient Fine-Tuning (PEFT) with FL:
  - [Ye et al. (2024)](https://arxiv.org/abs/2402.06954), [Sun et al. (2024)](https://arxiv.org/abs/2403.12313), [Zhang et al. (2023)](https://arxiv.org/abs/2305.05644)
  - These focus on federated adaptation of LLMs but do not address RLHF directly.

Federated RLHF:

- The only work to date on Federated RLHF comes from two subsequent papers by Feijie Wu et al. ([2024](https://arxiv.org/abs/2407.03038), [2025](https://openreview.net/forum?id=mqNKiEB6pd)):
  - Focuses on a DPO-based approach for federated RLHF.

Gap in Literature:

- No existing work explores alternative Federated RLHF methods such as classical PPO that may incorporate Federated Learning setup much more naturally than DPO.

---

### Our Contribution

Our primary contribution is the development of the **Federated PPO** algorithm, which facilitates communication between agents through a **generalized KL-penalty**.

Key Innovations:

- **Generalized KL-Penalty:**

  - Traditionally used as a trust-region soft constraint for stable training.
  - Extended to act as an *optimization trajectory attractor*, guiding individual policies toward the learning directions of other agents.

- **Private and Effective Information Exchange:**

  - Enables decentralized agents to share knowledge without sharing raw data.
  - Enhances collaborative training through stable and cooperative updates.

Implementation:

- The algorithm pipeline is implemented on top of **Hugging Face frameworks family**.

Initial Results:

- Conducted toy experiments demonstrating **promising performance improvements**:
  - Models trained collaboratively outperform those trained in isolation.

---

### Algorithm Overview

Federated PPO Algorithm Steps:

1. Local Training:
   - Agents train locally using standard PPO.
2. Reference Policy Exchange:
   - Agents communicate their reference policies.
3. Generalized KL-Penalty:
   - Combines received reference policies into a cooperative reference.
   - Use this as the new penalty for local PPO updates:
4. Policy Update:
   - Perform PPO updates using the generalized KL-penalty.

---

### Experimental Setup

Framework and Tools:

- RL Framework: HuggingFace Reinforcement Learning libraries.
- Federated Learning: Simulated federated training setup.

Environments:

- Shards of TRL's Sentiment and Descriptiveness Preference Dataset

Models:
- Elaither AI's 70M Pythias

Baselines for Comparison:

1. Isolated PPO: Agents train independently.
2. Federated PPO: Agents cooperate using the generalized KL-divergence.

Metrics:

- Cumulative reward.
- Convergence speed.

---

### Results

Performance Comparisons:

- Learning Curves:
  - Plot reward curves for Isolated PPO vs. Federated PPO.
- Key Metrics:
  - Federated PPO achieves faster convergence.
  - Higher cumulative rewards compared to isolated agents.

Observations:

- Cooperative training provides performance gains.
- Generalized KL-penalty effectively attracts policy updates toward optimal solutions.
  
---

### Discussion

Key Insights:

- Federated PPO enables cooperation without compromising privacy.
- The KL-divergence generalization provides a principled method to incorporate other agents' knowledge.

Challenges:

1. Communication Overhead: Policy sharing can incur costs.
2. Scalability

---

### Future Work

Next Steps:

1. Scale Up Experiments:
   - Apply Federated PPO to more complex RL environments (datasets).
2. Enhance Privacy Guarantees:
   - Incorporate secure aggregation techniques.
3. Benchmarking on Large Models:
   - Extend the approach to real-world LLMs.
4. Efficiency Improvements:
   - Reduce communication overhead during reference policy sharing.

Broader Impact:

- Federated RLHF as a pathway for collaborative yet private LLM training.

---

### Conclusion

Summary of Contributions:

1. Proposed a novel Federated PPO algorithm for RLHF.
2. Generalized KL-penalty enables cooperative and stable policy updates.
3. Preliminary experiments show performance gains for collaborative agents vs. isolated agents.

Questions?
- Contact us: ark.vladimirov@outlook.com evgurovv@gmail.com
  
---

