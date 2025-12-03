# Two-view Graph Neural Networks for Knowledge Graph Completion on YAGO3-10

**Salavat Faizullin, Ilya Grigorev, Ivan Ershov, DS-01 Group**

*Innopolis University, Innopolis, Russia*

## Abstract

We explore knowledge graph completion using the YAGO3-10 dataset through Two-view Graph Neural Networks (WGE). Knowledge graphs consist of triplets in the form (head, relation, tail), where head and tail are entities connected by specific relations. The primary task involves link prediction for answering queries about likely associations. WGE processes knowledge graphs from dual perspectives: entity connections and relationship patterns, using quaternion algebra for enhanced representation learning. Our approach demonstrates suitability for YAGO3-10's diverse entity types and 37 distinct relation types spanning spatial, temporal, and social dimensions. However, experimental results reveal a significant performance asymmetry between head and tail prediction tasks, indicating challenges in modeling certain relational patterns. We analyze these findings in the context of relation type characteristics and propose directions for architectural improvements.

## Introduction

Knowledge graph completion addresses the problem of predicting missing links in knowledge bases structured as triplets (head, relation, tail). Users typically seek answers to queries regarding terms highly likely to be associated with given head/tail entities and relations. This constitutes a link-level prediction problem on knowledge graphs where traditional models often miss potentially useful relation structure information, limiting their ability to handle correctly interpreted input queries.

The YAGO3-10 dataset exemplifies this challenge with its diverse entity types (personalities, locations, organizations, concepts) and 37 distinct relation types. Many entities exhibit weak associations with others, requiring models that leverage both direct entity connections and relational dependencies. We apply WGE's two-view approach to effectively address YAGO3-10's incompleteness while maintaining interpretability through human-readable entity and relation names.

## Method

### 1. Dataset Information

We selected YAGO3-10 for knowledge graph completion with the following characteristics:

**Key Characteristics:**
- **Entities**: 123,182 real-world entities with meaningful names
- **Relations**: 37 diverse relationship types
- **Triples**: 1,079,040 factual statements
- **Domains**: People, locations, organizations, events, and concepts

**Prediction Tasks:**
- **Head Entity Prediction**: Given (?, relation, tail), predict missing head entity
- **Tail Entity Prediction**: Given (head, relation, ?), predict missing tail entity

**Evaluation Metrics:**
- **Mean Reciprocal Rank (MRR)**: $$\text{MRR} = \frac{1}{|Q|} \sum\_{i=1}^{|Q|} \frac{1}{\text{rank}\_i}$$
- **Hits@K**: Proportion of correct entities ranked in top K positions
- **Filtered Setting**: Standard evaluation protocol removing other valid triples during ranking


### 2.1 Dataset Preparation

#### 2.1.1 Graph Construction

**Entity-Focused Graph**: Convert the directed KG to an undirected graph where entities are nodes and edges represent any relationship between them. For each triple $(h, r, t)$, entities $h$ and $t$ become connected nodes.

**Relation-Focused Graph**: Construct from Relation-Focused (RF) constraints $(r_s, e_p, r_o)$, where $r_s$ and $r_o$ are relations connected through predicate entity $e_p$. We select the top $\beta$ fraction of RF constraints based on $(r_s, r_o)$ co-occurrence frequency.

#### 2.1.2 Negative Sampling Strategy

**Training Stage**: Generate invalid triples by corrupting each training triple, replacing either head or tail with random entities from the entire dataset. This produces balanced training data with assigned labels: $1$ for valid triples, $0$ for corrupted ones.

**Evaluation Stage**: For each test triple $(h, r, t)$, generate candidate sets:
- Tail prediction: $(h, r, e_i)$ for $K$ entity candidates
- Head prediction: $(e_i, r, t)$ for $K$ entity candidates
Apply filtration to remove candidates existing in the KG. The model scores $K$ negative candidates plus one positive triple for ranking.

### Model Architecture

WGE processes knowledge graphs through dual perspectives and quaternion algebra:

**Core Components:**

1. **Dual Graph Construction:**
   - **Entity-Focused Graph**: $\mathcal{G}\_{ef} = (\mathcal{V}\_{ef}, \mathcal{E}\_{ef})$ where $|\mathcal{V}\_{ef}| = |\mathcal{V}|$
   - **Relation-Focused Graph**: $\mathcal{G}\_{rf} = (\mathcal{V}\_{rf}, \mathcal{E}\_{rf})$ with nodes from RF constraints $(r\_s, e\_p, r\_o)$

2. **QGNN Processing:**
   Entity-focused view updates:
   $$\boldsymbol{h}\_{\mathsf{v},ef}^{(k+1),Q} = \mathsf{g}\left(\sum\_{\mathbf{u} \in \mathcal{N}\_{\mathsf{v}} \cup \{\mathsf{v}\}} a\_{\mathsf{v},\mathbf{u},ef}\boldsymbol{W}\_{ef}^{(k),Q} \otimes \boldsymbol{h}\_{\mathbf{u},ef}^{(k),Q}\right)$$

   Relation-focused view updates:
   $$\boldsymbol{h}\_{\mathsf{v},rf}^{(k+1),Q} = \mathsf{g}\left(\sum\_{\mathbf{u} \in \mathcal{N}\_{\mathsf{v}} \cup \{\mathsf{v}\}} a\_{\mathsf{v},\mathbf{u},rf}\boldsymbol{W}\_{rf}^{(k),Q} \otimes \boldsymbol{h}\_{\mathbf{u}}^{(k),Q}\right)$$

   where $\boldsymbol{h}\_{\mathbf{u}}^{(k),Q}$ is defined as:
   $$\boldsymbol{h}\_{\mathbf{u}}^{(k),Q} =
   \boldsymbol{h}\_{\mathbf{u},ef}^{(k),Q}$$ if $$\mathbf{u}$$ is an entity node;
   $$\boldsymbol{h}\_{\mathbf{u},rf}^{(k),Q}$$ if $$\mathbf{u}$$ is a relation node

3. **Cross-View Interaction:**
   $$\boldsymbol{h}\_{\mathbf{u},ef}^{(k),Q} = \boldsymbol{h}\_{\mathbf{u},ef}^{(k),Q} * \boldsymbol{h}\_{\mathbf{u},rf}^{(k),Q}$$

4. **Multi-Layer Scoring:**
   $$f\_k(h,r,t) = \left(\boldsymbol{h}\_h^{(k),Q} \otimes \boldsymbol{h}\_r^{\triangle,(k),Q}\right) \bullet \boldsymbol{h}\_t^{(k),Q}$$
   $$f(h,r,t) = \sum\_k \alpha\_k f\_k(h,r,t)$$

5. **Training Objective:**
   $$\mathcal{L} = -\sum\_{(h,r,t) \in \{\mathcal{T} \cup \mathcal{T}'\}} \sum\_k \alpha\_k \Big( l\_{(h,r,t)} \log \left( p\_k(h,r,t) \right) + \left( 1 - l\_{(h,r,t)} \right) \log \left( 1 - p\_k(h,r,t) \right) \Big)$$

where:
   - $l\_{(h,r,t)} = 1 \text{ for } (h,r,t) \in \mathcal{T}, 0 \text{ for } (h,r,t) \in \mathcal{T}'$
   - $p\_k(h,r,t) = \sigma\left( f\_k(h,r,t) \right)$
   - $\mathcal{T}$: collection of valid triples
   - $\mathcal{T}'$: collection of invalid triples collected by corrupting valid triples

**Notation:**
- $\otimes$: Hamilton product
- $\bullet$: Quaternion inner product  
- $*$: Quaternion element-wise product
- $\mathsf{g}$: $\tanh$ activation function
- $\alpha\_k$: Fixed important weight of $k$-th layer with $\sum\_k \alpha\_k = 1$
- $\boldsymbol{h}^{(k),Q}$: Quaternion vector from $k$-th layer
- $\triangle$: Denotes normalized quaternion
- $\sigma$: Sigmoid function

## Results

### Training Configuration

The model was trained under the following settings:
```
Device: cuda
Embedding dimension: 64
Number of layers: 1
Learning rate: 1e-3
Batch size: 256
Number of epochs: 10
Beta (RF constraint ratio): 0.2

Number of validation batches: 5000
Number of test batches: 5000
```

After each training epoch, the model performace was evaluated and the best model was tracked down. The results after the training were obtained, below you can see the key metrics evaluated on the test dataset:

**Comparative Analysis of WGE Implementation on YAGO3-10**

| Metric | Value |
|--------|-------|
| **Overall MRR** | 0.193 |
| **Overall Hits@10** | 0.346 |
| **Head Prediction MRR** | 0.06 |
| **Tail Prediction MRR** | 0.325 |
| **Head Prediction Hits@10** | 0.132 |
| **Tail Prediction Hits@10** | 0.560 |

### Comparative Analysis

#### Performance Profile
The model demonstrates a **significant performance asymmetry** between head and tail prediction tasks, with tail prediction (MRR: 0.325) substantially outperforming head prediction (MRR: 0.06). This discrepancy is more pronounced than typically observed in KG completion benchmarks, suggesting specific challenges in head entity prediction on YAGO3-10. The overall MRR of 0.193 and Hits@10 of 0.346 indicate that the current implementation achieves basic functionality but falls short of state-of-the-art performance on this dataset.

#### Contextual Comparison
While the original WGE paper reported results on FB15K-237, CoDEx, and LitWD benchmarks, direct numerical comparison with YAGO3-10 is constrained by dataset differences. However, several observations emerge:

1. **Performance Gap**: Our implementation's overall MRR (0.193) is substantially lower than WGE's reported results on comparable datasets (typically 0.30-0.45 MRR). This suggests potential underfitting or suboptimal hyperparameter configuration.

2. **Architectural Fidelity**: The maintained performance gap between head and tail predictions (ΔMRR = 0.265) exceeds typical asymmetries in KG completion, indicating that the relation-focused constraints may not adequately capture inverse relationship patterns crucial for head prediction.

3. **Training Efficiency**: With only 10 epochs of training, the model may not have converged fully. The original WGE paper utilized 3000 epochs with early stopping, suggesting our abbreviated training schedule limits performance potential.

## Discussion

WGE's two-view approach is theoretically suitable for YAGO3-10 because the entity-focused graph captures YAGO3-10's rich entity neighborhood structures while the relation-focused graph models dependencies between diverse relation types through RF constraints. However, our experimental results reveal limitations in handling the full spectrum of relational patterns present in YAGO3-10.

### Relation Type Coverage Analysis

The performance asymmetry between head and tail prediction suggests challenges in modeling certain relation types:

1. **Inverse Relations**: The substantial head prediction deficit (MRR 0.06 vs tail MRR 0.325) indicates poor modeling of inverse relationships. Many relations in YAGO3-10 have natural inverses (e.g., "bornIn" vs "hasBirthplace"), but the current RF constraint selection (β=0.2) may insufficiently capture these bidirectional dependencies.

2. **One-to-Many Relations**: YAGO3-10 contains numerous one-to-many relationships where a single head connects to multiple tails (e.g., "hasWritten" connecting an author to multiple works). The current scoring function may struggle with these imbalanced distributions, particularly when predicting heads given specific tails.

3. **Symmetric vs Anti-symmetric Relations**: The model's differential performance may reflect varying effectiveness across symmetric (e.g., "marriedTo") and anti-symmetric (e.g., "locatedIn") relations. The quaternion scoring function inherently supports anti-symmetry through the Hamilton product, but may require architectural adjustments for optimal symmetric relation handling.

4. **Transitive Relations**: Hierarchical relations in YAGO3-10 (e.g., "isLocatedIn" forming location hierarchies) benefit from multi-hop reasoning. The single-layer architecture (K=1) limits transitive relation modeling, potentially explaining the suboptimal Hits@10 performance.

### Architectural Considerations

The technique examines knowledge graphs from two perspectives simultaneously: entity connections (how entities connect like a social network) and relationship patterns (how different relationship types connect through shared entities). This dual perspective helps the model understand both who is connected to whom and how different connection types relate to each other, analogous to studying a city map by examining building locations and road connections simultaneously.

However, several implementation factors likely contributed to the observed performance:

1. **Underparameterization**: The 64-dimensional embeddings and single-layer architecture may insufficiently capture YAGO3-10's complexity (123K entities, 37 relations).

2. **Training Duration**: 10 epochs represents minimal training for a dataset of this scale. The original WGE implementation utilized 3000 epochs with careful early stopping.

3. **RF Constraint Selection**: The β=0.2 threshold for RF constraint retention may exclude valuable relation pair dependencies, particularly for less frequent but semantically important relationships.

4. **Evaluation Protocol**: Using 5000 randomly sampled test batches (approximately 38% of total test triples) provides statistical reliability but may not fully represent performance across all relation types.

The achieved metrics validate the core WGE architecture's basic functionality while highlighting substantial areas for improvement. The tail prediction performance gap may be addressed through refined relation-focused constraint selection or enhanced negative sampling strategies. Future work should investigate hyperparameter optimization and potential architectural extensions to balance directional prediction capabilities.

## Conclusion

Our implementation of Two-view Graph Neural Networks (WGE) for knowledge graph completion on the YAGO3-10 dataset demonstrates the feasibility of dual-perspective graph learning but reveals significant performance limitations in its current configuration. The pronounced asymmetry between head and tail prediction (ΔMRR = 0.265) suggests specific challenges in modeling inverse relationships and handling the diverse relation types present in YAGO3-10.

Several corrective measures are indicated by our analysis:

1. **Extended Training**: Increasing training epochs to 3000 with proper early stopping would better align with the original WGE methodology and potentially improve convergence.

2. **Architectural Enhancements**: Increasing embedding dimensions (≥256), adding GNN layers (2-3), and adjusting the RF constraint retention (β) could better capture complex relational patterns.

3. **Relation-Type Specific Adjustments**: Incorporating explicit modeling of relation properties (symmetry, transitivity, inversion) through specialized scoring functions or constraint mechanisms.

4. **Comprehensive Evaluation**: Full evaluation on all test triples with per-relation performance analysis to identify specific relation types challenging for the current architecture.

WGE's dual-view approach remains promising for knowledge graph completion, particularly for datasets like YAGO3-10 with rich relational structures. However, achieving competitive performance requires careful attention to training duration, architectural capacity, and relation-specific modeling. Future work should focus on these enhancements while maintaining the interpretability advantages of WGE's human-readable entity and relation representations.

## References

1. F. Mahdisoltani, J. Biega, and F. Suchanek, "YAGO3: A Knowledge Base from Multilingual Wikipedias." In *CIDR*, 2015.

2. V. Tong, D. Nguyen, D. Phung, and D. Nguyen, "Two-view Graph Neural Networks for Knowledge Graph Completion." The Semantic Web. ESWC 2023. Lecture Notes in Computer Science, vol. 13870. Springer, Cham. DOI: 10.1007/978-3-031-33455-916, 2023.