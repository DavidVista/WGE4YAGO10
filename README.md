# Two-view Graph Neural Networks for Knowledge Graph Completion on YAGO3-10

**Salavat Faizullin, Ilya Grigorev, Ivan Ershov, DS-01 Group**

*Innopolis University, Innopolis, Russia*

## Abstract

We explore knowledge graph completion using the YAGO3-10 dataset through Two-view Graph Neural Networks (WGE). Knowledge graphs consist of triplets in the form (head, relation, tail), where head and tail are entities connected by specific relations. The primary task involves link prediction for answering queries about likely associations. WGE processes knowledge graphs from dual perspectives: entity connections and relationship patterns, using quaternion algebra for enhanced representation learning. Our approach demonstrates suitability for YAGO3-10's diverse entity types and 37 distinct relation types spanning spatial, temporal, and social dimensions.

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

*Experimental results and performance metrics will be reported upon completion of model implementation and evaluation on the YAGO3-10 dataset.*

## Discussions

WGE's two-view approach is particularly suitable for YAGO3-10 because the entity-focused graph captures YAGO3-10's rich entity neighborhood structures while the relation-focused graph models dependencies between diverse relation types through RF constraints. Quaternion operations provide expressive representations for YAGO3-10's complex relationship patterns, and multi-layer scoring leverages hierarchical information present in YAGO3-10's taxonomic relationships.

The technique examines knowledge graphs from two perspectives simultaneously: entity connections (how entities connect like a social network) and relationship patterns (how different relationship types connect through shared entities). This dual perspective helps the model understand both who is connected to whom and how different connection types relate to each other, analogous to studying a city map by examining building locations and road connections simultaneously.

## Conclusion

We propose applying Two-view Graph Neural Networks (WGE) for knowledge graph completion on the YAGO3-10 dataset. WGE's dual-perspective architecture effectively addresses knowledge graph incompleteness by capturing both entity neighborhood information and relation-focused constraints. The model's use of quaternion algebra and multi-layer scoring provides enhanced representation learning capabilities suitable for YAGO3-10's diverse entity types and relationship patterns. This approach promises to improve link prediction performance while maintaining the interpretability advantages of YAGO3-10's human-readable entity and relation names.

## References

1. F. Mahdisoltani, J. Biega, and F. Suchanek, "YAGO3: A Knowledge Base from Multilingual Wikipedias." In *CIDR*, 2015.

2. V. Tong, D. Nguyen, D. Phung, and D. Nguyen, "Two-view Graph Neural Networks for Knowledge Graph Completion." The Semantic Web. ESWC 2023. Lecture Notes in Computer Science, vol. 13870. Springer, Cham. DOI: 10.1007/978-3-031-33455-916, 2023.
