# Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion

<!---------------------------------------------------------------------------------------------------------------->
## Preamble

A） Generally, HOI detection includes two steps: <b>Object detection</b> && <b>Interaction Inference</b>.

B） As for Interaction Inference, many previous works <b>mainly focused on features of the human as well as the directly interacted object</b>. 

C） Our insights: Not only the <b>primary relations</b> but also the <b>subsidiary relations</b> will provide significant cues to do intercation inference: <b>Contextual Information</b>.

- Consider the left image on Fig. b. If just focusing on the features of the girl and the directly interacted knife, it seems enough for the model to infer the action 'hold' but the subsidiary relations that the spatial relationship in (knife, cake) or (knife,desk) can make the model more certain that the 'cut' action has the low probabilty to happen. 
- Consider the right image on Fig.b. If the model ignores the contextual infromation while just focus on the primary object pair (boy,knife), it is hard for it to distinguish whether the action is 'stab' or 'cut'. However, if we let the model know there is cake here(semantic message) as well as the spatial relationship of subsidiary object pairs (cake, knife), it can help the model to infer the correct action.

<img align='center' src='./assets/introduce_image.png' width='700' heigth='700'>

<!---------------------------------------------------------------------------------------------------------------->
## VS-GATs

we study the disambiguating power of subsidiary scene relations via a <b>double Graph Attention Network</b> that aggregates <b>visual-spatial, and semantic information</b> in parallel. The network uses attention to leverage primary and subsidiary contextual cues to gain additional disambiguating power.

<img align='center' src='./assets/vs_gats.png' width='1000'>

<b>Visual-Semantic Graph Attention Network</b>: After instance detection, a visual-spatial and a semantic graph are created. Node edge weights are dynamically through attention. We combine these graphs and then perform a readout step on box-pairs to infer all possible predicates between one subject and one object. 

<!---------------------------------------------------------------------------------------------------------------->
## Graph
### Preliminary
<p><font size='4'>A graph $G$ is defined as $G=(V, E)$ that consists of a set of $V$ nodes and a set of $E$ edges. Node features and edge features are denoted by $\mathbf{h}_v$ and $\mathbf{h}_e$ respectively. Let $v_i \in V$  be the $ith$ node and $e_{i,j}=(v_i,v_j) \in E$ be the directed edge from $v_i$ to $v_j$.</font></p>

<p><font size='4'>A graph with $n$ nodes has a node features matrix $\mathbf{X}_v \in \mathbf{R}^{n \times d}$ and an edge feature matrix $ \mathbf{X}_e \in \mathbf{R}^{m \times c} $ where $\mathbf{h}_{v_i} \in \mathbf{R}^d$ is the feature vector of node $i$ and $\mathbf{h}_{e_{i,j}} \in \mathbf{R}^c $ is the feature vector of edge $(i,j)$. Fully connected edges imply $e_{i,j} \neq e_{j,i}$. </font></p>

### DGL Basics
<p><font size='4'><a href='https://docs.dgl.ai/en/latest/install/index.html'>DGL</a> is a Python package dedicated to deep learning on graphs, built atop existing tensor DL frameworks (e.g. Pytorch, MXNet) and simplifying the implementation of graph-based neural networks.</font></p>

<!---------------------------------------------------------------------------------------------------------------->
## Code Overview
In this project, there are three main folders: 

- `datasets/`: contains codes to prepare the train/test data;
- `model/`: contains codes to constructe the model;
- `result/`: contains codes to evalute the model;

In the following, we 
### Features Generation
<p><font size='4'>Main file :<code>hico_process.sh</code></font></p>

<ol>
    <li><font size='4'><b>Visual Features</b>:<code>run_faster_rcnn.py  select_confident_boxes.py  hico_train_val_test_data.py</code></font></li>
    <li><font size='4'><b>Spatial Features</b>:<code>spatial_feature.py</code></font></li>
    <li><font size='4'><b>Word embeddings</b>:<code>hico_word2vec.py</code></font></li>
</ol>

