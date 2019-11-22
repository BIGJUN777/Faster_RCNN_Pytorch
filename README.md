## Visual-Semantic Graph Attention Network for Human-Object Interaction Detecion

### Preamble
A） Generally, HOI detection includes two steps: <b>Object detection</b> && <b>Interaction Inference</b>.

B） As for Interaction Inference, many previous works <b>mainly focused on features of the human as well as the directly interacted object</b>. 

C） Our insights: Not only the <b>primary relations</b> but also the <b>subsidiary relations</b> will provide significant cues to do intercation inference: <b>Contextual Information</b>.

<ol>
    <li><font size='3'>Consider the left image on Fig. b. If just focusing on the features of the girl and the directly interacted knife, it seems enough for the model to infer the action 'hold' but the subsidiary relations that the spatial relationship in (knife, cake) or (knife,desk) can make the model more certain that the 'cut' action has the low probabilty to happen. </font></li>
    <li><font size='3'>Consider the right image on Fig.b. If the model ignores the contextual infromation while just focus on the primary object pair (boy,knife), it is hard for it to distinguish whether the action is 'stab' or 'cut'. However, if we let the model know there is cake here(semantic message) as well as the spatial relationship of subsidiary object pairs (cake, knife), it can help the model to infer the correct action.</font></li>
</ol>

<img align='center' src='./assets/introduce_image.png' width='700' heigth='700'>

### VS-GATs
                                         
<p><font size='4'>we study the disambiguating power of subsidiary scene relations via a <b>double Graph Attention Network</b> that aggregates <b>visual-spatial, and semantic information</b> in parallel. The network uses attention to leverage primary and subsidiary contextual cues to gain additional disambiguating power.</font></p>

<img align='center' src='./assets/vs_gats.png' width='1000'>

<p><font size='4'> <b>Visual-Semantic Graph Attention Network</b>: After instance detection, a visual-spatial and a semantic graph are created. Node edge weights are dynamically through attention. We combine these graphs and then perform a readout step on box-pairs to infer all possible predicates between one subject and one object.</font></p> 

## Code Overview

### Features
<p><font size='4'>Main file :<code>hico_process.sh</code></font></p>

<ol>
    <li><font size='4'><b>Visual Features</b>:<code>run_faster_rcnn.py  select_confident_boxes.py  hico_train_val_test_data.py</code></font></li>
    <li><font size='4'><b>Spatial Features</b>:<code>spatial_feature.py</code></font></li>
    <li><font size='4'><b>Word embeddings</b>:<code>hico_word2vec.py</code></font></li>
</ol>
