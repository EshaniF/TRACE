Python code for the paper 'LLM-Guided Relational Transition Graphs for Temporal Knowledge Graph Reasoning'

Note: Experiments were conducted on ICEWS14, ICEWS18, and GDELT. All datasets are public benchmarks.


Train models
Then the following commands can be used to train the proposed models. The evaluation results and visualizations will be generated when training completes.

cd src

Step 1: Build the Transition Graph (once per dataset)

python llm_transition_scorer.py -d ICEWS14

This queries Llama-3.1-8B once per relation and saves 
`rel_transition_graph.pkl` to `data/ICEWS14/`.

The LLM prompt used for transition graph construction is provided in `transition_prompt.txt`.

Step 2: Train

***Full model***

python maintr.py -d ICEWS14 --train-history-len 7 --test-history-len 7 --run-statistic --n-epochs 80 --evaluate-every 1 

***Draw heatmap with the following***

 python relation_similarity_heatmap.py  --ckpt-no-reg ../models/logcl_ICEWS14  --ckpt-with-reg ../models/ctstkg_ICEWS14_0.3 --num-rels 230  --relation2id ../data/ICEWS14/relation2id.txt  --transition-graph ../data/ICEWS14/rel_transition_graph.pkl --anchor-relation "Engage_in_negotiation" --top-k 3 --out relation_similarity.png

 ***Ablation***
 ***Mechanism A only (embedding regularisation, no re-ranking)***

python maintr.py -d ICEWS14 --use-transition --rerank-alpha 0.0

***Mechanism B only (re-ranking, no regularisation)***

python maintr.py -d ICEWS14 --use-transition --lambda-trans 0.0

*******************************************************************************************************************************
Code partially adapted from authors' implementation of LogCL, RE-GCN, TIRGN models.
W. Chen, H. Wan, Y. Wu, S. Zhao, J. Cheng, Y. Li, and Y. Lin,
“Local-global history-aware contrastive learning for temporal knowledge
graph reasoning,” in 2024 IEEE 40th International Conference on Data
Engineering (ICDE). IEEE, 2024, pp. 733–746.

Z. Li, X. Jin, W. Li, S. Guan, J. Guo, H. Shen, Y. Wang, and
X. Cheng, “Temporal knowledge graph reasoning based on evolutional
representation learning,” in Proceedings of the 44th international ACM
SIGIR conference on research and development in information retrieval,
2021, pp. 408–417.

Y. Li, S. Sun, and J. Zhao, “Tirgn: Time-guided recurrent graph network
with local-global historical patterns for temporal knowledge graph
reasoning.” in IJCAI, 2022, pp. 2152–2158.

*********************************************************************************************************************************
Claude (Anthropic) and Gemini (Google) were used as programming aids during development; All code was reviewed, tested, and modified by the author.



