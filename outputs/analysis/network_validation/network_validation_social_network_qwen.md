# Structural validation: social_network_qwen

nodes: 100   directed edges: 824   density: 0.083
out-degree (follows):   min 3  median 8  mean 8.2  max 22
in-degree (followers):  min 0  median 7  mean 8.2  max 27
agents nobody follows (posts unread): 1 -> agent_005
agents following nobody (read nothing): 0
weak components: 1 (largest = 100 nodes)
reciprocity (edge has reverse edge): 0.26
mean local clustering (undirected): 0.28

## Homophily vs node-label permutation null (n=2000, seed 42)
attribute                     observed      null       z        p
age (|gap| in years)             6.508     7.400   -3.88   0.0005  (negative z = assortative: smaller age gaps than chance)
gender                           0.562     0.496    3.63   0.0005
relationship_status              0.465     0.374    4.60   0.0005
planning_area                    0.313     0.050   32.20   0.0005
education                        0.504     0.377    5.31   0.0005

Interpretation: 'same'-kind rows are the share of edges joining same-category agents (positive z = homophily); the age row is the mean absolute age gap across edges (negative z = homophily).
