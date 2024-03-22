## causalfair

This is the repository to the Python package *causalfair*. It can learn DAGs using background knowledge and identify which variables might be problematic when we learn ML models on the data.

### Requirements
To run *causalfair*, simply download or clone the folder and make sure that R and Python are installed on your computer and you installed the following requirements.

For R:
- bnlearn

For Python:
- lingam==1.8.2
- numpy==1.24.4
- sklearn==1.2.2
- scipy==1.10.1
- pgmpy==0.1.20
- pandas==1.5.3
- networkx==2.8.8
- matplotlib==3.7.4

*causalfair* should be upwards compatible, so it should not be a problem if you have newer versions.

### Example
ExampleNotebook.ipynb displays the most important functionalities of causalfair.

### Documentation

**learn_DAG(data, method, dominant_data_type, tiers=[], in_tier_connection=False, unspecified="ignore", further_blacklist=[], fixed_edges=[])**
With this function you can learn a DAG from data and optionally background knowledge. It returns an adjacency matrix and the correpsonding list of nodes. 
You must set the following parameters:
- data: a pandas dataframe
- method: one of \{PC,HC,LiNGAM\}
- dominant_data_type: one of \{continuous,discrete\}

You can specify:
- tiers: A list of lists that encode the background information.The first tier is the target, the second normal predictive variables, and the third the demographic variables. You can ooptionally specify more tiers.
- in_tier_conection: States whether connections within a tier should also be restricted. If False, this is not the case.
- unspeciefied: one of \{start,end,ignore\}. States whether variables we have not mentioned should be ignored in the restrictions which means that relationships to and from then are not restricted at all or whether these variables should only be leave nodes (start) or root nodes (end).
- further_blacklist: A list of tuples with restrictions beyond the tiers.
- fixed_edges: A list of edges that must be in the resulting graph. Note that this only works for HC and LiNGAM.

**identify_structures(adjacency_matrix, nodes, sensitive_variables, target_variable, draw_problematic = False)**
This function takes as input the adjacency matrix and list of nodes produced by learn_DAG as well as the list of edmographic variables and the name of the target variable and returns an object.
Optionally, one can specify whether DAG should be drawn with problematic structures highlighted (draw_problematic).
Attributes of the objects can be called using these keywords:
- problematic_structures: returns a list of problematic paths found. 
- problematic_variables: returns a list of the problematic demographic variables
- missing_structures: returns a list of the paths that indicate a missing variable
- missing: returns a list of demographic variables that are affected by these missing variables
- blocked: returns a list of blocked demographic variables
- not_connected: returns a list of demographic variables with no connection

**compare_dags((adjacency_matrix, nodes), (adjacency_matrix, nodes))**
This function compares two DAGs with identical sets of nodes against each other. It takes the adjacency matrices of both.
It returns information on matching edges, edges that are present in DAG2 but not DAG 1, and vice versa.

**compare_to_groundtruth((adjacency_matrix, nodes), edges)**
This function compares two DAGs (one is the ground truth DAG) with identical sets of nodes against each other. It takes the adjacency matrix from one node and a list of edges from the other (the ground truth). It is designed to compare a learned DAG to a ground truth where the adjacency matrix is from the learned graph.
It returns information on correctly found, incorrectly found, and missing edges.

**draw_dag(adjacency_matrix,nodes)**
This function draws a dag given an adjacency matrix and a list of corresponding nodes. The function can be directly fed usig the output from learn_dag().

**draw_problematic_dag((adjacency_matrix, nodes), highlight_path=None, highlight_path2 = None)**
This function takes the learned DAG as input and returns a drawn DAG with highlighted path.
Either, we can highlight probelamtic structures passing the problematic structures from the object returned by identify_structures(). Or we highlight the variables involved in a structure indicating a missing variable passing the missing structures from the object returned by identify_structures().

**draw_ground_truth_graph(edges)**
This function takes an edge list as input and simply draws the graph. The edge list should be a list of tuples. The first element of each tuple is the parent node of the second element. See the notebook for an example.

