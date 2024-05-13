from lingam.utils import make_prior_knowledge
import numpy as np
from sklearn.metrics import matthews_corrcoef
from scipy.stats import pearsonr
from scipy import stats
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.estimators import PC
import lingam
import pandas as pd
from lingam.utils import make_prior_knowledge
import networkx as nx
import matplotlib.pyplot as plt
import subprocess

class dag_structures:
    def __init__(self, flattened_structures_found_p, flattened_structures_found_v, problematic_variables, hidden_variables, blocked_variables, unproblematic_variables):
        self.problematic_structures = flattened_structures_found_p
        self.problematic_variables = problematic_variables
        self.missing_structures = flattened_structures_found_v
        self.missing = hidden_variables
        self.blocked = blocked_variables
        self.not_connected = unproblematic_variables



def unnest_list(nested_list):
    '''
    helper function used in other function
    '''
    flattened_list = []
    for element in nested_list:
        if isinstance(element, list):
            flattened_list.extend(unnest_list(element))
        else:
            flattened_list.append(element)
    return flattened_list

def check_tuples(tuples_list):
    '''
    This is a helper function for orientCPDAG. It takes the edge list as input and returns a modified version.
    It orients all edges that can be oriented given the new information and does so repeatedly.
    '''
    continue_loop = True
    while continue_loop == True:
        cons = False
        for tuple_ in tuples_list:
            first_element, second_element = tuple_
            if (second_element,first_element) not in tuples_list:

                matching_tuples = [(x, y) for (x, y) in tuples_list if y == second_element and (x, y) != tuple_]
                matching_tuples2 = [(x, y) for (x, y) in tuples_list if x == second_element and (y, x) != tuple_]

                for matching_tuple in matching_tuples:
                    reversed_tuple = (matching_tuple[1], matching_tuple[0])
                    if reversed_tuple in matching_tuples2:
                        tuples_list.remove(matching_tuple)
                        cons = True
        if cons == False:
            continue_loop = False
                    
    return tuples_list

def orientCPDAG(edge_list, black_list):
    '''
    This function takes an edge list and a black list and returns only those edges that 
    a) were already oriented or be
    b) that were unoriented and are allowed (i.e., can be oriented)
    '''
    unique_tuples = set()
    tuples_to_remove = set()
    edge_list = list(edge_list)
    for item in edge_list:
        reverse_tuple = (item[1], item[0])
        if reverse_tuple in unique_tuples:
            if item in black_list:
                tuples_to_remove.add(item)
            if reverse_tuple in black_list:
                tuples_to_remove.add(reverse_tuple)
        unique_tuples.add(item)

    for item in tuples_to_remove:
        edge_list.remove(item)
    
    edge_list = check_tuples(edge_list)
    
    return edge_list

def create_adjacency_matrix(data, edges):
    '''
    Takes as input the data and the edge_list from HC and PC.
    It creates an adjacency matrix. We try to infer whether relationships are positive or negative by 
    performing correlation tests and then make the values -1/1 respectively. 
    '''
    nodes = list(data.columns.values)
    adjacency_matrix = np.zeros((len(nodes), len(nodes)), dtype=int)

    index_mapping = {value: index for index, value in enumerate(nodes)}

    for edge in edges:
        row_index = index_mapping[edge[0]]
        col_index = index_mapping[edge[1]]
        adjacency_matrix[row_index, col_index] = 1

    for edge in edges:
        X = data[edge[0]]
        y = data[edge[1]]
        xn = X.nunique()
        yn = y.nunique()
        if yn == 2:
            if xn == 2:
                corr_val = matthews_corrcoef(X, y)
            else:
                res = stats.spearmanr(X, y)
                corr_val = res.statistic
        else:
            if xn == 2:
                res = stats.spearmanr(X, y)
                corr_val = res.statistic
            else:
                corr_val, _ = pearsonr(X, y)
        if corr_val < 0:
            adjacency_matrix[index_mapping[edge[0]], index_mapping[edge[1]]] = -1
        else:
            adjacency_matrix[index_mapping[edge[0]], index_mapping[edge[1]]] = 1

    return adjacency_matrix, nodes


def create_prior_knowledge_lingam(black_list, data, fixed_edges = []):
    '''
    This function takes a black list created by SetTIERS as input together with a dataset 
    and returns the prior knowledge necessary for calling LiNGAM.
    - optional parameter: fixed_list (list of definitely fixed relationships in the graph)
    '''
    n_variables = len(data.columns)
    variable_indices = dict(zip(data.columns, range(len(data.columns))))
    black_list_indices = [(variable_indices[key1], variable_indices[key2]) for key1, key2 in black_list]
    black_tuples = tuple(black_list_indices)
    if not fixed_edges:
        prior_knowledge = make_prior_knowledge(n_variables=n_variables, no_paths=black_tuples)
        return prior_knowledge
    else:
        fixed_list_indices = [(variable_indices[key1], variable_indices[key2]) for key1, key2 in fixed_edges]
        fixed_tuples = tuple(fixed_list_indices)
        prior_knowledge = make_prior_knowledge(n_variables=n_variables, paths=fixed_tuples, no_paths=black_tuples)
        return prior_knowledge

def setTIERS(data, list_of_tiers, in_tier_connection=False, unspecified="ignore"):
    '''
    This function takes as input:
    - the data
    - a list of tiers: a nested list ordered according to the tiers, with the first sublist not allowed to have any any children sppecified in further sublists
    - a parameter that specifies whether nodes within a TIER can be connected (default is False)
    - a parameter that states what should happen with variables not metioned
    - unspecified may be "end", "start" or "ignore" (default)
    '''

    if unspecified == "end" or unspecified == "start":
        col_names = list(data.columns.values)
        unnested_list_of_tiers = unnest_list(list_of_tiers)
        end_tier = []
        for n in col_names:
            if n not in unnested_list_of_tiers:
                end_tier.append(n)
        if unspecified == "end":
            list_of_tiers.append(end_tier)
        else:
            list_of_tiers.insert(0, end_tier)

        
    if in_tier_connection == False:
        black_list = []
        for i in range(len(list_of_tiers)):
            for j in range(len(list_of_tiers[i])):
                current_element = list_of_tiers[i][j]
                for k in range(i + 1, len(list_of_tiers)):
                    for l in range(len(list_of_tiers[k])):
                        black_list.append((current_element, list_of_tiers[k][l]))
        return black_list
    else:
        black_list = []
        for i in range(len(list_of_tiers)):
            for j in range(len(list_of_tiers[i])):
                current_element = list_of_tiers[i][j]
                for k in range(i + 1, len(list_of_tiers)):
                    for l in range(len(list_of_tiers[k])):
                        black_list.append((current_element, list_of_tiers[k][l]))
                for l in range(j + 1, len(list_of_tiers[i])):
                    black_list.append((current_element, list_of_tiers[i][l]))
                    black_list.append((list_of_tiers[i][l], current_element))
        return black_list


def learn_DAG(data, method, dominant_data_type, tiers=[], in_tier_connection=False, unspecified="ignore", further_blacklist=[], fixed_edges=[]):
    '''
    This function takes the data and the background information and learns a DAG. It takes as input:
    - data
    - method: one of {"LiNGAM","PC","HC"}
    - dominant_data_type: one of {"discrete","continuous"}
    - tiers: list of tiers as required for setTIERS()
    - the two parameters optionally taken in setTIERS()
    - allows an additional further blacklist
    - allows a set of fixed egdes of format: [("X","Y")]
    '''

    if fixed_edges:
        if method == "PC":
            return print("Error: PC does not work with fixed_edges right now.")

    black_list =[]
    if tiers:
        black_list = setTIERS(data=data, list_of_tiers=tiers, in_tier_connection=in_tier_connection, unspecified=unspecified)
    if further_blacklist:
        for e in further_blacklist:
            black_list.append(e)
    
    if method == "HC":
        if dominant_data_type == "discrete":
            est = HillClimbSearch(data)
            model = est.estimate(scoring_method=BicScore(data),black_list=black_list, fixed_edges=fixed_edges)
            edge_list = model.edges()
            adjacency_matrix, nodes = create_adjacency_matrix(data, edge_list)
        else:
            df_black_list = pd.DataFrame(black_list, columns=['Column1', 'Column2'])

            data.to_csv("Storage/data.csv", index=False)
            df_black_list.to_csv("Storage/black_list.csv", index=False)

            subprocess.call("Rscript Storage/bnlearn_hc.R", shell=True)

            edge_df = pd.read_csv("Storage/edge_df.csv")
            edge_list = [(row['from'], row['to']) for _, row in edge_df.iterrows()]
            adjacency_matrix, nodes = create_adjacency_matrix(data, edge_list)
    
    elif method == "PC":
        if dominant_data_type == "discrete":
            est = PC(data)
            model = est.estimate(ci_test='chi_square', return_type="cpdag")
            edge_list = model.edges()
            edge_list = orientCPDAG(edge_list, black_list)
            adjacency_matrix, nodes = create_adjacency_matrix(data, edge_list)
        else:
            data.to_csv("Storage/data.csv", index=False)
            subprocess.call("Rscript Storage/bnlearn_pc.R", shell=True)

            edge_df = pd.read_csv("Storage/edge_df.csv")
            edge_list = [(row['from'], row['to']) for _, row in edge_df.iterrows()]

            edge_list = orientCPDAG(edge_list, black_list)
            adjacency_matrix, nodes = create_adjacency_matrix(data, edge_list)

    else: 
        nodes = list(data.columns.values)
        prior_knowledge = create_prior_knowledge_lingam(black_list=black_list, data=data, fixed_edges = fixed_edges)
        model = lingam.DirectLiNGAM(prior_knowledge=prior_knowledge)
        model.fit(data)
        adj_mat = model.adjacency_matrix_
        adjacency_matrix = np.where(adj_mat < 0, -1, np.where(adj_mat > 0, 1, 0))
        adjacency_matrix = np.transpose(adjacency_matrix)

    return adjacency_matrix, nodes

def draw_dag(dag):
    '''
    Draws the graph with the information outputted by learn_DAG(): adjacency_matrix, nodes
    '''
    G = nx.DiGraph()
    nodes = dag[1]
    adjacency_matrix = dag[0]
    G.add_nodes_from(nodes)

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if adjacency_matrix[i, j] != 0:
                G.add_edge(nodes[i], nodes[j])

    pos = nx.shell_layout(G) 
    nx.draw(G, pos, with_labels=True, font_weight='bold', arrowsize=20, node_size=700, node_color='lightblue')
    plt.show()

def draw_problematic_dag(dag, highlight_path=None, highlight_path2 = None):
    '''
    Draws the graph with the information outputted by learn_DAG() andd identify_structures.
    Highlights problematic or asked for paths.
    '''
    G = nx.DiGraph()
    nodes = dag[1]
    adjacency_matrix = dag[0]
    G.add_nodes_from(nodes)

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if adjacency_matrix[i, j] != 0:
                edge_color = 'black'  # Default color for non-highlighted edges

                if highlight_path is not None:
                    for p in highlight_path:
                        # Check if the edge is part of the highlight_path
                        if nodes[i] in p and nodes[j] in p:
                            edge_color = 'red'  # Set color to red for the highlighted path
                if highlight_path2 is not None:
                    for p in highlight_path2:
                        # Check if the edge is part of the highlight_path
                        if nodes[i] in p and nodes[j] in p:
                            edge_color = 'green'  # Set color to red for the highlighted path

                G.add_edge(nodes[i], nodes[j], color=edge_color)

    pos = nx.shell_layout(G)
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    
    nx.draw(G, pos, with_labels=True, font_weight='bold', arrowsize=20, node_size=700, node_color='lightblue', edge_color=edge_colors)
    plt.show()



def specifiy_DAG(adjacency_matrix, nodes):
    '''
    takes as input an adjacency matrix specified by a user (with 1 for a positive and -1 for a negative relationship) 
    and returns it as well as the graph representation
    '''
    draw_dag(adjacency_matrix)


def iter_paths(adj, min_length=2, path=None):
    '''
    Performs Depth-First-Search to find all paths from the adjacency matrix;
    Returns all paths, only used inside remove_sublists().
    '''
    if not path:
        for start_node in range(len(adj)):
            yield from iter_paths(adj, min_length, [start_node])
    else:
        if len(path) >= min_length:
            yield path
        if path[-1] in path[:-1]:  
            return
        current_node = path[-1]
        for next_node in range(len(adj[current_node])):
            if adj[current_node][next_node] == 1 or adj[current_node][next_node] == -1:
                yield from iter_paths(adj, min_length, path + [next_node])
                
def remove_sublists(adjacency_matrix):
    '''
    Removes subpaths found with the DFS algorithm and returns only the complete paths.
    Used inside identify_structures().
    '''
    lst = list(iter_paths(adjacency_matrix))
    helper_lst = [[str(e) for e in sublist] for sublist in lst]
    str_list = [''.join(sublist) for sublist in helper_lst]
    
    entailed_strings = []
    for i in range(len(str_list)):
        for j in range(len(str_list)):
            if i != j and str_list[i] in str_list[j]:
                entailed_strings.append(str_list[i])
    
    entailed_strings = list(set(entailed_strings))
    
    helper_lst2 = [list(e) for e in entailed_strings]
    removal_lst = [[int(e) for e in sublist] for sublist in helper_lst2]

    result = [sublist_A for sublist_A in lst if sublist_A not in removal_lst]
    return result 


def identify_structures(adjacency_matrix, nodes, sensitive_variables, target_variable, draw_problematic = False):
    '''
    Takes as input the adjacency list produced before, a list of nodes with the same order as the variables are in the 
    adjacency matrix, a list of sensitive variables, and the target variable.
    It first uses the DFS algorithm and removes the encompassed paths and from the result finds 
    a) potentially problematic structures and
    b) structures that might indicate a missing variable that we can specifically ask for.
    '''
    all_paths = remove_sublists(adjacency_matrix)
    paths = []
    for sublist in all_paths:
        updated_sublist = [nodes[index] for index in sublist]
        paths.append(updated_sublist)

    problematic_structures = []
    ask_for_hidden = []
    for a in sensitive_variables:
        a_structure = []
        for path in paths:
            if a in path and target_variable in path:
                a_structure.append(path)
        if all(len(sublist) == 2 for sublist in a_structure):
            problematic_structures.append(a_structure)
        elif not any(len(sublist) == 2 for sublist in a_structure):
            problematic_structures.append(a_structure)
        else:
            effect_indirect = []
            for ai in range(len(a_structure)):
                struc = a_structure[ai]
                a_ix = nodes.index(a)
                b_ix = nodes.index(struc[1])
                c_ix = nodes.index(struc[-1])
                val1 = adjacency_matrix[a_ix][b_ix]
                val2 = adjacency_matrix[b_ix][c_ix]
                if len(struc) == 2:
                    effect_direct = val1
                else:
                    ix1 = a_ix
                    vals_path = []
                    for s in range(1, len(struc)):
                        con = struc[s]
                        ix2 = nodes.index(con)
                        valx = adjacency_matrix[ix1][ix2]
                        vals_path.append(valx)
                        ix1 = ix2
                    effect_indirect.append((ai, vals_path))

            # neu
            for i in range(len(effect_indirect)):
                ef = effect_indirect[i][1]
                if effect_direct == 1 and all(e == -1 for e in ef):
                    ask_for_hidden.append([a_structure[i]])
                elif effect_direct == -1 and all(e == 1 for e in ef):
                    ask_for_hidden.append([a_structure[i]])
                else:
                    problematic_structures.append([a_structure[i]])

    # the new part
    problematic_variables = []
    flattened_structures_found_p = [item for sublist in problematic_structures for item in sublist]
    for a in sensitive_variables:
        for f in flattened_structures_found_p:
            if a in f and a not in problematic_variables:
                problematic_variables.append(a)
    hidden_variables = []
    flattened_structures_found_v = [item for sublist in ask_for_hidden for item in sublist]
    for a in sensitive_variables:
        for f in flattened_structures_found_v:
            if a in f and a not in hidden_variables:
                hidden_variables.append(a)

    remaining_sensitives = [elem for elem in sensitive_variables if elem not in problematic_variables and elem not in hidden_variables]
    blocked_variables = []
    if remaining_sensitives:
        for rem in remaining_sensitives:
            all_vars_to_look_for = []
            found = False
            for path in paths:
                if rem in path:
                    vars_to_look_for = path
                    vars_to_look_for.remove(rem)
                    all_vars_to_look_for = all_vars_to_look_for + vars_to_look_for
            all_vars_to_look_for2 = []
            for v in all_vars_to_look_for:
                for path in paths:
                    if v in path:
                        vars_to_look_for2 = path
                        vars_to_look_for2.remove(v)
                        all_vars_to_look_for2 = all_vars_to_look_for2 + vars_to_look_for2

            for v in all_vars_to_look_for2:
                for path in paths:
                    if v in path and target_variable in path:
                        blocked_variables.append(rem)
                        found = True
                        break
                if found == True:
                    break

    no_connection_variables = [elem for elem in sensitive_variables if elem not in problematic_variables and elem not in hidden_variables and elem not in blocked_variables]
    structures = dag_structures(flattened_structures_found_p, flattened_structures_found_v, problematic_variables, hidden_variables, blocked_variables, no_connection_variables)

    if draw_problematic == True:
        draw_problematic_dag((adjacency_matrix,nodes), highlight_path=flattened_structures_found_p, highlight_path2 = ask_for_hidden)
    print(f"The following variables are likely problematic: {problematic_variables}.")
    print(f"They are part of the following paths: {flattened_structures_found_p}.")
    print(f"The following variables are potentially dangerous to remove: {hidden_variables}.")
    print(f"They are part of the following paths: {flattened_structures_found_v}.")
    print(f"The following variables are blocked and unproblematic: {blocked_variables}.")
    print(f"The following variables are not connected and unproblematic: {no_connection_variables}.")
    return structures

def draw_ground_truth_graph(edges):
    '''
    draws the ground truth dags that are provided by the pgmpy packages.
    '''
    dag = nx.DiGraph()
    dag.add_edges_from(edges)

    pos = nx.shell_layout(dag)
    nx.draw(dag, pos, with_labels=True, font_weight='bold', node_size=700, node_color='skyblue', arrowsize=20)
    plt.show()


def compare_to_groundtruth(dag, edges):
    '''
    Outputs a comparison between the ground truth graph and the predicted graph
    '''
    G = nx.DiGraph()
    nodes = dag[1]
    adjacency_matrix = dag[0]
    G.add_nodes_from(nodes)

    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if adjacency_matrix[i, j] != 0:
                G.add_edge(nodes[i], nodes[j])

    dag = nx.DiGraph()
    dag.add_edges_from(edges)

    correct_edges = set(dag.edges()) & set(G.edges())
    wrong_edges = set(G.edges()) - set(dag.edges())
    missing_edges = set(dag.edges()) - set(G.edges())

    print(f"Correct Edges: {len(correct_edges)}")
    print(f"Wrong Edges: {len(wrong_edges)}")
    print(f"Missing Edges: {len(missing_edges)}")
    return correct_edges, wrong_edges, missing_edges, len(edges)

def compare_dags(dag1, dag2):
    '''
    Outputs a comparison between two DAGs. Compares the second against the first.
    '''
    G1 = nx.DiGraph()
    nodes1 = dag1[1]
    adjacency_matrix1 = dag1[0]
    G1.add_nodes_from(nodes1)

    for i in range(len(nodes1)):
        for j in range(len(nodes1)):
            if adjacency_matrix1[i, j] != 0:
                G1.add_edge(nodes1[i], nodes1[j])

    G2 = nx.DiGraph()
    nodes2 = dag2[1]
    adjacency_matrix2 = dag2[0]
    G2.add_nodes_from(nodes2)

    for i in range(len(nodes2)):
        for j in range(len(nodes2)):
            if adjacency_matrix2[i, j] != 0:
                G2.add_edge(nodes2[i], nodes2[j])

    correct_edges = set(G2.edges()) & set(G1.edges())
    wrong_edges = set(G2.edges()) - set(G1.edges())
    missing_edges = set(G1.edges()) - set(G2.edges())

    print(f"Matching Edges: {len(correct_edges)}")
    print(f"Edges DAG2 has DAG1 has not: {len(wrong_edges)}")
    print(f"Edges DAG1 has DAS2 has not: {len(missing_edges)}")
    return correct_edges, wrong_edges, missing_edges


