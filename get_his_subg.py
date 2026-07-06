import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import time
import pickle
import pandas as pd
import random
import dgl
# from torch_geometric.data import Data
# from torch_geometric.utils import degree
# from torch_geometric.utils import to_networkx

def get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, triples,num_nodes, num_rels):
    ###original
    #calculation of inverse triplets
    # q_to_sro = defaultdict(list)
    q_to_sro = set()
    inverse_triples = triples[:, [2, 1, 0]]
    inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
    all_triples = np.concatenate([triples, inverse_triples])
    # ent_set = set(all_triples[:, 0])
    src_set = set(triples[:, 0])
    dst_set = set(triples[:, 2])

    # ---------------- Second-Order Neighbor Sampling -----------------------
    # er_list = list(set([(tri[0],tri[1]) for tri in all_triples]))
    er_list = list(set([(tri[0],tri[1]) for tri in triples]))
    er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
    # ent_list = list(ent_set)
    # rel_list = list(set(all_triples[:, 1]))

    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
    
    # Integrate repeated triples and count the frequency of triples, using the frequency of triples as the fourth column of data
    subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'}) 
    keys = list(sr_to_sro.keys()) #keys = (s,r)
    values = list(sr_to_sro.values()) # values = {o}
    df_dic =  pd.DataFrame({'sr': keys, 'dst': values}) #Convert query field to pandas; dictionary with (s, r) as keys and {o} as values
    #dataframe.query : Query the columns of a DataFrame with a boolean expression and give rows which satify the condition.
    #only get (s,r) in er_list (<- triplets) which appear in df_dic (<- sr_to_sro)
    dst_df = df_dic.query('sr in @er_list')  #Get query entities and relationships pandas
    dst_get = dst_df['dst'].values    #Get the target tail entity
    two_ent = set().union(*dst_get)   #find unique dst entities
    all_ent = list(src_set|two_ent)  #union without repetition 
    result = subg_df.query('src in @all_ent')

    dst_df_inv = df_dic.query('sr in @er_list_inv')  #Get query entities and relationships pandas
    dst_get_inv = dst_df_inv['dst'].values    #Get the target tail entity
    two_ent_inv = set().union(*dst_get_inv)   #Integrate the head entity with the tail entity
    all_ent_inv = list(dst_set|two_ent_inv)  
    result_inv = subg_df.query('src in @all_ent_inv')

    q_tri = result.to_numpy()
    q_tri_inv = result_inv.to_numpy()

    return  q_tri,q_tri_inv

# def get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, triples,num_nodes, num_rels):
#     # forest fire sampling with degree-biased selection. 
#     burn_prob=0.6
#     inverse_triples = triples[:, [2, 1, 0]]
#     inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
#     all_triples = np.concatenate([triples, inverse_triples])

#     inverse_subg = subg_arr[:, [2, 1, 0]]
#     inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
#     subg_triples = np.concatenate([subg_arr, inverse_subg])
#     df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
#     subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'freq'})
#     degrees = subg_df['src'].value_counts().to_dict()

#     def forest_fire():
#         visited_nodes = set()
#         sampled_triples = set()
#         # random.seed(42)
#         seed_node = random.choice(triples[:, 0])
#         frontier = [seed_node]  

#         while frontier:
#             next_frontier = []
#             for node in frontier:
#                 if node in visited_nodes:
#                     continue
#                 visited_nodes.add(node)

#                 # Get all triples starting from this node
#                 node_triples = subg_df[subg_df['src'] == node]
                
#                 # Spread to neighbors with degree-biased burn probability
#                 for _, triple in node_triples.iterrows():
#                     # Scale burn probability by destination node's degree
#                     neighbor_degree = degrees.get(triple['dst'], 1)
#                     degree_biased_prob = burn_prob * (neighbor_degree / max(degrees.values()))
                    
                    
#                     if random.random() < degree_biased_prob:
                    
                        
#                         sampled_triples.add((triple['src'], triple['rel'], triple['dst'], triple['freq']))
#                         next_frontier.append(triple['dst'])  # Spread to the destination node

#             frontier = next_frontier
        
#         # Convert sampled triples to df
#         sampled_df = pd.DataFrame(list(sampled_triples), columns=['src', 'rel', 'dst', 'freq'])
#         return sampled_df

#     # Retry sampling until both q_tri and q_tri_inv are non-empty
#     while True:
#         sampled_df = forest_fire()
#         q_tri = sampled_df[sampled_df['rel'] < num_rels]  # Filter for original triples
#         q_tri_inv = sampled_df[sampled_df['rel'] >= num_rels]  # Filter for inverse triples
#         if not q_tri.empty and not q_tri_inv.empty:
#             break  # Stop retrying if both are non-empty

#     return q_tri.reset_index(drop=True), q_tri_inv.reset_index(drop=True)

# def get_sample_from_history_graph(subg_arr, s_to_sro, sr_to_sro, sro_to_fre, triples, num_nodes, num_rels):
#     # Forest Fire Sampling with degree-biased selection and geometric distribution
#     burn_prob = 0.4

#     # Create inverse triples
#     inverse_triples = triples[:, [2, 1, 0]]
#     inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
#     all_triples = np.concatenate([triples, inverse_triples])

#     # Create inverse subgraph triples
#     inverse_subg = subg_arr[:, [2, 1, 0]]
#     inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
#     subg_triples = np.concatenate([subg_arr, inverse_subg])
    
#     # Convert to DataFrame
#     df = pd.DataFrame(subg_triples, columns=['src', 'rel', 'dst'])
#     subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'freq'})
#     degrees = subg_df['src'].value_counts().to_dict()  # Get degree of each source node

#     def forest_fire():
#         visited_nodes = set()
#         sampled_triples = set()
#         seed_node = random.choice(triples[:, 0])  # Pick a random starting node
#         frontier = [seed_node]  

#         while frontier:
#             next_frontier = []
#             for node in frontier:
#                 if node in visited_nodes:
#                     continue
#                 visited_nodes.add(node)

#                 # Get all triples starting from this node
#                 node_triples = subg_df[subg_df['src'] == node]

#                 if not node_triples.empty:
#                     # Calculate number of neighbors to burn using geometric distribution
#                     num_to_burn = max(1, np.random.geometric(burn_prob) - 1)  

#                     # Select the top-N neighbors based on degree bias
#                     sorted_neighbors = node_triples.sort_values(by='freq', ascending=False)  # Bias towards high-degree
#                     selected_triples = sorted_neighbors.head(num_to_burn)  # Select top-N neighbors

#                     for _, triple in selected_triples.iterrows():
#                         sampled_triples.add((triple['src'], triple['rel'], triple['dst'], triple['freq']))
#                         next_frontier.append(triple['dst'])  # Add new frontier nodes

#             frontier = next_frontier
        
#         # Convert sampled triples to DataFrame
#         sampled_df = pd.DataFrame(list(sampled_triples), columns=['src', 'rel', 'dst', 'freq'])
#         return sampled_df

#     # Retry sampling until both q_tri and q_tri_inv are non-empty
#     while True:
#         sampled_df = forest_fire()
#         q_tri = sampled_df[sampled_df['rel'] < num_rels]  # Original triples
#         q_tri_inv = sampled_df[sampled_df['rel'] >= num_rels]  # Inverse triples
#         if not q_tri.empty and not q_tri_inv.empty:
#             break  # Stop retrying if both are non-empty

#     return q_tri.reset_index(drop=True), q_tri_inv.reset_index(drop=True)

# def get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, triples,num_nodes, num_rels):
#     
#     #forest fire sampling 
#     
#     burn_prob=0.3
#     inverse_triples = triples[:, [2, 1, 0]]
#     inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
#     all_triples = np.concatenate([triples, inverse_triples])

#     inverse_subg = subg_arr[:, [2, 1, 0]]
#     inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
#     subg_triples = np.concatenate([subg_arr, inverse_subg])
#     df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
#     subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0: 'freq'})

#     def forest_fire():
#         visited_nodes = set()
#         sampled_triples = set()
        
#         # Start from a random seed node in triples
#         seed_node = random.choice(triples[:, 0])
#         frontier = [seed_node]  # Initialize frontier with the seed node

#         while frontier:
#             next_frontier = []
#             for node in frontier:
#                 if node in visited_nodes:
#                     continue
#                 visited_nodes.add(node)

#                 # Get all triples starting from this node
#                 node_triples = subg_df[subg_df['src'] == node]
                
#                 # Spread to neighbors with burn probability
#                 for _, triple in node_triples.iterrows():
#                     if random.random() < burn_prob:
#                         sampled_triples.add((triple['src'], triple['rel'], triple['dst'], triple['freq']))
#                         next_frontier.append(triple['dst'])  # Spread to the destination node

#             frontier = next_frontier
        
#         # Convert sampled triples to DataFrame
#         sampled_df = pd.DataFrame(list(sampled_triples), columns=['src', 'rel', 'dst', 'freq'])
#         return sampled_df

#     # Retry sampling until both q_tri and q_tri_inv are non-empty
#     while True:
#         sampled_df = forest_fire()
#         q_tri = sampled_df[sampled_df['rel'] < num_rels]  # Filter for original triples
#         q_tri_inv = sampled_df[sampled_df['rel'] >= num_rels]  # Filter for inverse triples
#         if not q_tri.empty and not q_tri_inv.empty:
#             break  # Stop retrying if both are non-empty

#     return q_tri.reset_index(drop=True), q_tri_inv.reset_index(drop=True)

# def forest_fire_sampling(df, burning_ratio=0.3, forward_prob=0.3, backward_prob=0.3):
    # Forest Fire sampling on a graph represented as a DataFrame
#     """
#     # Get unique nodes
#     all_nodes = set(df['src'].unique()) | set(df['dst'].unique())
    
#     # Initialize sets for burning and burned nodes
#     burning = set()
#     burned = set()
    
#     # Start with a random node
#     if len(all_nodes) > 0:
#         burning.add(random.choice(list(all_nodes)))
    
#     sampled_edges = []
    
#     while burning:
#         # Get a burning node
#         current_node = burning.pop()
#         burned.add(current_node)
        
#         # Get forward edges (outgoing)
#         forward_edges = df[df['src'] == current_node]
#         if not forward_edges.empty:
#             # Sample forward edges based on burning ratio
#             num_forward = int(max(1, len(forward_edges) * forward_prob))
#             sampled_forward = forward_edges.sample(n=min(num_forward, len(forward_edges)))
#             sampled_edges.extend(sampled_forward.values.tolist())
            
#             # Add unburned neighbors to burning set
#             for _, edge in sampled_forward.iterrows():
#                 if edge['dst'] not in burned and random.random() < burning_ratio:
#                     burning.add(edge['dst'])
        
#         # Get backward edges (incoming)
#         backward_edges = df[df['dst'] == current_node]
#         if not backward_edges.empty:
#             # Sample backward edges based on burning ratio
#             num_backward = int(max(1, len(backward_edges) * backward_prob))
#             sampled_backward = backward_edges.sample(n=min(num_backward, len(backward_edges)))
#             sampled_edges.extend(sampled_backward.values.tolist())
            
#             # Add unburned neighbors to burning set
#             for _, edge in sampled_backward.iterrows():
#                 if edge['src'] not in burned and random.random() < burning_ratio:
#                     burning.add(edge['src'])
    
#     return pd.DataFrame(sampled_edges, columns=['src', 'rel', 'dst'])

# def get_sample_from_history_graph(subg_arr, triples, num_nodes, num_rels):
#     # Immediate for ff sampling above
#     inverse_triples = triples[:, [2, 1, 0]]
#     inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
#     all_triples = np.concatenate([triples, inverse_triples])
    
#     er_list = list(set([(tri[0], tri[1]) for tri in triples]))
#     er_list_inv = list(set([(tri[0], tri[1]) for tri in inverse_triples]))
    
#     inverse_subg = subg_arr[:, [2, 1, 0]]
#     inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
#     subg_triples = np.concatenate([subg_arr, inverse_subg])
    
#     df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
#     subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'})
#     subg_df['sr'] = list(zip(subg_df.src, subg_df.rel))
    
#     # Filter and apply Forest Fire sampling
#     result_df = subg_df[subg_df['sr'].isin(er_list)]
#     result_df = result_df.drop(columns=['sr'])
#     sampled_df = forest_fire_sampling(result_df)
#     q_tri = sampled_df.to_numpy()
    
#     # Process inverse triples
#     result_df_inv = subg_df[subg_df['sr'].isin(er_list_inv)]
#     result_df_inv = result_df_inv.drop(columns=['sr'])
#     sampled_df_inv = forest_fire_sampling(result_df_inv)
#     q_tri_inv = sampled_df_inv.to_numpy()
    
#     return q_tri, q_tri_inv

# def get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, triples,num_nodes, num_rels):
# ### weighted sampling for src nodes
#     np.random.seed(42)
#     inverse_triples = triples[:, [2, 1, 0]]
#     inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
#     all_triples = np.concatenate([triples, inverse_triples])
#     # ent_set = set(all_triples[:, 0])
#     # ent_set = set(all_triples[:, 0])
#     src_set = set(triples[:, 0])
#     src_list = list(src_set)
#     dst_set = set(triples[:, 2])
#     dst_list = list(dst_set)

#     inverse_subg = subg_arr[:, [2, 1, 0]]
#     inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
#     subg_triples = np.concatenate([subg_arr, inverse_subg])

#     src, rel, dst= subg_triples.transpose()
#     src = torch.tensor(src, dtype=torch.long)
#     dst = torch.tensor(dst, dtype=torch.long)
#     datag = dgl.graph((src, dst))
#     networkX_graph = dgl.to_networkx(datag)
#     degrs = {node:val for (node, val) in networkX_graph.degree()}

#     df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
#     # df['deg'] = np.NaN
#     df['deg'] = df['src'].apply(lambda x: degrs.get(x))
#     subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'}) 
#     subg_df = subg_df.sample(frac=0.2, replace=False, weights='deg', random_state=42)
#     # subg_df['sr'] = list(zip(subg_df.src, subg_df.rel))
#     subg_df = subg_df.drop(columns=['deg'])

#     # er_list = list(set([(tri[0],tri[1]) for tri in all_triples]))
#     er_list = list(set([(tri[0],tri[1]) for tri in triples]))
#     er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
#     # ent_list = list(ent_set)
#     # rel_list = list(set(all_triples[:, 1]))

#     result = subg_df[subg_df['src'].isin(src_list)]  #Get query entities and relationships pandas

#     result_inv = subg_df[subg_df['src'].isin(dst_list)]  #Get query entities and relationships pandas

#     q_tri = result.to_numpy()
#     q_tri_inv = result_inv.to_numpy()

#     return  q_tri,q_tri_inv

# def get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, triples,num_nodes, num_rels):
# ### simple random sampling 
#     random.seed(42)
#     #calculation of inverse triplets
#     # q_to_sro = defaultdict(list)
#     q_to_sro = set()
#     inverse_triples = triples[:, [2, 1, 0]]
#     inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
#     all_triples = np.concatenate([triples, inverse_triples])
#     # ent_set = set(all_triples[:, 0])
#     src_set = set(triples[:, 0])
#     dst_set = set(triples[:, 2])

#     # ---------------- Second-Order Neighbor Sampling -----------------------
#     # er_list = list(set([(tri[0],tri[1]) for tri in all_triples]))
#     er_list = list(set([(tri[0],tri[1]) for tri in triples]))
#     er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
#     # ent_list = list(ent_set)
#     # rel_list = list(set(all_triples[:, 1]))

#     inverse_subg = subg_arr[:, [2, 1, 0]]
#     inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
#     subg_triples = np.concatenate([subg_arr, inverse_subg])
#     df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
#     #Integrate repeated triplets and count the frequency of triplets, The frequency of the triples is used as the fourth column of data
#     subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'}) 
#     q_tri = subg_df.sample(frac=0.022, random_state=42).to_numpy()
#     q_tri_inv = subg_df.sample(frac=0.022, random_state=42).to_numpy()

#     return  q_tri,q_tri_inv

# def get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, triples,num_nodes, num_rels):
# ### random sampling from first hop and connecting second hop
#     random.seed(42)
#     #calculation of inverse triplets
#     # q_to_sro = defaultdict(list)
#     q_to_sro = set()
#     inverse_triples = triples[:, [2, 1, 0]]
#     inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
#     all_triples = np.concatenate([triples, inverse_triples])
#     # ent_set = set(all_triples[:, 0])
#     src_set = set(triples[:, 0])
#     dst_set = set(triples[:, 2])

#     # ---------------- Second-Order Neighbor Sampling -----------------------
#     # er_list = list(set([(tri[0],tri[1]) for tri in all_triples]))
#     er_list = list(set([(tri[0],tri[1]) for tri in triples]))
#     er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
#     # ent_list = list(ent_set)
#     # rel_list = list(set(all_triples[:, 1]))

#     inverse_subg = subg_arr[:, [2, 1, 0]]
#     inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
#     subg_triples = np.concatenate([subg_arr, inverse_subg])
#     df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
#     #Integrate repeated triplets and count the frequency of triplets, The frequency of the triples is used as the fourth column of data
#     subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'}) 
#     keys = list(sr_to_sro.keys())
#     values = list(sr_to_sro.values())
#     df_dic =  pd.DataFrame({'sr': keys, 'dst': values}) #Convert the query field to pandas

#     dst_df = df_dic.query('sr in @er_list')  #Get query entities and relationships pandas
#     dst_dff = dst_df.sample(frac=0.3, replace=False, random_state=42)
#     dst_get = dst_dff['dst'].values    #Get the target tail entity
#     two_ent = set().union(*dst_get)   #Integrate the head entity with the tail entity
#     two_ent = set(random.sample(list(two_ent), k=round(len(two_ent) * 0.3)))
#     all_ent = list(src_set|two_ent)   
#     result = subg_df.query('src in @all_ent')

#     dst_df_inv = df_dic.query('sr in @er_list_inv')  #Get query entities and relationships pandas
#     dst_dff_inv = dst_df_inv.sample(frac=0.1, replace=False, random_state=42)
#     dst_get_inv = dst_dff_inv['dst'].values    #Get the target tail entity
#     two_ent_inv = set().union(*dst_get_inv)   #Integrate the head entity with the tail entity
#     two_ent_inv = set(random.sample(list(two_ent_inv), k=round(len(two_ent_inv) * 0.3)))
#     all_ent_inv = list(dst_set|two_ent_inv)  
#     result_inv = subg_df.query('src in @all_ent_inv')

#     q_tri = result.to_numpy()
#     q_tri_inv = result_inv.to_numpy()

#     return  q_tri,q_tri_inv

# def get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, triples,num_nodes, num_rels):
#     ## stratified sampling strata with relations
#     inverse_triples = triples[:, [2, 1, 0]]
#     inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
#     all_triples = np.concatenate([triples, inverse_triples])
#     ent_set = set(all_triples[:, 0])
#     src_set = set(triples[:, 0])
#     src_list = list(src_set)
#     dst_set = set(triples[:, 2])
#     dst_list = list(dst_set)
#     er_list = list(set([(tri[0],tri[1]) for tri in triples]))
#     er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
    
    
#     inverse_subg = subg_arr[:, [2, 1, 0]]
#     inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
#     subg_triples = np.concatenate([subg_arr, inverse_subg])
#     df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
#     subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'}) 

#     result_df = subg_df[subg_df['src'].isin(src_list)]
#     result =  result_df.groupby('rel').sample(frac=0.3, random_state=42)
#     q_tri = result.to_numpy()

#     result_df_inv = subg_df[subg_df['src'].isin(dst_list)]
#     result_inv = result_df_inv.groupby('rel').sample(frac=0.3, random_state=42)
#     q_tri_inv = result_inv.to_numpy()

#     return  q_tri,q_tri_inv

# def get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, triples,num_nodes, num_rels):
# ###random sampling from 1st hop only (no second hop)
#     q_to_sro = set()
#     inverse_triples = triples[:, [2, 1, 0]]
#     inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
#     all_triples = np.concatenate([triples, inverse_triples])
#     ent_set = set(all_triples[:, 0])
#     src_set = set(triples[:, 0])
#     dst_set = set(triples[:, 2])
#     er_list = list(set([(tri[0],tri[1]) for tri in triples]))
#     er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
    

#     inverse_subg = subg_arr[:, [2, 1, 0]]
#     inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
#     subg_triples = np.concatenate([subg_arr, inverse_subg])
#     df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
#     subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'}) 
#     subg_df['sr'] = list(zip(subg_df.src, subg_df.rel))

#     result_df = subg_df[subg_df['sr'].isin(er_list)]
#     result_df = result_df.drop(columns=['sr'])
#     result = result_df.sample(frac=0.01, replace=False, random_state=42)
#     q_tri = result.to_numpy()

#     result_df_inv = subg_df[subg_df['sr'].isin(er_list_inv)]
#     result_df_inv = result_df_inv.drop(columns=['sr'])
#     result_inv = result_df_inv.sample(frac=0.01, replace=False, random_state=42)
#     q_tri_inv = result_inv.to_numpy()

#     return  q_tri,q_tri_inv

# def get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, triples,num_nodes, num_rels):
# ### original upto and including all 2nd hop
#     random.seed(42)
#     #calculation of inverse triplets
#     # q_to_sro = defaultdict(list)
#     q_to_sro = set()
#     inverse_triples = triples[:, [2, 1, 0]]
#     inverse_triples[:, 1] = inverse_triples[:, 1] + num_rels
#     all_triples = np.concatenate([triples, inverse_triples])
#     # ent_set = set(all_triples[:, 0])
#     src_set = set(triples[:, 0])
#     dst_set = set(triples[:, 2])

#     # ---------------- Second-Order Neighbor Sampling -----------------------
#     # er_list = list(set([(tri[0],tri[1]) for tri in all_triples]))
#     er_list = list(set([(tri[0],tri[1]) for tri in triples]))
#     er_list_inv = list(set([(tri[0],tri[1]) for tri in inverse_triples]))
#     # ent_list = list(ent_set)
#     # rel_list = list(set(all_triples[:, 1]))

#     inverse_subg = subg_arr[:, [2, 1, 0]]
#     inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
#     subg_triples = np.concatenate([subg_arr, inverse_subg])
#     df = pd.DataFrame(np.array(subg_triples), columns=['src', 'rel', 'dst'])
#     #Integrate repeated triplets and count the frequency of triplets, The frequency of the triples is used as the fourth column of data
#     subg_df = df.groupby(df.columns.tolist()).size().reset_index().rename(columns={0:'freq'}) 
#     keys = list(sr_to_sro.keys())
#     values = list(sr_to_sro.values())
#     df_dic =  pd.DataFrame({'sr': keys, 'dst': values}) #Convert the query field to pandas

#     dst_df = df_dic.query('sr in @er_list')  #Get query entities and relationships pandas
#     dst_get = dst_df['dst'].values    #Get the target tail entity
#     two_ent = set().union(*dst_get)   #Integrate the head entity with the tail entity
#     two_ent = set(random.sample(list(two_ent), k=round(len(two_ent) * 0.3)))
#     all_ent = list(src_set|two_ent)   
#     result = subg_df.query('src in @all_ent')

#     dst_df_inv = df_dic.query('sr in @er_list_inv')  #Get query entities and relationships pandas
#     dst_get_inv = dst_df_inv['dst'].values    #Get the target tail entity
#     two_ent_inv = set().union(*dst_get_inv)   #Integrate the head entity with the tail entity
#     two_ent_inv = set(random.sample(list(two_ent_inv), k=round(len(two_ent_inv) * 0.3)))
#     all_ent_inv = list(dst_set|two_ent_inv)  
#     result_inv = subg_df.query('src in @all_ent_inv')

#     q_tri = result.to_numpy()
#     q_tri_inv = result_inv.to_numpy()

#     return  q_tri,q_tri_inv




def update_dict(subg_arr, s_to_sro, sr_to_sro,num_rels):
    # 根据输入的每一个时间的图来更新查询查询 - Update the query based on the graph at each time input
    inverse_subg = subg_arr[:, [2, 1, 0]]
    inverse_subg[:, 1] = inverse_subg[:, 1] + num_rels
    subg_triples = np.concatenate([subg_arr, inverse_subg])
    for j, (src, rel, dst) in enumerate(subg_triples):
        s_to_sro[src].add((src, rel, dst))
        sr_to_sro[(src, rel)].add(dst)

def split_by_time(data):
    snapshot_list = []
    snapshot = []
    snapshots_num = 0
    latest_t = 0
    for i in range(len(data)):
        t = data[i][3]
        train = data[i]
        # latest_t表示读取的上一个三元组发生的时刻，要求数据集中的三元组是按照时间发生顺序排序的 
        #latest_t Indicates the time when the last triplet read occurred. The triples in the data set must be sorted in chronological order.
        if latest_t != t:  # 同一时刻发生的三元组 - Triples that occur at the same time
            # show snapshot
            latest_t = t
            if len(snapshot):
                snapshot_list.append(np.array(snapshot).copy())
                snapshots_num += 1
            snapshot = []
        snapshot.append(train[:3])
    # 加入最后一个shapshot - Add the last snapshot
    if len(snapshot) > 0:
        snapshot_list.append(np.array(snapshot).copy())
        snapshots_num += 1

    union_num = [1]
    nodes = []
    rels = []
    for snapshot in snapshot_list:
        uniq_v, edges = np.unique((snapshot[:,0], snapshot[:,2]), return_inverse=True)  # relabel
        uniq_r = np.unique(snapshot[:,1])
        edges = np.reshape(edges, (2, -1))
        nodes.append(len(uniq_v))
        rels.append(len(uniq_r)*2)
    print("# Sanity Check:  ave node num : {:04f}, ave rel num : {:04f}, snapshots num: {:04d}, max edges num: {:04d}, min edges num: {:04d}, max union rate: {:.4f}, min union rate: {:.4f}"
          .format(np.average(np.array(nodes)), np.average(np.array(rels)), len(snapshot_list), max([len(_) for _ in snapshot_list]), min([len(_) for _ in snapshot_list]), max(union_num), min(union_num)))
    return snapshot_list

def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r', encoding='utf-8') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r', encoding='utf-8') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r', encoding='utf-8') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)

# dataset_list = ["ICEWS14", "ICEWS18","ICEWS05-15"]
dataset_list = ["ICEWS18"]
for dataset in dataset_list:
    train_data, train_times = load_quadruples('data/{}'.format(dataset), 'train.txt')
    num_nodes,num_rels= get_total_number('data/{}'.format(dataset), 'stat.txt')
    print("the number of entity and relation", num_nodes,num_rels)

    train_list = split_by_time(train_data)
    id_list = [_ for _ in range(len(train_list))]
    # sample_len = 3

    # save_dir_subg = '../{}/his_graph_for/'.format(dataset)
    # save_dir_obj = '../{}/his_graph_inv/'.format(dataset)
    # save_dir_sub = '../{}/his_dict/'.format(dataset)
    # save_dir_subg = 'data/{}/his_graph_for/'.format(dataset)
    # save_dir_obj = 'data/{}/his_graph_inv/'.format(dataset)
    # save_dir_sub = 'data/{}/his_dict/'.format(dataset)
    save_dir_subg = 'data/{}/his_graph_for_new/'.format(dataset)
    save_dir_obj = 'data/{}/his_graph_inv_new/'.format(dataset)
    save_dir_sub = 'data/{}/his_dict_new/'.format(dataset)

    def mkdirs(path):
        if not os.path.exists(path):
            os.makedirs(path)

    mkdirs(save_dir_obj)
    mkdirs(save_dir_sub)
    mkdirs(save_dir_subg)

    # f2 = open('./data/{}/copy_seq_graph/train_h_r_copy_seq.pkl'.format(args.dataset), 'rb')
    # que_subg = pickle.load(f2)
    sr_to_sro = defaultdict(set)
    s_to_sro = defaultdict(set)
    sro_to_fre = dict()
    subgraph_arr = []
    subgraph_arr_inv = []
    print("------------{}sample history graph-------------------------------------".format(dataset))
    all_list= train_list
    idx = [_ for _ in range(len(all_list))]
    snaplen = []
    for train_sample_num in idx:
    # for train_sample_num in tqdm(idx):
        if train_sample_num == 0: continue
        output = all_list[train_sample_num:train_sample_num+1]
        history_graph = all_list[train_sample_num-1:train_sample_num]
        update_dict(history_graph[0], s_to_sro, sr_to_sro, num_rels)
        if train_sample_num > 0:
            his_list = all_list[:train_sample_num]
            subg_arr = np.concatenate(his_list)
            sub_snap,sub_snap_inv = get_sample_from_history_graph(subg_arr,s_to_sro, sr_to_sro,sro_to_fre, output[0], num_nodes,num_rels)
            print('length of sub snap is', len(sub_snap))
            snaplen.append(len(sub_snap))
            print(train_sample_num)
        #.npy files are binary files to store numpy arrays.
        # np.save('../{}/his_graph_for/train_s_r_{}.npy'.format(dataset, train_sample_num), sub_snap)
        # np.save('../{}/his_graph_inv/train_o_r_{}.npy'.format(dataset, train_sample_num), sub_snap_inv)
        # np.save('data/{}/his_graph_for/train_s_r_{}.npy'.format(dataset, train_sample_num), sub_snap)
        # np.save('data/{}/his_graph_inv/train_o_r_{}.npy'.format(dataset, train_sample_num), sub_snap_inv)
        np.save('data/{}/his_graph_for_new/train_s_r_{}.npy'.format(dataset, train_sample_num), sub_snap)
        np.save('data/{}/his_graph_inv_new/train_o_r_{}.npy'.format(dataset, train_sample_num), sub_snap_inv)
    # np.save('../{}/his_dict/train_s_r.npy'.format(dataset), sr_to_sro)
    # np.save('data/{}/his_dict/train_s_r.npy'.format(dataset), sr_to_sro)
    np.save('data/{}/his_dict_new/train_s_r.npy'.format(dataset), sr_to_sro)
    print('sum of snap lengths ',sum(snaplen))
    # arr = np.load('./{}/his_graph_for/train_s_r_{}.npy'.format(dataset, train_sample_num))
    # print(arr)
    # print(len(sr_to_sro.keys()))
    # sr_dic = np.load('./{}/his_dict/train_s_r.npy'.format(dataset), allow_pickle=True).item()
    # print(len(sr_dic.keys()))

    

# t1 = time.time()
# que_subg_list = defaultdict(list)
# que_subg_len = defaultdict(set)
# for id in tqdm(id_list):
#     triple = train_list[id:id+1]
#     sample_seq_graph = train_list[max(0, id-sample_len):min(id+sample_len, len(id_list))]
#     his_arr = np.concatenate(sample_seq_graph)
#     # que_subg = his_graph_sample1(his_arr, triple[0], num_r,que_subg_list, que_subg_len)
#     que_subg = get_sample_from_history_graph3(his_arr, triple[0], num_rels,que_subg_list, que_subg_len)
#     # with open('./data/{}/copy_seq_graph/train_h_r_copy_seq_{}.pkl'.format(args.dataset, id), 'wb') as f:
#     #     pickle.dump(que_subg, f)
# with open('./{}/copy_seq_graph/train_h_r_copy_seq.pkl'.format(dataset), 'wb') as f1:
#     pickle.dump(que_subg, f1)
# t2 = time.time() 