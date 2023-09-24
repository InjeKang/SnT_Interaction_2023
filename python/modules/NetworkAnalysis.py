from modules.GlobalVariables import *

import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import trange


class MergeData:
    def __init__(self):
        self = self
    
    def doc_by_topics(self, run, no_topic):
        if run:
            data = LoadData().read_data("03.stm_data.pkl")
            data_by_topic = pd.read_excel(f"stm\\topics_{no_topic}\\3.doc_by_topics{no_topic}.xlsx")
            data_by_topic["top5_topics"] = data_by_topic.swifter.apply(lambda x: self._get_topi_five_topics(x, no_topic), axis=1)
            merged_data = data.merge(data_by_topic, on=["id"], how="right")
            merged_data = merged_data[["id", "year", "country", "top5_topics"]]
            merged_data.to_excel(f"stm\\topics_{no_topic}\\3.doc_by_topics{no_topic}_v2(merged).xlsx")
            return merged_data


    def _get_topi_five_topics(self, x, no_topic):
        # Exclude the 'ID' column and sort the remaining columns by values in descending order
        sorted_topics = x[2:no_topic+1].sort_values(ascending=False)
        # Get the top 5 topics
        top_five_topics = sorted_topics.index[:5]
        # Join the top five topics with ';'
        return ';'.join(top_five_topics)        

class NetworkAnalysis:
    def __init__(self):
        self = self
    
    def knowledge_network(self, run, no_topic):
        if run:
            data = pd.read_excel(f"stm\\topics_{no_topic}\\3.doc_by_topics{no_topic}_v2(merged).xlsx")
            # subset_data = data[data["year"]>=2011]
            period_list_ = FilterData().period_list
            region_list_ = FilterData().region_list
            # period_Edgelist = []
            # period_AdjMat = []
            period_Centrality = []
            period_adj = []
            for region in region_list_:
                subset_byRegion = data[data["country"].str.contains(region)]
                for period in period_list_:
                    subset_byPeriod = subset_byRegion[(subset_byRegion["year"]>=period[0]) & (subset_byRegion["year"]<period[1])]
                    # analyze the network structure by consturcting an edge list and an adjacency matrix
                    df_edgeList, node_freq, df_adjacency_matrix = NetworkStructure()._adjacency_matrix(subset_byPeriod, "top5_topics", ";")
                    centrality_byPeriod= NetworkStructure()._measure_centrality(subset_byPeriod, "top5_topics",  ";")
                    # visualize the network
                    name_ = region + "_" + str(period) + "_" + "topic" + str(no_topic)
                    NetworkStructure()._visualize_network(df_adjacency_matrix, no_topic, name_)

                    # period_Edgelist.append(df_edgeList)
                    period_adj.append(df_adjacency_matrix)
                    period_Centrality.append(centrality_byPeriod)

                with  pd.ExcelWriter(f"stm\\topics_{no_topic}\\7.KnowledgeNetwork_AdjMat_{region}.xlsx") as writer:
                    for i in trange(len(FilterData().period_list)):
                        period_adj[i].to_excel(writer, sheet_name = str(period_list_[i]), index=False)  

                with  pd.ExcelWriter(f"stm\\topics_{no_topic}\\7.KnowledgeNetwork_Centrality_{region}.xlsx") as writer:
                    for i in trange(len(FilterData().period_list)):
                        period_Centrality[i].to_excel(writer, sheet_name = str(period_list_[i]), index=False)  



class NetworkStructure():
    def __init__(self):
        self = self

    def _adjacency_matrix(self, data, column_, column_split):
        # convert into an edge list
        df_edgeList = self._edge_list(data, column_, column_split)      
        # an edge list with frequency
        column_freq, nodes_df = self._edgeList_to_dataframe(df_edgeList, column_)
        # identifying network structure
        node_freq = [tuple(row) for row in column_freq.to_records(index=False)]
        # Get the unique values
        nodes = set(pd.concat([column_freq["column1"], column_freq["column2"]]))
        nodes = {x for x in nodes if x != ""}
        # Create an empty adjacency matrix with the same length as the node list
        to_adjacency_matrix = pd.DataFrame(0, index=nodes, columns=nodes)
        # Iterate over the rows of the dataframe and update the adjacency matrix
        for row in column_freq.itertuples(index=False):
            col1, col2, freq = row
            to_adjacency_matrix.loc[col1, col2] += freq
            to_adjacency_matrix.loc[col2, col1] += freq
        return column_freq, node_freq, to_adjacency_matrix

    def _edge_list(self, data, column_, column_split):        
        data2 = data.copy()
        data2.dropna(subset=[column_], inplace=True)
        data2[column_] = data2[column_].str.split(column_split)
        co_words_list = []
        for co_word in data2[column_]:
            # get all combinations of co-words (pairs)  and reflect undirected network
            co_word_combinations = [tuple(sorted(pair)) for pair in itertools.combinations(co_word, 2)]
            co_words_list.append(co_word_combinations)
        co_words_list = list(itertools.chain.from_iterable(co_words_list))
        co_words_df = pd.DataFrame(co_words_list, columns=["column1", "column2"])
        return co_words_df      

    def _edgeList_to_dataframe(self, df_edgeList, column_):
        # Measure network characteristics for an undirected network
        column_freq = df_edgeList.groupby(["column1", "column2"]).size().reset_index(name="freq")
        # Create dataframe for edges and nodes
        edges = column_freq.copy()
        # number of observations in the edge list
        nodes_df = pd.concat([edges["column1"], edges["column2"]]).value_counts().reset_index()
        nodes_df.columns = [column_, "freq"]
        return column_freq, nodes_df

    def _measure_centrality(self, data, column_, delimiter):
        # region_ = overall or by region // collab_ = collab or sole
        df_edgeList = self._edge_list(data, column_, delimiter)
        # convert edgelist to a dataframe
        column_freq, nodes_df = self._edgeList_to_dataframe(df_edgeList, column_)
        # identifying network structure
        node_freq = [tuple(row) for row in column_freq.to_records(index=False)]
        G = nx.Graph()
        G.add_weighted_edges_from(list(node_freq))    
        # measuring three types of centrality        
        degree_centrality_ = nx.degree_centrality(G)
        closeness_centrality_ = nx.closeness_centrality(G)
        betweenness_centrality_ = nx.betweenness_centrality(G)
        # constraints_ = nx.constraint(G)            
        # clustering_coefficient_ = nx.clustering(G)
        # convert the results into a dataframe
        toDF = pd.DataFrame({
            "degree": pd.Series(degree_centrality_),
            "closeness": pd.Series(closeness_centrality_),
            "betweenness": pd.Series(betweenness_centrality_),
            # "constraints": pd.Series(constraints_),
            # "clustering" : pd.Series(clustering_coefficient_)
        })
        # from index to a column
        toDF.reset_index(inplace=True)
        toDF = toDF.rename(columns = {"index" : column_})
        # toDF = toDF.assign(structural_holes = 2 - toDF["constraints"])
        # merge two dataframes - toDF and nodes_df
        mergedDF = pd.merge(toDF, nodes_df, on=column_, how = "inner")
        mergedDF = mergedDF[[column_, "freq", "degree", "closeness", "betweenness"]] # , "clustering", "constraints" , "structural_holes"
        mergedDF = mergedDF.apply(lambda x: x.round(3) if x.name in ["degree", "closeness", "betweenness"] else x) # round the values of the three columns only
        mergedDF = mergedDF.sort_values(by="freq", ascending=False)
        return mergedDF    

    def _visualize_network(self, adjacency_matrix_, no_topic, name_):
        G = nx.from_pandas_adjacency(adjacency_matrix_)
        # Calculate node sizes based on node frequency
        node_sizes = [100 * G.degree[node] for node in G.nodes()]
        # Calculate edge widths based on edge strength (you can replace this with your edge metric)
        edge_widths = [weight * 1/20 for _, _, weight in G.edges(data='weight')]
        # Draw the network graph
        pos = nx.spring_layout(G)  # Position the nodes using a spring layout algorithm
        nx.draw_networkx(G, pos, with_labels=True, node_size=node_sizes, node_shape="o", width=edge_widths) # , font_family=fontprop.get_name()
        # Customize the plot appearance
        plt.title(f"Knowledge Network")
        plt.axis('off')    
        # plt.show()
        plt.savefig(f"stm\\topics_{no_topic}\\7.KnowledgeNetwork{name_}.png")
        plt.clf()   
