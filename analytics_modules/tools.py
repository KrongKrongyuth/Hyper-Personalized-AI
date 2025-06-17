from agent import PandasAgent, define_model
from dataset import KaggleDatasetLoader

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from pydantic import BaseModel, Field
from scipy.sparse import vstack
from tqdm import tqdm
from dotenv import load_dotenv; load_dotenv()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
import os

MODEL_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

def load_MAIN_DF():
    data_dict = KaggleDatasetLoader().load_dataset()
    crm_df = data_dict['crm1.csv']
    device_df = data_dict['device1.csv']
    rev_df = data_dict['rev1.csv']
    MAIN_DF = pd.merge(pd.merge(crm_df, device_df, on='msisdn'),rev_df,on='msisdn').dropna()
    
    return MAIN_DF.loc[:50000, :]

class ClusteringOutput(BaseModel):
    
    """
    This object use to structure the output format after apply Clustering.
    """
    
    K: int = Field(description="Optimal value of K for clustering")

class ClusteringCustomer():
    def __init__(
        self,
        dataframe:pd.DataFrame = None,
        model_name:str = MODEL_NAME,
        ):
        
        self.agent = PandasAgent(model_name=model_name)
        self.MODEL = define_model()
        self.parser = JsonOutputParser(pydantic_object=ClusteringOutput)
        
        if dataframe is not None:
            self.MAIN_DF = dataframe
        else:
            self.MAIN_DF = load_MAIN_DF()
        
        self.cluster_prompt = PromptTemplate(
            template="""
            I have performed clustering using different numbers of clusters and have computed the Silhouette Scores and Within-Cluster-Sum-of-Squares (WCSS) for each.
            Please help me analyze the results and recommend the most appropriate number of clusters.
            
            Here are the results:
            Number of Clusters (k): {number_of_cluster}
            Silhouette Scores: {silohouette_scores}
            WCSS: {wcss}
            
            Based on this information, please:
            Identify the optimal number of clusters using both metrics.
            Explain the reasoning (e.g. elbow method, peak silhouette, diminishing returns in WCSS).
            Suggest what to do if the metrics disagree.
            
            Output format instruction: {output_format}
            """,
            input_variables=['number_of_cluster', 'silohouette_scores', 'wcss'],
            partial_variables={"output_format": self.parser.get_format_instructions()}
            
            # To provide an image, this prompt should work
            # I also have a plot of WCSS vs k, which you can use this information to help you make the appropriate decision.
            # {graph_img}
        )
        self.naming_prompt = PromptTemplate(
            template="""
            You are provided with a DataFrame named Clustred_dataset containing clustered data.
            The DataFrame has the following structure:
                * A column named "Cluster" indicating the cluster ID each row belongs to.
                * Other columns contain features or attributes that describe each data point.
            Your task is to:
                1.Analyze all clusters in the DataFrame.
                2.For each cluster:
                    * Assign a meaningful and intuitive name based on the common characteristics of the data points in that cluster.
                    * Provide a brief, clear description summarizing the cluster's distinguishing features.
                3.Select the single cluster that is most relevant to the following query:
                "{query}"
                4.Explain why that cluster was selected.
            Use statistical summaries, feature distributions, or other patterns in the DataFrame to guide your decision. 
            Be thoughtful and accurate in interpreting the data.
            
            <Your response language will be based on the language of the query you received.>
            """,
            input_variables=['query', 'clusters']
        )
    
    def apply_KMeans(
        self,
        objective:str,
        history:list = [],
        display_graph:bool = False,
        number_of_cluster:int = 10,
        save_clustred:bool = True
        ):
        featrues = self.agent.call_agent(query=objective, dataframe=self.MAIN_DF)
        wcss = []
        silhouette_scores = []
        
        for k in tqdm(range(2,number_of_cluster+1)):
            preprocessor = ColumnTransformer(
                transformers=[
                    ("onehot_encoding", OneHotEncoder(), featrues['onehot_features']),
                    ("ordinal_encoding", OrdinalEncoder(), featrues["ordinal_features"])
                ],
                remainder="passthrough"
            )
            cluster_pipeline = Pipeline(
                [
                    ("K-Means", KMeans(
                        n_clusters=k,
                        init="k-means++",
                        n_init='auto',
                        random_state=0
                        )
                    )
                ]
            )
            encode_workflow = Pipeline(
                [
                    ("Columns Processer", preprocessor),
                ]
            )
            cluster_workflow = Pipeline(
                [
                    ("Standardize", StandardScaler(with_mean=False)),
                    ("Cluster", cluster_pipeline)
                ]
            )
            data = encode_workflow.fit_transform(self.MAIN_DF[featrues['features']].copy())
            cluster_workflow.fit(data)
            wcss.append(cluster_workflow["Cluster"]["K-Means"].inertia_)
            score = silhouette_score(data,cluster_workflow["Cluster"]["K-Means"].labels_)
            silhouette_scores.append(score)
        
        if display_graph:
            plt.style.use("fivethirtyeight")
            plt.plot(range(2, number_of_cluster+1), wcss)
            plt.xticks(range(2, number_of_cluster+1))
            plt.xlabel("Number of Clusters")
            plt.ylabel("WCSS")
            plt.show()
        
        clustering_chain = self.cluster_prompt | self.MODEL | (lambda x: x.content) | self.parser
        cluster_setting = clustering_chain.invoke(
            {
                'number_of_cluster': range(2,number_of_cluster+1),
                'silohouette_scores': silhouette_scores,
                'wcss': wcss
            }
            )
        
        # Apply KMeans with the optimal K
        raw_data = encode_workflow.fit_transform(self.MAIN_DF[featrues['features']].copy())
        k_means = KMeans(
                        n_clusters=cluster_setting['K'],
                        init="k-means++",
                        n_init='auto',
                        random_state=0
                        )
        try:
            self.MAIN_DF['cluster_features'] = list(raw_data)
            self.MAIN_DF["Cluster"] = k_means.fit_predict(vstack(self.MAIN_DF['cluster_features'].values))
        except:
            self.MAIN_DF['cluster_features'] = list(raw_data)
            self.MAIN_DF["Cluster"] = k_means.fit_predict(np.stack(self.MAIN_DF['cluster_features'].values))
        
        if save_clustred:
            self.MAIN_DF.to_csv('./analytics_modules/Clustered_dataset.csv')
            print("data is saved.")
        
        # Execute agent
        agent_response = self.agent.call_agent(query=objective, dataframe=self.MAIN_DF, prompt=self.naming_prompt)
        
        return agent_response

if __name__ == "__main__":
    dataloader = KaggleDatasetLoader(kaggle_path="youssefaboelwafa/clustering-penguins-species")
    data_dict = dataloader.load_dataset()
    raw_data = data_dict['penguins.csv'].dropna()
    agent = ClusteringCustomer(dataframe=raw_data)
    # agent = ClusteringCustomer()
    
    # On terminal interface
    # while True:
    #     user_query = input("\nYour query: ")
        
    #     if user_query == "/goodbye":
    #         print("\n********** Goodbye! **********\n")
    #         break
        
    #     print("\n********** Agent response **********")
    #     print(agent.apply_KMeans(objective=user_query))
    
    # Apply gradio interface
    app = gr.ChatInterface(
        fn = agent.apply_KMeans,
        type="messages"
    )
    app.launch()