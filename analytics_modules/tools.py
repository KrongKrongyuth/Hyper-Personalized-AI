from dataset import KaggleDatasetLoader
from agent import PandasAgent, define_model

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder

from pydantic import BaseModel, Field
from tqdm import tqdm
from dotenv import load_dotenv; load_dotenv()

import matplotlib.pyplot as plt
import os

MODEL_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
MAIN_DF = KaggleDatasetLoader().load_dataset()

class ClusteringOutput(BaseModel):
    
    """
    This object use to structure the output format after apply Clustering.
    """
    
    K: int = Field(description="Optimal value of K for clustering")

class ClusteringCustomer():
    def __init__(
        self,
        model_name:str = MODEL_NAME
        ):
        
        self.agent = PandasAgent(model_name)
        self.MAIN_DF = self.agent.MAIN_DF
        self.MODEL = define_model()
        self.parser = JsonOutputParser(pydantic_object=ClusteringOutput)
        
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
    
    def apply_KMeans(
        self,
        business_obj:str,
        display_graph:bool = False,
        number_of_cluster:int = 10
        ):
        featrues = self.agent.call_agent(query=business_obj)
        wcss = []
        silhouette_scores = []
        
        for k in tqdm(range(2,number_of_cluster+1)):
            onehot_trans = ColumnTransformer(
                [
                    ("onehot_encoding", OneHotEncoder(), featrues['onehot_features'])
                ],
                remainder="passthrough"
            )
            ordinal_trans = ColumnTransformer(
                [
                    ("ordinal_encoding", OrdinalEncoder(), featrues['ordinal_features'])
                ],
                remainder="passthrough"
            )
            label_trans = ColumnTransformer(
                [
                    ("label_encoding", LabelEncoder(), featrues['label_features'])
                ],
                remainder="passthrough"
            )
            cluster_pipeline = Pipeline(
                [
                    ("K-Means", KMeans(
                        n_clusters=k,
                        init="k-means++",
                        n_init='auto',
                        # max_iter=500,
                        random_state=42,
                        )
                    )
                ]
            )
            encode_workflow = Pipeline(
                [
                    ("Onehot", onehot_trans),
                    ("Ordinal", ordinal_trans),
                    ("Label_pipeline", label_trans)
                ]
            )
            cluster_workflow = Pipeline(
                [
                    ("Cluster", cluster_pipeline)
                ]
            )
            
            data = encode_workflow.fit_transform(self.MAIN_DF.loc[:1000, featrues['features']].copy())
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
        raw_data = encode_workflow.fit_transform(self.MAIN_DF.loc[:1000, featrues['features']].copy())
        k_means = KMeans(
                        n_clusters=cluster_setting['K'],
                        init="k-means++",
                        n_init=50,
                        max_iter=500,
                        random_state=42,
                        )
        k_means.fit(raw_data)
        labels = k_means.labels_
        center = k_means.cluster_centers_
        
        return labels, center

if __name__ == "__main__":
    agent = ClusteringCustomer()
    
    while True:
        user_query = input("\nYour query: ")
        
        if user_query == "/goodbye":
            print("\n********** Goodbye! **********\n")
            break
        
        print("\n********** Agent response **********")
        print(agent.apply_KMeans(business_obj=user_query))