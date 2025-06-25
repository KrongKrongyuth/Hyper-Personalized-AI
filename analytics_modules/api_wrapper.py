from fastapi import FastAPI

from analytics_modules.tools import ClusteringCustomer

api = FastAPI()
agent = ClusteringCustomer()

@api.get('/response')
def get_cluster_response(
    query:str
    ):
    return agent.apply_KMeans(query)