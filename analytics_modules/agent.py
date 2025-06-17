from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from pydantic import BaseModel, Field
from dotenv import load_dotenv; load_dotenv()

import pandas as pd
import os

MODEL_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')

def define_model(
    model_name:str = MODEL_NAME,
    temp: float = 0
    ):
    model = AzureChatOpenAI(
                model=model_name,
                temperature=temp
                )
        
    return model

class FeatureOutput(BaseModel):
    """
    Structured object for defining selected features and their encoding suitability for clustering tasks.
    """

    onehot_features: list = Field(
        description="List of categorical feature names suitable for one-hot encoding, in string format."
    )
    ordinal_features: list = Field(
        description="List of categorical feature names suitable for ordinal encoding, in string format."
    )
    numerical_features: list = Field(
        description="List of numerical feature names, in string format."
    )
    features: list = Field(
        description="Combined list of all selected feature names (excluding identifiers or primary keys), in string format."
    )

class PandasAgent():
    def __init__(
        self,
        model_name:str = MODEL_NAME,
        verbose:bool = True,
        ):
        
        self.MODEL_NAME = model_name
        self.MODEL = define_model()
        self.parser = JsonOutputParser(pydantic_object=FeatureOutput)
        self.VERBOSE = verbose
        
        self.agent_prompt = PromptTemplate(
            template="""
            The query you will receive describes an objective or goal related to clustering data.
            Your task is to identify and categorize the relevant features that should be used for the clustering process based on this objective.
            
            *** You must generate your response strictly following the structure defined in the "output format instructions" below. ***
            
            Objective: {query}
            Output format instructions: {output_format}
            """,
            input_variables=["query"],
            partial_variables={"output_format": self.parser.get_format_instructions()}
            )

    def call_agent(
        self,
        query:str,
        dataframe:pd.DataFrame,
        prompt:PromptTemplate = None
        ):
        if not prompt:
            prompt = self.agent_prompt
        
            agent = create_pandas_dataframe_agent(
                llm = self.MODEL,
                df = dataframe,
                agent_type = AgentType.OPENAI_FUNCTIONS,
                allow_dangerous_code = True,
                verbose = self.VERBOSE
            )
            
            chain = prompt | agent | (lambda x: x['output']) | self.parser
        else:
            agent = create_pandas_dataframe_agent(
                    llm = self.MODEL,
                    df = dataframe,
                    agent_type = AgentType.OPENAI_FUNCTIONS,
                    allow_dangerous_code = True,
                    verbose = self.VERBOSE
                )
            chain = prompt | agent | (lambda x: x['output'])
        agent_response = chain.invoke(query)
        
        return agent_response