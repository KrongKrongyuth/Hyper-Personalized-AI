from dataset import KaggleDatasetLoader

from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from pydantic import BaseModel, Field
from dotenv import load_dotenv; load_dotenv()

import os

MODEL_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
MAIN_DF = KaggleDatasetLoader().load_dataset()

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
    This object use to structure the output format of features name.
    """
    
    onehot_features: list = Field(description="List of categorical feature names that are suitable to encode by using one-hot encoding in string format.")
    ordinal_features: list = Field(description="List of categorical feature names that are suitable to encode by using ordinal encoding in string format.")
    label_features: list = Field(description="List of categorical feature names that are suitable to encode by using lable encoding in string format.")
    numerical_features: list = Field(description="List of numerical features name in string format.")
    features: list = Field(description="All used feature names are in string format, excluding primary keys such as msisdn.")

class PandasAgent():
    def __init__(
        self,
        model_name:str = MODEL_NAME,
        verbose:bool = True
        ):
        
        self.MAIN_DF = MAIN_DF
        self.MODEL_NAME = model_name
        self.MODEL = define_model()
        self.parser = JsonOutputParser(pydantic_object=FeatureOutput)
        self.VERBOSE = verbose

    def call_agent(
        self,
        query:str
        ):
        prompt = PromptTemplate(
            template="""
            The query you will receive is the business objective from the company;
            you need to identify the features that should be use for clustered into the correct customer target.
            
            *** You must give the answer based on the "output format instructions" only. ***
            
            Business objective: {query}
            Output format instructions: {output_format}
            """,
            input_variables=["query"],
            partial_variables={"output_format": self.parser.get_format_instructions()}
        )
        
        agent = create_pandas_dataframe_agent(
            llm = self.MODEL,
            df = self.MAIN_DF,
            agent_type = AgentType.OPENAI_FUNCTIONS,
            allow_dangerous_code = True,
            verbose = self.VERBOSE
        )
        
        chain = prompt | agent | (lambda x: x['output']) | self.parser
        agent_response = chain.invoke(query)
        
        return agent_response