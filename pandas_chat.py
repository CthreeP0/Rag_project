from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd

class PandasChat:
    def chat(self, user_question):
        df = pd.read_excel(
            "post_criteria_evaluation.xlsx"
        )

        agent = create_pandas_dataframe_agent(
            ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"),
            df,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )
        result = agent.invoke(user_question)

        return result
