from langchain_openai import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import pandas as pd

class PandasChat:
    def chat(self, user_question):
        df = pd.read_excel(
            "post_criteria_evaluation.xlsx"
        )
        try:
            agent = create_pandas_dataframe_agent(
                ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0125"),
                df,
                verbose=True,
                agent_type=AgentType.OPENAI_FUNCTIONS,
            )
            result = agent.invoke(user_question)

            return result['output']
        except Exception as e:
            print(f"Error with Pandas DataFrame agent: {e}")
            return self.default_chat(user_question)
        
    def default_chat(self, user_question):
        df_dict = df.to_dict()
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        messages = [
            ("system", "You are a helpful assistant that helps to answer questions about a list of evaluation results from the candidate's resume. You will be provided context about the evaluation results. "),
            ("system", f"[Evaluation Results]{df_dict}[End of Evaluation Results]"),
            ("human", user_question),
        ]
        llm.invoke(messages)
        
        return messages['output']
