import re
from typing import List, Union, Tuple
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from langchain_groq import ChatGroq

class CustomAgent:
    def __init__(self):
        self.llm = ChatGroq(
            temperature=0.7,
            model_name="llama-3.2-3b-preview",
            max_tokens=4096
        )
    
    def should_use_tool(self, query: str) -> bool:
        scheduling_keywords = ['schedule', 'appointment', 'book', 'call me', 'contact']
        return any(keyword in query.lower() for keyword in scheduling_keywords)

    def parse_output(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
            
        action_match = re.search(r"Action: (.*?)[\n]*Action Input: (.*)", llm_output, re.DOTALL)
        if not action_match:
            return AgentFinish(
                return_values={"output": llm_output},
                log=llm_output,
            )
            
        action = action_match.group(1).strip()
        action_input = action_match.group(2).strip()
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)