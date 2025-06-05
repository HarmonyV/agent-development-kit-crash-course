from .sub_agents.recipe_composer.agent import recipe_composer_agent
from .sub_agents.brainstorm_agent.agent import brainstorm_agent
from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

root_agent = LlmAgent(
    model='gemini-2.0-flash-001',
    name='composer_manager',
    description='A manager that orchestrates and manages the recipe creation process',
    instruction='''
    You are a manager that orchestrates and manages the recipe creation process.
  

    You have the following tools at your disposal:
    - brainstorm_agent: To search the web for information and brainstorm ideas.

    You have the following sub-agents at your disposal:
    - recipe_composer_agent: To compose a recipe based on the user's input.


   if the user wants to brainstorm ideas or wants suggestions, use the brainstorm_agent tool.
   if the user wants to create a recipe, use the recipe_composer_agent tool and skip the brainstorm_agent tool.
    ''',
    tools=[AgentTool(brainstorm_agent)],
    sub_agents=[recipe_composer_agent],
   
)
