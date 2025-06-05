from google.adk.agents import LlmAgent
from google.adk.tools import google_search

brainstorm_agent = LlmAgent(
    model='gemini-2.0-flash-001',
    name='brainstorm_agent',
    description='A brainstorming agent that helps the user brainstorm ideas for a recipe',
    instruction='''
    You are a brainstorming agent that helps the user brainstorm ideas for a recipe.
    Based on the user's input, you will need to help the user brainstorm ideas for a recipe.
    You have the following tools at your disposal:
    - google_search: To search the web for ideas and inspiration for a recipe.
    
    Keep your response concise and to the point.
    ''',
    tools=[google_search],
    output_key="search_response",
)
