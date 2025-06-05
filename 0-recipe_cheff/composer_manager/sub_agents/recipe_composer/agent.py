from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.tools.tool_context import ToolContext

def save_recipe(recipe: str, tool_context: ToolContext) -> dict:
    """Save the recipe to the the session state.

    Args:
        recipe: The recipe to save
        tool_context: Context for accessing and updating session state

    Returns:
        A confirmation message
    """
    print(f"Saving recipe")
    tool_context.state["recipe"] = recipe
    return {"action": "save_recipe", "recipe": recipe}


recipe_composer_agent = LlmAgent(
    name="recipe_composer",
    model="gemini-2.0-flash",
    description="Construct well-structured recipes based on the user's input",
    instruction="""
    You are a helpful cheff that crafts recipes based on the user's input, while following the following guidelines:
    - The recipe should be well-structured and easy to follow
    - The recipe should be easy to prepare
    - The recipe should be healthy and nutritious
    - The recipe should be have a high enought calorie density to keep the user full for a long time

    You can find a refference of suggested ideas here:
    {search_response}

    You have following tools at your disposal:
    - save_recipe: To save the recipe to the session state

    Output the recipe in a well-structured format, including:
    - Ingredients
    - Preparation steps
    - Cooking instructions
    - Serving suggestions
    - Nutritional information

    The recipe should be in the following format:
    - Title
    - Ingredients
    - Preparation steps
    - Cooking instructions
    - Serving suggestions
    - Nutritional information

    ----Important----
    Before responding, make sure to save the recipe to the session state using the save_recipe tool.
    """,
    tools=[save_recipe],
    output_key="recipe_response",
)

