import asyncio
import os
from typing import Optional
import uuid

from dotenv import load_dotenv
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService, DatabaseSessionService
from supabase_session_service_v3 import SupabaseSessionService
from utils import call_agent_async
from composer_manager.agent import root_agent

import warnings
warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level=logging.ERROR)

# Gemini API Key (Get from Google AI Studio: https://aistudio.google.com/app/apikey)
os.environ["GOOGLE_API_KEY"] = "my_key" 
os.environ['SUPABASE_URL'] = "my_url"
os.environ['SUPABASE_KEY'] = 'my_key'

print(f"Google API Key set: {'Yes' if os.environ.get('GOOGLE_API_KEY') and os.environ['GOOGLE_API_KEY'] != 'YOUR_GOOGLE_API_KEY' else 'No (REPLACE PLACEHOLDER!)'}")

session_service = SupabaseSessionService(supabase_url=os.environ['SUPABASE_URL'], supabase_key=os.environ['SUPABASE_KEY'])

APP_NAME = "Recipe Composer"
SESSION_ID = str(uuid.uuid4())
USER_ID = "harmony"
INITIAL_STATE = {
    "recipe": "Unknown",
}



async def main_async(session_id: str = SESSION_ID, user_id: str = USER_ID):

    maybe_session = session_service.get_session(
        app_name=APP_NAME,
        user_id=user_id,
        session_id=session_id,
    )

    if maybe_session:
       print(f"Continuing existing session: {session_id}")
    else:
        session_service.create_session(
            session_id=session_id,
            app_name=APP_NAME,
            user_id=user_id,
            state=INITIAL_STATE,
        )
 
    runner = Runner(
        agent=root_agent, # The agent we want to run
        app_name=APP_NAME,   # Associates runs with our app
        session_service=session_service # Uses our session manager
    )

    await call_agent_async(runner, user_id, session_id, "give me some filling pork recipe suggestions",)
    await call_agent_async(runner, user_id, session_id, "please create a recipe based on the first suggestion",)

if __name__ == "__main__":
    asyncio.run(main_async())

