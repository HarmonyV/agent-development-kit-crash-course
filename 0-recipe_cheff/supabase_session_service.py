import logging
import os
from datetime import datetime
import uuid
from typing import Any, Dict, List, Optional, Union

from supabase import Client, create_client
from supabase.lib.client_options import ClientOptions
from typing_extensions import override
from dotenv import load_dotenv

from google.adk.events import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.base_session_service import BaseSessionService, GetSessionConfig, ListEventsResponse, ListSessionsResponse
from google.adk.sessions.session import Session

logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class SupabaseSessionService(BaseSessionService):
    """Connects to a Supabase backend for session and event management."""

    SESSIONS_TABLE = "sessions"
    EVENTS_TABLE = "events"

    def __init__(
        self,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        schema: str = "public",
        client_options: Optional[ClientOptions] = None,
    ):
        self.supabase_url = supabase_url or os.environ.get("SUPABASE_URL")
        self.supabase_key = supabase_key or os.environ.get("SUPABASE_KEY")
        self.schema = schema
        self.client_options = client_options or ClientOptions()

        if not self.supabase_url:
            raise ValueError("Supabase URL must be provided or set in SUPABASE_URL environment variable.")
        if not self.supabase_key:
            raise ValueError("Supabase key must be provided or set in SUPABASE_KEY environment variable.")

        try:
            self.client: Client = create_client(
                self.supabase_url,
                self.supabase_key,
                options=self.client_options,
            )
            self.client.schema(self.schema) # type: ignore
            logger.info("Supabase client initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise

    @override
    def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[Dict[str, Any]] = None,
        session_id: Optional[str] = None,
    ) -> Session:
        new_session_id = session_id or str(uuid.uuid4())
        current_timestamp = datetime.now().isoformat()

        session_data = {
            "id": new_session_id,
            "app_name": app_name,
            "user_id": user_id,
            "session_state": state or {},
            "last_update_time": current_timestamp, # Storing as Unix timestamp
            "created_at": current_timestamp, # Supabase typically uses timestamptz
        }

        try:
            response = (
                self.client.table(self.SESSIONS_TABLE)
                .insert(session_data)
                .execute()
            )
            logger.info(f"Create Session response: {response}")

            if not response.data:
                 # Newer supabase-py might return data directly in response, older in response.data
                if hasattr(response, 'data') and not response.data and hasattr(response, 'error') and response.error:
                    raise Exception(f"Failed to create session: {response.error.message}") # type: ignore
                elif not hasattr(response, 'data'): # Check if response itself is the data
                    pass # Assuming response is the data
                else:
                    raise Exception("Failed to create session: No data returned and no error indicated.")

            created_session_data = response.data[0] if hasattr(response, 'data') and response.data else response
            if not created_session_data or not created_session_data.get("id"):
                 # Check if the response is a list and take the first element
                if isinstance(created_session_data, list) and created_session_data:
                    created_session_data = created_session_data[0]
                else:
                    raise Exception(f"Failed to parse created session data from response: {response}")


        except Exception as e:
            logger.error(f"Error creating session in Supabase: {e}")
            # Consider specific error handling or re-raising with context
            raise

        return Session(
            app_name=app_name,
            user_id=user_id,
            id=created_session_data["id"],
            state=created_session_data.get("session_state", {}),
            last_update_time= datetime.fromisoformat(created_session_data["last_update_time"]).timestamp(),
        )

    @override
    def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Session:
        try:
            response = (
                self.client.table(self.SESSIONS_TABLE)
                .select("*, events(*)") # Select session and all its related events
                .eq("id", session_id)
                .eq("app_name", app_name)
                .eq("user_id", user_id)
                .maybe_single() # Returns one record or None, errors if multiple (should not happen with id)
                .execute()
            )
            logger.info(f"Get Session response: {response}")

            if not response.data:
                 # Handle cases where data might not be directly in response.data for older supabase-py or specific errors
                if hasattr(response, 'data') and not response.data:
                    if hasattr(response, 'error') and response.error:
                        # Distinguish between "not found" and other errors
                        if response.error.code == "PGRST116": # "JSON object requested, multiple (or no) rows returned"
                            logger.warning(f"Session not found or multiple returned for id {session_id}, app {app_name}, user {user_id}")
                            raise FileNotFoundError(f"Session with id {session_id} not found for app {app_name} and user {user_id}.")
                        raise Exception(f"Error fetching session: {response.error.message}") # type: ignore
                    else:
                        # No data and no error might mean not found if maybe_single() was used
                        logger.warning(f"Session not found for id {session_id}, app {app_name}, user {user_id}")
                        raise FileNotFoundError(f"Session with id {session_id} not found for app {app_name} and user {user_id}.")
                elif not hasattr(response, 'data'): # if response *is* the data and it's None/empty
                    logger.warning(f"Session not found (direct response check) for id {session_id}, app {app_name}, user {user_id}")
                    raise FileNotFoundError(f"Session with id {session_id} not found for app {app_name} and user {user_id}.")

            session_data = response.data
            if not isinstance(session_data, dict):
                # If maybe_single() somehow returns a list with one item
                if isinstance(session_data, list) and len(session_data) == 1:
                    session_data = session_data[0]
                else:
                    logger.error(f"Unexpected session data format: {session_data}")
                    raise ValueError(f"Unexpected data format for session {session_id}.")

        except FileNotFoundError:
            raise # Re-raise FileNotFoundError to be handled by caller
        except Exception as e:
            logger.error(f"Error fetching session {session_id} from Supabase: {e}")
            raise

        # Extract events from session_data if a relationship 'events' was defined and fetched
        # Supabase returns related records as a list under the relationship name.
        # If no foreign key relationship named 'events' is explicitly set up in Supabase pointing from 'events' to 'sessions',
        # then we need to fetch events separately.
        # For now, let's assume the "events(*)" syntax works due to a foreign key `session_id` in `events` table.

        raw_events = session_data.pop("events", []) # Pop events so they are not in session_state accidentally
        last_update_time_str = session_data["last_update_time"]

        session_obj = Session(
            id=session_data["id"],
            app_name=session_data["app_name"],
            user_id=session_data["user_id"],
            state=session_data.get("session_state", {}),
            last_update_time=datetime.fromisoformat(last_update_time_str).timestamp(),
            events=[] # Initialize with empty list, will be populated below
        )

        if raw_events:
            session_obj.events = [_from_supabase_event_dict(event_dict) for event_dict in raw_events]
            # Filter events to be at or before the session's last_update_time (as per original logic)
            session_obj.events = [
                event for event in session_obj.events if event.timestamp <= session_obj.last_update_time
            ]
            session_obj.events.sort(key=lambda event: event.timestamp)
        else:
            # If events were not fetched with session (e.g. no FK relation or different strategy)
            # We would fetch them here using list_events logic, then assign.
            # For now, relying on "events(*)" from Supabase.
            logger.info(f"No events found directly with session {session_id}. Can be normal or indicate missing FK relation for joined query.")

        if config:
            if config.num_recent_events is not None and config.num_recent_events > 0:
                session_obj.events = session_obj.events[-config.num_recent_events:]
            elif config.after_timestamp is not None:
                session_obj.events = [e for e in session_obj.events if e.timestamp > config.after_timestamp]

        return session_obj

    @override
    def list_sessions(
        self, *, app_name: str, user_id: str
    ) -> ListSessionsResponse:
        try:
            response = (
                self.client.table(self.SESSIONS_TABLE)
                .select("id, app_name, user_id, session_state, last_update_time, created_at")
                .eq("app_name", app_name)
                .eq("user_id", user_id)
                .order("last_update_time", desc=True) # Optional: order by most recently updated
                .execute()
            )
            logger.info(f"List Sessions response: {response}")

            if not response.data:
                if hasattr(response, 'data') and not response.data:
                    if hasattr(response, 'error') and response.error:
                        raise Exception(f"Error listing sessions: {response.error.message}") # type: ignore
                    # No error but no data means an empty list, which is valid.
                    return ListSessionsResponse(sessions=[])
                elif not hasattr(response, 'data'): # Response itself is data
                    if not response: # If response (as data) is empty
                        return ListSessionsResponse(sessions=[])
                    # If response is data, it should be a list here
                
            # If we reach here, response.data (or response itself) should be a list of session dicts
            raw_sessions_data = response.data if hasattr(response, 'data') else response
            if not isinstance(raw_sessions_data, list):
                 logger.error(f"Unexpected data format for list_sessions: {raw_sessions_data}")
                 raise ValueError("Unexpected data format when listing sessions.")

        except Exception as e:
            logger.error(f"Error listing sessions for app {app_name}, user {user_id} from Supabase: {e}")
            raise

        sessions_list = []
        for session_data in raw_sessions_data:
            last_update_time_str = session_data["last_update_time"]
            # last_update_time = float(session_data["last_update_time"]) # Assuming stored as float
            
            session_obj = Session(
                id=session_data["id"],
                app_name=session_data["app_name"],
                user_id=session_data["user_id"],
                state=session_data.get("session_state", {}),
                last_update_time=datetime.fromisoformat(last_update_time_str).timestamp(),
                # list_sessions typically does not load events for each session to keep it light
                events=[] 
            )
            sessions_list.append(session_obj)
        
        return ListSessionsResponse(sessions=sessions_list)

    @override
    def delete_session(
        self, *, app_name: str, user_id: str, session_id: str
    ) -> None:
        try:
            # First, ensure the session belongs to the specified app_name and user_id before deleting.
            # This is a safety check, although delete by id should be specific enough.
            # However, Supabase `delete` doesn't typically return the deleted record for easy verification of other fields.
            # So, we either trust `session_id` is unique and proceed, or do a select first (adds overhead).
            # For this implementation, we'll proceed with direct delete by id, app_name, and user_id matching.

            response = (
                self.client.table(self.SESSIONS_TABLE)
                .delete()
                .eq("id", session_id)
                .eq("app_name", app_name) # Ensure it matches app_name
                .eq("user_id", user_id)   # Ensure it matches user_id
                .execute()
            )
            logger.info(f"Delete Session response: {response}")

            # response.data for delete is usually a list of the deleted records.
            # If nothing was deleted (e.g., session_id didn't match filters), data might be empty.
            if hasattr(response, 'data') and not response.data:
                if hasattr(response, 'error') and response.error:
                     # An error occurred during the delete operation
                    raise Exception(f"Error deleting session {session_id}: {response.error.message}") # type: ignore
                else:
                    # No error, but no data returned. This implies the session_id (with app_name/user_id) was not found.
                    logger.warning(f"Session {session_id} not found for app {app_name}, user {user_id} during delete, or already deleted.")
                    # Depending on desired behavior, this could raise a FileNotFoundError or just be a no-op.
                    # For consistency with other services that might error on deleting non-existent items, let's consider this.
                    # However, typical REST DELETE is idempotent, so no error if not found is also common.
                    # The original VertexAI service doesn't specify behavior for deleting non-existent.
                    # Let's assume no error if not found is acceptable.
                    pass 
            elif not hasattr(response, 'data') and response: # Response itself is data
                if not response: # Empty list means not found
                    logger.warning(f"Session {session_id} not found (direct response check) for app {app_name}, user {user_id} during delete, or already deleted.")
            
            # If cascading delete is set up in Supabase for events based on session_id, 
            # events will be deleted automatically. Otherwise, they would need to be deleted manually here.
            # E.g., self.client.table(self.EVENTS_TABLE).delete().eq("session_id", session_id).execute()
            # For now, relying on DB-level cascade.

        except Exception as e:
            logger.error(f"Error deleting session {session_id} from Supabase: {e}")
            raise
        return # Explicitly return None

    @override
    def list_events(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> ListEventsResponse:
        try:
            response = (
                self.client.table(self.EVENTS_TABLE)
                .select("*")
                .eq("session_id", session_id)
                .order("timestamp", desc=False) # Fetch in ascending order of timestamp
                .execute()
            )
            logger.info(f"List Events response for session {session_id}: {response}")

            if not response.data:
                if hasattr(response, 'data') and not response.data:
                    if hasattr(response, 'error') and response.error:
                        raise Exception(f"Error listing events for session {session_id}: {response.error.message}") # type: ignore
                    # No error but no data means an empty list of events, which is valid.
                    return ListEventsResponse(events=[])
                elif not hasattr(response, 'data') and not response: # Response itself is data and is empty
                     return ListEventsResponse(events=[])
            
            raw_events_data = response.data if hasattr(response, 'data') else response
            if not isinstance(raw_events_data, list):
                logger.error(f"Unexpected data format for list_events: {raw_events_data}")
                raise ValueError("Unexpected data format when listing events.")

        except Exception as e:
            logger.error(f"Error listing events for session {session_id} from Supabase: {e}")
            raise

        events_list = [_from_supabase_event_dict(event_dict) for event_dict in raw_events_data]
        
        return ListEventsResponse(events=events_list)

    @override
    def append_event(self, session: Session, event: Event) -> Event:
        # Update the in-memory session first (as in BaseSessionService or original VertexAI impl)
        super().append_event(session=session, event=event)

        if not event.id:
            event.id = str(uuid.uuid4())
        if event.timestamp is None: # Ensure event.timestamp is float if not set
            event.timestamp = datetime.now().timestamp()
        
        event_data = _convert_event_to_supabase_dict(event, session.id)
        # For DB update, session's last_update_time needs to be ISO string
        session_last_update_time_iso = datetime.fromtimestamp(event.timestamp).isoformat()

        try:
            # Insert the event
            response_event = (
                self.client.table(self.EVENTS_TABLE)
                .insert(event_data)
                .execute()
            )
            logger.info(f"Append Event response: {response_event}")

            if not response_event.data:
                if hasattr(response_event, 'data') and not response_event.data and hasattr(response_event, 'error') and response_event.error:
                    raise Exception(f"Failed to append event: {response_event.error.message}") # type: ignore
                elif not hasattr(response_event, 'data'):
                     pass # Allow if response itself is data
                else:
                    raise Exception("Failed to append event: No data returned and no error indicated.")

            # Update the session's last_update_time
            response_session_update = (
                self.client.table(self.SESSIONS_TABLE)
                .update({"last_update_time": session_last_update_time_iso})
                .eq("id", session.id)
                .execute()
            )
            logger.info(f"Update session last_update_time response: {response_session_update}")

            if not response_session_update.data:
                if hasattr(response_session_update, 'data') and not response_session_update.data and hasattr(response_session_update, 'error') and response_session_update.error:
                     # Log warning but don't fail the event append if only session update time fails
                    logger.warning(f"Failed to update session last_update_time for session {session.id}: {response_session_update.error.message}") # type: ignore
                elif not hasattr(response_session_update, 'data'):
                    pass # Allow if response itself is data
                else:
                     logger.warning(f"Failed to update session last_update_time for session {session.id}: No data returned, no error.")
            
            # Update session object in memory with new float timestamp
            session.last_update_time = event.timestamp

        except Exception as e:
            logger.error(f"Error appending event or updating session in Supabase: {e}")
            raise

        return event

def _convert_actions_to_dict(actions: Optional[EventActions]) -> Optional[Dict[str, Any]]:
    if not actions:
        return None
    return {
        "skip_summarization": actions.skip_summarization,
        "state_delta": actions.state_delta,
        "artifact_delta": actions.artifact_delta,
        "transfer_agent": actions.transfer_to_agent,
        "escalate": actions.escalate,
        "requested_auth_configs": actions.requested_auth_configs,
    }

def _dict_to_actions(actions_dict: Optional[Dict[str, Any]]) -> Optional[EventActions]:
    if not actions_dict:
        return None
    return EventActions(
        skip_summarization=actions_dict.get("skip_summarization"),
        state_delta=actions_dict.get("state_delta", {}),
        artifact_delta=actions_dict.get("artifact_delta", {}),
        transfer_to_agent=actions_dict.get("transfer_agent"),
        escalate=actions_dict.get("escalate"),
        requested_auth_configs=actions_dict.get("requested_auth_configs", {}),
    )

def _convert_event_to_supabase_dict(event: Event, session_id: str) -> Dict[str, Any]:
    """Converts an Event object to a dictionary for Supabase, including event_metadata."""
    event_metadata_dict = {
        "partial": event.partial,
        "turn_complete": event.turn_complete,
        "interrupted": event.interrupted,
        "branch": event.branch,
        "long_running_tool_ids": (
            list(event.long_running_tool_ids)
            if event.long_running_tool_ids
            else None
        ),
        "grounding_metadata": (
            event.grounding_metadata.model_dump(exclude_none=True)
            if event.grounding_metadata
            else None
        ),
    }

    # Ensure timestamp is a float (Unix timestamp)
    # Supabase client typically handles datetime objects for timestamptz columns,
    # but if storing as float, ensure it's float.
    # For timestamptz, it's better to convert to ISO 8601 string or let Supabase client handle datetime.
    # Assuming 'timestamp' column in Supabase is timestamptz.
    # Supabase client will convert float epoch to timestamptz string if column is timestamptz.

    event_dict = {
        "id": event.id or str(uuid.uuid4()), # Generate ID if not present
        "session_id": session_id,
        "invocation_id": event.invocation_id,
        "author": event.author,
        "actions": _convert_actions_to_dict(event.actions),
        "content": event.content.model_dump(exclude_none=True) if event.content else None,
        "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(), # Convert float to ISO string for DB
        "error_code": event.error_code,
        "error_message": event.error_message,
        "event_metadata": event_metadata_dict,
    }
    return event_dict

def _from_supabase_event_dict(supabase_event: Dict[str, Any]) -> Event:
    """Converts a dictionary from Supabase to an Event object, including event_metadata."""
    actions = _dict_to_actions(supabase_event.get("actions"))
    
    # Handle content (assuming it's stored as JSON and needs to be parsed if not already a dict)
    # For this example, assuming content is already a dict if present.
    # If content comes from model_dump, it's already a dict.

    # Supabase returns timestamptz as ISO 8601 string e.g., '2024-07-30T10:30:00+00:00'
    # We need to convert it to a float timestamp (Unix epoch).
    # The original VertexAISessionService uses isoparse(api_event['timestamp']).timestamp()
    # Supabase select query returns float if the column is float, or string if timestamptz.
    # Assuming we store as Unix float timestamp directly or convert on read.
    # If it's stored as timestamptz, Supabase returns an ISO string. We'll need to parse it.
  
    
    timestamp_val_iso = supabase_event["timestamp"]
    # Assuming 'timestamp' is stored and retrieved as a Unix float timestamp.
    timestamp = datetime.fromisoformat(timestamp_val_iso).timestamp() # Convert ISO string from DB to float

    event = Event(
        id=supabase_event["id"],
        invocation_id=supabase_event.get("invocation_id"),
        author=supabase_event.get("author"),
        actions=actions,
        content=supabase_event.get("content"), # Assuming content_data is in the correct format or None
        timestamp=timestamp,
        error_code=supabase_event.get("error_code"),
        error_message=supabase_event.get("error_message"),
    )

    metadata_dict = supabase_event.get("event_metadata")
    if metadata_dict:
        event.partial = metadata_dict.get("partial")
        event.turn_complete = metadata_dict.get("turn_complete")
        event.interrupted = metadata_dict.get("interrupted")
        event.branch = metadata_dict.get("branch")
        long_running_tool_ids_list = metadata_dict.get("long_running_tool_ids")
        event.long_running_tool_ids = (
            set(long_running_tool_ids_list)
            if long_running_tool_ids_list
            else None
        )
        event.grounding_metadata = metadata_dict.get("grounding_metadata") 

    return event
