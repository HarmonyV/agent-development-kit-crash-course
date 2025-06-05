import base64
import copy
from datetime import datetime, timezone # Use timezone-aware datetimes
import json
import logging
from typing import Any, Optional, Union # Added Union
import uuid

from google.genai import types # Assuming this is still used for Event content
from supabase import create_client, Client # Supabase client
from postgrest import APIResponse # For type hinting Supabase responses

# Assuming these classes are defined elsewhere in your ADK project
from google.adk.events.event import Event # from your_adk_path.events.event import Event
from google.adk.sessions.base_session_service import BaseSessionService
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.base_session_service import ListEventsResponse
from google.adk.sessions.base_session_service import ListSessionsResponse
from google.adk.sessions.session import Session
from google.adk.sessions.state import State


logger = logging.getLogger(__name__)

# Helper functions (mostly reused from your original code)
def _extract_state_delta(state: Optional[dict[str, Any]]): # Added Optional
    app_state_delta = {}
    user_state_delta = {}
    session_state_delta = {}
    if state:
        for key in state.keys():
            if key.startswith(State.APP_PREFIX):
                app_state_delta[key.removeprefix(State.APP_PREFIX)] = state[key]
            elif key.startswith(State.USER_PREFIX):
                user_state_delta[key.removeprefix(State.USER_PREFIX)] = state[key]
            elif not key.startswith(State.TEMP_PREFIX): # Ensure TEMP_PREFIX is handled
                session_state_delta[key] = state[key]
    return app_state_delta, user_state_delta, session_state_delta


def _merge_state(app_state: Optional[dict], user_state: Optional[dict], session_state: Optional[dict]) -> dict:
    # Ensure inputs are dictionaries, defaulting to empty if None
    s_state = session_state if session_state is not None else {}
    a_state = app_state if app_state is not None else {}
    u_state = user_state if user_state is not None else {}

    merged_state = copy.deepcopy(s_state) # Start with session state

    for key, value in a_state.items():
        merged_state[State.APP_PREFIX + key] = value
    for key, value in u_state.items():
        merged_state[State.USER_PREFIX + key] = value
    return merged_state


def _decode_content(
    content: Optional[dict[str, Any]],
) -> Optional[types.Content]:
    if not content:
        return None
    # Ensure content is mutable for modification
    content_copy = copy.deepcopy(content)
    for p in content_copy.get("parts", []): # Use .get for safety
        if "inline_data" in p and isinstance(p["inline_data"].get("data"), str):
            # Original code expects a list/tuple, but b64encode might return str
            # Safely assume it might be a single string from DB or previous encoding
            data_val = p["inline_data"]["data"]
            if isinstance(data_val, (list, tuple)) and data_val:
                 p["inline_data"]["data"] = base64.b64decode(data_val[0])
            elif isinstance(data_val, str):
                 p["inline_data"]["data"] = base64.b64decode(data_val)
            # If it's already bytes, do nothing.
    return types.Content.model_validate(content_copy)

def _encode_content_for_db(event_content: Optional[types.Content]) -> Optional[dict[str, Any]]:
    if not event_content:
        return None
    encoded_content = event_content.model_dump(exclude_none=True)
    # Workaround for multimodal Content throwing JSON not serializable
    # error with SQLAlchemy (and potentially direct JSON dump if data is bytes).
    for p in encoded_content.get("parts", []):
        if "inline_data" in p and isinstance(p["inline_data"].get("data"), bytes):
            p["inline_data"]["data"] = base64.b64encode(
                p["inline_data"]["data"]
            ).decode("utf-8")
    return encoded_content

def _datetime_to_iso(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc) # Assume UTC if naive
    return dt.isoformat()

def _iso_to_datetime(iso_str: Optional[str]) -> Optional[datetime]:
    if iso_str is None:
        return None
    return datetime.fromisoformat(iso_str.replace('Z', '+00:00'))


class SupabaseSessionService(BaseSessionService):
    """A session service that uses Supabase (PostgreSQL) for storage."""

    def __init__(self, supabase_url: str, supabase_key: str):
        """
        Args:
            supabase_url: The URL of your Supabase project.
            supabase_key: The anon or service_role key for your Supabase project.
        """
        try:
            self.supabase: Client = create_client(supabase_url, supabase_key)
        except Exception as e:
            logger.error(f"Failed to create Supabase client: {e}")
            raise ValueError(f"Failed to create Supabase client for URL '{supabase_url}'") from e
        logger.info(f"Supabase client initialized for URL: {supabase_url}")
        # Table creation is handled by SQL DDL in Supabase UI or migrations, not here.

    def _get_app_state(self, app_name: str) -> dict[str, Any]:
        # response_obj will be an APIResponse object if a row is found, or None if no row is found.
        response_obj: Optional[APIResponse] = self.supabase.table("app_states") \
                                                  .select("state") \
                                                  .eq("app_name", app_name) \
                                                  .maybe_single() \
                                                  .execute()

        if response_obj and response_obj.data:
            # response_obj.data here is the dictionary of the single row, e.g., {"state": {"theme": "dark"}}
            # We want the value associated with the "state" key from this row dictionary.
            return response_obj.data.get("state", {})
        else:
            # This handles two cases:
            # 1. response_obj is None (no row found)
            # 2. response_obj is an APIResponse, but response_obj.data is None/empty (shouldn't happen with maybe_single if a row is selected, but good for robustness)
            return {}

    def _get_user_state(self, app_name: str, user_id: str) -> dict[str, Any]:
        # Similar logic for user state
        response_obj: Optional[APIResponse] = self.supabase.table("user_states") \
                                                   .select("state") \
                                                   .eq("app_name", app_name) \
                                                   .eq("user_id", user_id) \
                                                   .maybe_single() \
                                                   .execute()
        if response_obj and response_obj.data:
            return response_obj.data.get("state", {})
        else:
            return {}

    def _update_app_state(self, app_name: str, state_delta: dict[str, Any]):
        if not state_delta:
            return
        current_state = self._get_app_state(app_name)
        current_state.update(state_delta)
        self.supabase.table("app_states").upsert({"app_name": app_name, "state": current_state}).execute()

    def _update_user_state(self, app_name: str, user_id: str, state_delta: dict[str, Any]):
        if not state_delta:
            return
        current_state = self._get_user_state(app_name, user_id)
        current_state.update(state_delta)
        self.supabase.table("user_states").upsert({"app_name": app_name, "user_id": user_id, "state": current_state}).execute()


    def create_session(
        self,
        *,
        app_name: str,
        user_id: str,
        state: Optional[dict[str, Any]] = None,
        session_id: Optional[str] = None, # Can be string UUID
    ) -> Session:
        app_state = self._get_app_state(app_name)
        user_state = self._get_user_state(app_name, user_id)

        # Create state entries if they don't exist (covered by upsert later if deltas exist)
        # Ensure they exist for merging even if no deltas initially
        if not self.supabase.table("app_states").select("app_name").eq("app_name", app_name).maybe_single().execute().data:
            self.supabase.table("app_states").insert({"app_name": app_name, "state": {}}).execute()
        if not self.supabase.table("user_states").select("app_name").eq("app_name", app_name).eq("user_id", user_id).maybe_single().execute().data:
            self.supabase.table("user_states").insert({"app_name": app_name, "user_id": user_id, "state": {}}).execute()


        app_state_delta, user_state_delta, session_state_data = _extract_state_delta(state)

        if app_state_delta:
            self._update_app_state(app_name, app_state_delta)
            app_state.update(app_state_delta) # Keep local copy in sync
        if user_state_delta:
            self._update_user_state(app_name, user_id, user_state_delta)
            user_state.update(user_state_delta) # Keep local copy in sync

        session_data_to_insert: dict[str, Any] = {
            "app_name": app_name,
            "user_id": user_id,
            "state": session_state_data,
        }
        if session_id: # Allow pre-defined session ID
            session_data_to_insert["id"] = session_id
        # create_time and update_time have DB defaults

        response: APIResponse = self.supabase.table("sessions").insert(session_data_to_insert).execute()
        if not response.data:
            raise Exception(f"Failed to create session: {response.error.message if response.error else 'Unknown error'}")

        created_session_data = response.data[0]
        session_actual_state = created_session_data.get("state", {})
        merged_state = _merge_state(app_state, user_state, session_actual_state)
        
        update_time_dt = _iso_to_datetime(created_session_data["update_time"])
        
        return Session(
            app_name=created_session_data["app_name"],
            user_id=created_session_data["user_id"],
            id=str(created_session_data["id"]), # Ensure it's string
            state=merged_state,
            last_update_time=update_time_dt.timestamp() if update_time_dt else datetime.now(timezone.utc).timestamp(),
        )

    def get_session(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
        config: Optional[GetSessionConfig] = None,
    ) -> Optional[Session]:
        session_resp: APIResponse = (
            self.supabase.table("sessions")
            .select("*")
            .eq("app_name", app_name)
            .eq("user_id", user_id)
            .eq("id", session_id)
            .maybe_single()
            .execute()
        )

        if not session_resp or not session_resp.data:
            return None
        
        db_session = session_resp.data

        events_query = (
            self.supabase.table("events")
            .select("*")
            .eq("app_name", app_name)
            .eq("user_id", user_id)
            .eq("session_id", db_session["id"])
            .order("\"timestamp\"", desc=True) # Recent events first
        )

        if config:
            if config.after_timestamp:
                # Convert float timestamp to ISO string for query
                after_dt = datetime.fromtimestamp(config.after_timestamp, tz=timezone.utc)
                events_query = events_query.lt("\"timestamp\"", _datetime_to_iso(after_dt)) # Events before this timestamp
            if config.num_recent_events is not None: # Check for None explicitly
                events_query = events_query.limit(config.num_recent_events)
        
        events_resp: APIResponse = events_query.execute()
        db_events = events_resp.data if events_resp.data else []
        # Since we ordered desc for limit, reverse to get chronological for processing
        db_events.reverse()


        app_state = self._get_app_state(app_name)
        user_state = self._get_user_state(app_name, user_id)
        session_state_data = db_session.get("state", {}) # Use .get with default

        merged_state = _merge_state(app_state, user_state, session_state_data)

        update_time_dt = _iso_to_datetime(db_session["update_time"])
        session = Session(
            app_name=db_session["app_name"],
            user_id=db_session["user_id"],
            id=str(db_session["id"]),
            state=merged_state,
            last_update_time=update_time_dt.timestamp() if update_time_dt else datetime.now(timezone.utc).timestamp(),
        )
        
        session.events = [
            Event(
                id=e["id"],
                author=e["author"],
                branch=e.get("branch"), # Use .get for nullable fields
                invocation_id=e["invocation_id"],
                content=_decode_content(e.get("content")),
                actions=e.get("actions"), # Assuming actions are stored as JSON and directly usable
                timestamp=_iso_to_datetime(e["timestamp"]).timestamp() if e.get("timestamp") else datetime.now(timezone.utc).timestamp(),
                long_running_tool_ids=set(json.loads(e["long_running_tool_ids_json"])) if e.get("long_running_tool_ids_json") else set(),
                grounding_metadata=e.get("grounding_metadata"),
                partial=e.get("partial"),
                turn_complete=e.get("turn_complete"),
                error_code=e.get("error_code"),
                error_message=e.get("error_message"),
                interrupted=e.get("interrupted"),
            )
            for e in db_events
        ]
        return session

    def list_sessions(
        self, *, app_name: str, user_id: str
    ) -> ListSessionsResponse:
        response: APIResponse = (
            self.supabase.table("sessions")
            .select("id, update_time")
            .eq("app_name", app_name)
            .eq("user_id", user_id)
            .execute()
        )
        
        sessions = []
        if response.data:
            for db_session in response.data:
                update_time_dt = _iso_to_datetime(db_session["update_time"])
                session = Session(
                    app_name=app_name, # Passed in, not from select
                    user_id=user_id,   # Passed in, not from select
                    id=str(db_session["id"]),
                    state={}, # Per original implementation, state is empty for list
                    last_update_time=update_time_dt.timestamp() if update_time_dt else datetime.now(timezone.utc).timestamp(),
                )
                sessions.append(session)
        return ListSessionsResponse(sessions=sessions)

    def delete_session(
        self, app_name: str, user_id: str, session_id: str
    ) -> None:
        # ON DELETE CASCADE in SQL schema will handle associated events
        self.supabase.table("sessions").delete().match({
            "app_name": app_name,
            "user_id": user_id,
            "id": session_id
        }).execute()
        # No explicit commit needed; Supabase operations are typically auto-committed.

    def append_event(self, session: Session, event: Event) -> Event:
        logger.info(f"Append event: {event.id} to session {session.id}")

        if event.partial and not event.turn_complete: # Original logic skips partial, but maybe only if not turn_complete?
                                                # ADK spec usually means partial events are for streaming UI, not persistence
                                                # but let's stick to the original code's `if event.partial: return event`
            logger.debug(f"Skipping append for partial event {event.id} unless it completes a turn.")
            # The original code only returns `event` if `event.partial` is true.
            # It doesn't consider `event.turn_complete`.
            # To match exactly:
            if event.partial:
                 return event


        # 1. Check if timestamp is stale
        # Fetch current session update_time from DB for optimistic concurrency check
        session_resp: APIResponse = (
            self.supabase.table("sessions")
            .select("update_time, state") # Also fetch state to update it
            .eq("app_name", session.app_name)
            .eq("user_id", session.user_id)
            .eq("id", session.id)
            .single() # Expect one session
            .execute()
        )

        if not session_resp.data:
            raise ValueError(f"Session {session.id} not found for appending event.")

        db_session_data = session_resp.data
        db_update_time = _iso_to_datetime(db_session_data["update_time"])

        if db_update_time and db_update_time.timestamp() > session.last_update_time:
            raise ValueError(
                f"Session last_update_time {session.last_update_time} is stale. "
                f"Storage update_time is {db_update_time.timestamp()}"
            )

        # 2. Update states based on event actions
        app_state_delta, user_state_delta, session_state_delta = {}, {}, {}
        if event.actions and event.actions.state_delta: # Assuming EventActions has state_delta
            # If event.actions is not a Pydantic model but a dict from JSONB:
            state_delta_from_actions = event.actions.get("state_delta") if isinstance(event.actions, dict) else event.actions.state_delta

            if state_delta_from_actions:
                app_state_delta, user_state_delta, session_state_delta = _extract_state_delta(
                     state_delta_from_actions
                )

        if app_state_delta:
            self._update_app_state(session.app_name, app_state_delta)
        if user_state_delta:
            self._update_user_state(session.app_name, session.user_id, user_state_delta)
        
        # Update session state in DB if there are changes
        if session_state_delta:
            current_session_state = db_session_data.get("state", {})
            current_session_state.update(session_state_delta)
            self.supabase.table("sessions").update({"state": current_session_state}).eq("id", session.id).execute()
            session.state.update(session_state_delta) # Update in-memory session state as well

        # 3. Store event to table
        # Prepare actions for JSONB. If Event.actions is a Pydantic model, use model_dump()
        actions_for_db = None
        if event.actions:
            if hasattr(event.actions, 'model_dump'): # Pydantic model
                actions_for_db = event.actions.model_dump(exclude_none=True)
            elif isinstance(event.actions, dict): # Already a dict
                actions_for_db = event.actions
            else: # Attempt to convert, might fail for complex objects
                try:
                    actions_for_db = vars(event.actions)
                except TypeError:
                    logger.warning(f"Could not serialize event.actions of type {type(event.actions)} to dict for DB.")
                    actions_for_db = str(event.actions) # Fallback to string representation


        storage_event_data = {
            "id": event.id,
            "invocation_id": event.invocation_id,
            "author": event.author,
            "branch": event.branch,
            "actions": actions_for_db,
            "session_id": session.id,
            "app_name": session.app_name,
            "user_id": session.user_id,
            "timestamp": _datetime_to_iso(datetime.fromtimestamp(event.timestamp, tz=timezone.utc)),
            "long_running_tool_ids_json": json.dumps(list(event.long_running_tool_ids)) if event.long_running_tool_ids else None,
            "grounding_metadata": event.grounding_metadata,
            "partial": event.partial,
            "turn_complete": event.turn_complete,
            "error_code": event.error_code,
            "error_message": event.error_message,
            "interrupted": event.interrupted,
            "content": _encode_content_for_db(event.content)
        }

        event_insert_resp: APIResponse = self.supabase.table("events").insert(storage_event_data).execute()
        if not event_insert_resp.data:
             raise Exception(f"Failed to append event: {event_insert_resp.error.message if event_insert_resp.error else 'Unknown error'}")

        # After event insert, the session's update_time in DB would have been touched by the trigger
        # (if event insert causes session update, or if we manually update session.update_time)
        # For consistency, explicitly update session's update_time.
        # The trigger on 'sessions' table handles setting update_time = now().
        # We just need to perform an update operation. A minimal update is fine.
        # If session_state_delta was applied, update_time is already new.
        # If not, we should touch it.
        if not session_state_delta: # Only touch if no other state update happened
            # A bit of a hack: update with the same state to trigger update_time
            # A better way might be a specific RPC function `touch_session(session_id)`
            self.supabase.table("sessions").update({"state": db_session_data.get("state", {})}).eq("id", session.id).execute()


        # Refresh session's last_update_time from the database after all operations
        refreshed_session_resp: APIResponse = (
            self.supabase.table("sessions")
            .select("update_time")
            .eq("id", session.id)
            .single()
            .execute()
        )
        if refreshed_session_resp.data and refreshed_session_resp.data.get("update_time"):
            new_update_time = _iso_to_datetime(refreshed_session_resp.data["update_time"])
            session.last_update_time = new_update_time.timestamp() if new_update_time else datetime.now(timezone.utc).timestamp()
        else: # Fallback if refresh fails
            session.last_update_time = datetime.now(timezone.utc).timestamp()


        # Also update the in-memory session (as per original super call)
        super().append_event(session=session, event=event) # This updates session.events and session.state
        return event

    def list_events(
        self,
        *,
        app_name: str,
        user_id: str,
        session_id: str,
    ) -> ListEventsResponse:
        # Check if session exists first
        session_check = self.supabase.table("sessions").select("id").eq("id", session_id).eq("app_name", app_name).eq("user_id", user_id).maybe_single().execute()
        if not session_check.data:
            # Or raise an error, depending on desired behavior for non-existent session
            logger.warning(f"Session not found: app='{app_name}', user='{user_id}', session='{session_id}' for list_events.")
            return ListEventsResponse(events=[])


        response: APIResponse = (
            self.supabase.table("events")
            .select("*")
            .eq("app_name", app_name)
            .eq("user_id", user_id)
            .eq("session_id", session_id)
            .order("\"timestamp\"", desc=False) # Chronological order
            .execute()
        )
        
        events = []
        if response.data:
            for e_data in response.data:
                event_ts = _iso_to_datetime(e_data["timestamp"])
                event = Event(
                    id=e_data["id"],
                    author=e_data["author"],
                    branch=e_data.get("branch"),
                    invocation_id=e_data["invocation_id"],
                    content=_decode_content(e_data.get("content")),
                    actions=e_data.get("actions"),
                    timestamp=event_ts.timestamp() if event_ts else datetime.now(timezone.utc).timestamp(),
                    long_running_tool_ids=set(json.loads(e_data["long_running_tool_ids_json"])) if e_data.get("long_running_tool_ids_json") else set(),
                    grounding_metadata=e_data.get("grounding_metadata"),
                    partial=e_data.get("partial"),
                    turn_complete=e_data.get("turn_complete"),
                    error_code=e_data.get("error_code"),
                    error_message=e_data.get("error_message"),
                    interrupted=e_data.get("interrupted"),
                )
                events.append(event)
        return ListEventsResponse(events=events)