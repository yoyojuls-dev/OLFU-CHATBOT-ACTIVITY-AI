import os
import json
import google.generativeai as genai
import logging
from flask import Flask, request, jsonify, render_template, url_for # Import url_for
from dotenv import load_dotenv
from pathlib import Path
import datetime
import random
import string
from google.cloud import firestore # Import Firestore
from google.api_core.exceptions import NotFound, GoogleAPICallError

# --- Basic Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
load_dotenv()
API_KEY = os.getenv('GEMINI_API_KEY')
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT') # Optional for local auth, required by Firestore client implicitly

if not API_KEY:
    logging.error("FATAL ERROR: GEMINI_API_KEY environment variable not set.")
    exit("Please set the GEMINI_API_KEY environment variable.")

# --- Configure Firestore ---
try:
    # If PROJECT_ID is set explicitly: db = firestore.Client(project=PROJECT_ID)
    # Otherwise, let the library try to discover it (works well in GCP environments)
    db = firestore.Client()
    SESSIONS_COLLECTION = 'chatSessions' # Firestore collection name
    logging.info(f"Firestore client initialized for project: {db.project}")
except Exception as e:
    logging.error(f"Error initializing Firestore client: {e}")
    logging.error("Ensure Firestore API is enabled and authentication is configured (e.g., `gcloud auth application-default login` or service account).")
    exit("Failed to initialize Firestore.")

# --- Configure Gemini ---
try:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction="Act as an specific chatbot for 'Our Lady of Fatima University' located at 1 Esperanza, Quezon City, 1118 Metro Manila\n\nParent organization: Our Lady of Fatima University Valenzuela City\n\nFocus only on Senior High School\n\nIf the query is not related to Senior High School of Our Lady of Fatima University - Quezon City Campus, say 'I do not know'\n",
    )
    generation_config = genai.types.GenerationConfig(
        temperature=1,
        top_p=0.95,
        top_k=40,
        max_output_tokens=8192,
    )
    logging.info("Gemini AI configured successfully.")
except Exception as e:
    logging.error(f"Error configuring Gemini AI: {e}")
    exit("Failed to configure Gemini AI.")


# --- Flask App Initialization ---
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Load Base Chat History ---
base_chat_history = []
chat_history_file_path = Path(__file__).parent / 'chat_history.json'
try:
    if chat_history_file_path.is_file():
        with open(chat_history_file_path, 'r', encoding='utf-8') as f:
            raw_history = json.load(f)
            # Filter out fileData parts
            for entry in raw_history:
                if 'parts' in entry:
                    filtered_parts = [part for part in entry['parts'] if 'fileData' not in part]
                    if filtered_parts:
                        base_chat_history.append({"role": entry["role"], "parts": filtered_parts})
            logging.info(f"Base chat history loaded from {chat_history_file_path}.")
    else:
        logging.warning(f"Base chat history file not found: {chat_history_file_path}")
except (json.JSONDecodeError, IOError, Exception) as e:
    logging.error(f"Error loading or parsing base chat history {chat_history_file_path}: {e}")
    logging.warning("Proceeding with empty base history.")
    base_chat_history = []


# --- Helper Functions for Firestore Persistence ---

def get_session_ref(session_id):
    """Gets the Firestore DocumentReference for a session."""
    if not session_id or not isinstance(session_id, str) or '/' in session_id: # Basic validation
        return None
    return db.collection(SESSIONS_COLLECTION).document(session_id)

def load_history_from_firestore(session_id):
    """Loads history from Firestore."""
    session_ref = get_session_ref(session_id)
    if not session_ref:
        logging.warning(f"Attempted to load history with invalid session ID: {session_id}")
        return None
    try:
        doc_snapshot = session_ref.get()
        if doc_snapshot.exists:
            data = doc_snapshot.to_dict()
            if data and isinstance(data.get('history'), list):
                logging.debug(f"[{session_id}] History loaded from Firestore.")
                return data['history']
            else:
                logging.warning(f"[{session_id}] Invalid history format in Firestore.")
                return None # Treat invalid format as non-existent
        else:
            # logging.debug(f"[{session_id}] Session document not found in Firestore.")
            return None # File doesn't exist
    except GoogleAPICallError as e:
         logging.error(f"[{session_id}] Firestore API call error loading history: {e}")
         return None # Treat errors as non-existent
    except Exception as e:
        logging.exception(f"[{session_id}] Unexpected error loading history from Firestore: {e}")
        return None

def save_history_to_firestore(session_id, history_list):
    """Saves history to Firestore."""
    session_ref = get_session_ref(session_id)
    if not session_ref:
         logging.error(f"Attempted to save history with invalid session ID: {session_id}")
         return False
    if not isinstance(history_list, list):
        logging.error(f"[{session_id}] Attempted to save non-list history to Firestore.")
        return False
    try:
        session_ref.set({
            'history': history_list,
            'lastUpdated': firestore.SERVER_TIMESTAMP # Use server timestamp
        }, merge=False) # Overwrite completely
        # logging.debug(f"[{session_id}] History saved to Firestore.")
        return True
    except GoogleAPICallError as e:
        logging.error(f"[{session_id}] Firestore API call error saving history: {e}")
        return False
    except Exception as e:
        logging.exception(f"[{session_id}] Unexpected error saving history to Firestore: {e}")
        return False

def generate_session_id():
    """Generates a simple unique session ID."""
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d%H%M%S%f")
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=8))
    return f"sid_{timestamp}_{random_part}"

# --- Flask Routes ---

@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat_handler():
    """Handles chat requests."""
    if not request.is_json:
        logging.warning("Received non-JSON request to /api/chat")
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    user_message = data.get('message')
    session_id = data.get('sessionId') # Can be None
    current_history = None

    if not user_message or not isinstance(user_message, str) or not user_message.strip():
        return jsonify({"error": "Invalid message provided."}), 400

    # 1. Load or initialize history
    if session_id:
        current_history = load_history_from_firestore(session_id)
        if current_history is None:
            logging.info(f"[{session_id}] Provided session ID not found or history invalid. Starting new session.")
            session_id = None # Force new session ID generation

    if current_history is None: # Need to start a new session
        session_id = generate_session_id()
        logging.info(f"[{session_id}] Initializing new session with base history.")
        # IMPORTANT: Use a deep copy method if base_chat_history contains mutable objects
        # For lists of simple dicts like this, list comprehension is usually okay
        current_history = [item.copy() for item in base_chat_history]

    try:
        # 2. Start chat session with the history
        logging.debug(f"[{session_id}] Starting chat with history: {json.dumps(current_history, indent=2)}")
        chat_session = model.start_chat(history=current_history)

        # 3. Send message to Gemini
        logging.info(f"[{session_id}] Sending message to Gemini: \"{user_message[:50]}...\"")
        response = chat_session.send_message(
            user_message,
            generation_config=generation_config
        )

        # --- Check for safety issues / blocked content ---
        # Accessing safety feedback might differ slightly based on exact SDK version
        # This is a common pattern:
        prompt_feedback = getattr(response, 'prompt_feedback', None)
        if prompt_feedback and getattr(prompt_feedback, 'block_reason', None):
             block_reason = getattr(prompt_feedback, 'block_reason')
             logging.warning(f"[{session_id}] Response blocked due to: {block_reason}")
             # Save history *up to* the blocked point
             save_history_to_firestore(session_id, current_history + [{"role": "user", "parts": [{"text": user_message}]}])
             return jsonify({"error": f"Response blocked due to safety settings ({block_reason}). Please rephrase your message."}), 400

        # Check response validity
        if not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
             logging.error(f"[{session_id}] Gemini returned an empty or invalid response structure.")
             save_history_to_firestore(session_id, current_history + [{"role": "user", "parts": [{"text": user_message}]}])
             return jsonify({"error": "Received an empty response from the AI."}), 500

        model_response_text = response.text # Use the convenient .text accessor
        logging.info(f"[{session_id}] Received response from Gemini.")

        # 4. Update history list and save to Firestore
        updated_history = current_history + [
            {"role": "user", "parts": [{"text": user_message}]},
            {"role": "model", "parts": [{"text": model_response_text}]}
        ]

        if not save_history_to_firestore(session_id, updated_history):
            logging.error(f"[{session_id}] CRITICAL: Failed to save updated history to Firestore!")
            # Continue to send response, but log the error prominently

        # 5. Return response to client
        return jsonify({"response": model_response_text, "sessionId": session_id})

    except Exception as e:
        logging.exception(f"[{session_id or 'NEW'}] An error occurred during chat processing: {e}")
        # Attempt to save history up to the error point
        if session_id and current_history:
             save_history_to_firestore(session_id, current_history + [{"role": "user", "parts": [{"text": user_message}]}])
        return jsonify({"error": "An internal server error occurred."}), 500


@app.route('/api/health')
def health_check():
    """Simple health check endpoint."""
    return jsonify({"status": "OK", "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()})

# --- Run the Flask App ---
if __name__ == '__main__':
    # Use port from environment or default to 3000 for consistency
    port = int(os.environ.get("PORT", 3000))
    # Use debug=True only for local development
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() in ["true", "1"]
    app.run(debug=debug_mode, host='0.0.0.0', port=port)