"""
Multi-provider LLM call chain — copied nearly verbatim from the production
app's apps/chatbot/rag.py (_call_llm/_call_gemini/_call_groq), minus the
Redis-backed budget tracker (cache.py), which isn't needed for a low-
traffic portfolio demo. Provider priority: Gemini > Groq.
"""
import os
import requests

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-flash-latest')
GEMINI_URL = 'https://generativelanguage.googleapis.com/v1beta/openai/chat/completions'

GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')
GROQ_MODEL = os.environ.get('GROQ_MODEL', 'openai/gpt-oss-20b')
GROQ_URL = 'https://api.groq.com/openai/v1/chat/completions'

GENERATION_OPTIONS = {'temperature': 0}

SYSTEM_PROMPT = (
    "You are a data analyst assistant for a Spotify streaming analytics "
    "platform. Answer the user's question using ONLY the context provided "
    "below. If the context doesn't contain the answer, say so — do not "
    "make up numbers. When you cite a fact, name the artist/country/label "
    "and time period it came from. "
    "Format your response in markdown: bold key numbers and entity names "
    "with **asterisks**, use a bullet list when presenting 2 or more facts "
    "or a comparison. Write large stream counts in abbreviated form "
    "(e.g. 18.6B, 245M, 3.2K) instead of the full digit string. "
    "Lead with a short natural-language sentence before any bullets/table."
)

SMALLTALK_SYSTEM_PROMPT = (
    "You are the friendly assistant for a Spotify streaming analytics "
    "platform (StreamPulse). Reply warmly and briefly (1-2 sentences, no "
    "markdown lists/tables) to this greeting/thanks/meta message. If asked "
    "what you can do, mention you can answer questions about country, "
    "artist, label, and song streaming performance, trends, and "
    "comparisons — grounded in real data, not guesses."
)


class AllProvidersUnavailable(Exception):
    pass


def call_llm(prompt, system_prompt=None):
    messages = [
        {'role': 'system', 'content': system_prompt or SYSTEM_PROMPT},
        {'role': 'user', 'content': prompt},
    ]
    if GEMINI_API_KEY:
        try:
            return _call_gemini(messages)
        except (requests.exceptions.HTTPError, requests.exceptions.Timeout,
                requests.exceptions.ConnectionError):
            pass  # fall through to Groq, same as production
    if GROQ_API_KEY:
        try:
            return _call_groq(messages)
        except (requests.exceptions.HTTPError, requests.exceptions.Timeout,
                requests.exceptions.ConnectionError):
            pass
    raise AllProvidersUnavailable(
        'Gemini/Groq both unset, over quota, or unreachable. '
        'Set GEMINI_API_KEY and/or GROQ_API_KEY as environment variables.'
    )


def _call_gemini(messages):
    response = requests.post(
        GEMINI_URL,
        headers={'Authorization': f'Bearer {GEMINI_API_KEY}', 'Content-Type': 'application/json'},
        json={
            'model': GEMINI_MODEL,
            'messages': messages,
            'temperature': GENERATION_OPTIONS['temperature'],
            'max_tokens': 4096,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def _call_groq(messages):
    response = requests.post(
        GROQ_URL,
        headers={'Authorization': f'Bearer {GROQ_API_KEY}', 'Content-Type': 'application/json'},
        json={
            'model': GROQ_MODEL,
            'messages': messages,
            'temperature': GENERATION_OPTIONS['temperature'],
            'max_tokens': 1024,
        },
        timeout=30,
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']
