class SessionManager:
    """
    Simple in-memory session store for CSPService.
    Replace with Redis or DB if scaling later.
    """
    def __init__(self):
        self.sessions = {}

    def set(self, session_id, obj):
        self.sessions[session_id] = obj

    def get(self, session_id):
        return self.sessions.get(session_id)

    def delete(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]