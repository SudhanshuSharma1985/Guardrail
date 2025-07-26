import streamlit as st
from datetime import datetime
import json

def export_conversation():
    """Export conversation history as JSON."""
    if st.session_state.conversation_history:
        export_data = {
            "export_date": datetime.now().isoformat(),
            "total_messages": len(st.session_state.conversation_history),
            "conversation": st.session_state.conversation_history
        }
        return json.dumps(export_data, indent=2)
    return None

def get_conversation_stats():
    """Get statistics about the current conversation."""
    if not st.session_state.conversation_history:
        return {"total_messages": 0, "user_messages": 0, "ai_messages": 0}
    
    user_messages = sum(1 for msg in st.session_state.conversation_history if msg["role"] == "user")
    ai_messages = sum(1 for msg in st.session_state.conversation_history if msg["role"] == "assistant")
    
    return {
        "total_messages": len(st.session_state.conversation_history),
        "user_messages": user_messages,
        "ai_messages": ai_messages
    }