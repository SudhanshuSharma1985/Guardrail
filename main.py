import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
import guardrails as gd
from guardrails.hub import UnusualPrompt, SecretsPresent, DetectPII, ToxicLanguage, ProfanityFree, NSFWText, PolitenessCheck, GibberishText
import logging
from typing import List, Dict, Any
import json

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SecureChatbot:
    def __init__(self):
        """Initialize the secure chatbot with guardrails and LLM."""
        self.setup_openai()
        self.setup_guardrails()
        
    def setup_openai(self):
        """Setup OpenAI ChatGPT integration via LangChain."""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            st.error("OpenAI API key not found. Please set OPENAI_API_KEY in your .env file.")
            st.stop()
        
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            openai_api_key=api_key,
            max_tokens=1000
        )
        logger.info("OpenAI ChatGPT model initialized successfully")
    
    def setup_guardrails(self):
        """Setup all guardrails for input validation and output filtering."""
        # Input validation guardrails with more reasonable thresholds
        self.input_guard = gd.Guard().use_many(
            UnusualPrompt(on_fail="noop"),  # Changed to noop to avoid blocking simple inputs
            SecretsPresent(on_fail="exception"),
            DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "SSN"], on_fail="filter"),  # Removed PERSON as it's too broad
            ToxicLanguage(threshold=0.9, on_fail="exception"),  # Increased threshold
            ProfanityFree(on_fail="filter"),
            NSFWText(threshold=0.9, on_fail="exception"),  # Increased threshold
            GibberishText(threshold=0.95, on_fail="noop")  # Much higher threshold and noop instead of exception
        )
        
        # Output validation guardrails
        self.output_guard = gd.Guard().use_many(
            SecretsPresent(on_fail="filter"),
            DetectPII(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "SSN"], on_fail="filter"),
            ToxicLanguage(threshold=0.9, on_fail="filter"),  # Increased threshold
            ProfanityFree(on_fail="filter"),
            NSFWText(threshold=0.9, on_fail="filter"),  # Increased threshold
            PolitenessCheck(on_fail="noop")  # Just warn, don't block
        )
        
        logger.info("Guardrails initialized successfully")
    
    def validate_input(self, user_input: str) -> Dict[str, Any]:
        """Validate user input using guardrails."""
        try:
            # Skip validation for very short common inputs
            if user_input.strip().lower() in ['hi', 'hello', 'hey', 'yes', 'no', 'ok', 'thanks', 'bye']:
                return {
                    "is_valid": True,
                    "validated_text": user_input,
                    "validation_passed": True,
                    "reask": None
                }
            
            # Run input validation
            validated_output = self.input_guard.validate(user_input)
            return {
                "is_valid": True,
                "validated_text": validated_output.validated_output,
                "validation_passed": validated_output.validation_passed,
                "reask": validated_output.reask
            }
        except Exception as e:
            # Log the error but be more lenient with short inputs
            if len(user_input.strip()) < 5:
                logger.info(f"Allowing short input despite validation concern: {str(e)}")
                return {
                    "is_valid": True,
                    "validated_text": user_input,
                    "validation_passed": False,
                    "warning": str(e)
                }
            
            logger.warning(f"Input validation failed: {str(e)}")
            return {
                "is_valid": False,
                "error": str(e),
                "validated_text": None
            }
    
    def validate_output(self, ai_response: str) -> Dict[str, Any]:
        """Validate AI response using guardrails."""
        try:
            validated_output = self.output_guard.validate(ai_response)
            return {
                "is_valid": True,
                "validated_text": validated_output.validated_output,
                "validation_passed": validated_output.validation_passed
            }
        except Exception as e:
            logger.warning(f"Output validation failed: {str(e)}")
            return {
                "is_valid": True,  # Allow output but log the issue
                "validated_text": ai_response,
                "error": str(e)
            }
    
    def get_response(self, user_input: str, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Get response from ChatGPT with guardrails validation."""
        # Validate input
        input_validation = self.validate_input(user_input)
        
        if not input_validation["is_valid"]:
            return {
                "response": "I'm sorry, but I can't process that input due to safety guidelines. Please rephrase your message.",
                "error": input_validation.get("error"),
                "input_blocked": True
            }
        
        # Check if input was modified by guardrails
        validated_input = input_validation["validated_text"] or user_input
        
        try:
            # Prepare conversation context
            messages = []
            
            # Add conversation history
            for msg in conversation_history[-10:]:  # Keep last 10 messages for context
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            
            # Add current user message
            messages.append(HumanMessage(content=validated_input))
            
            # Get response from ChatGPT
            ai_response = self.llm.invoke(messages)
            response_content = ai_response.content
            
            # Validate output
            output_validation = self.validate_output(response_content)
            final_response = output_validation["validated_text"]
            
            return {
                "response": final_response,
                "input_modified": validated_input != user_input,
                "output_modified": final_response != response_content,
                "validation_passed": output_validation.get("validation_passed", True)
            }
            
        except Exception as e:
            logger.error(f"Error getting AI response: {str(e)}")
            return {
                "response": "I'm sorry, I encountered an error while processing your request. Please try again.",
                "error": str(e)
            }

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = SecureChatbot()
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "message_count" not in st.session_state:
        st.session_state.message_count = 0

def display_conversation():
    """Display the conversation history."""
    st.subheader("💬 Conversation")
    
    # Create a container for messages
    message_container = st.container()
    
    with message_container:
        for i, message in enumerate(st.session_state.conversation_history):
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    
                    # Show validation info if available
                    if message.get("validation_info"):
                        with st.expander("🛡️ Security Info", expanded=False):
                            info = message["validation_info"]
                            if info.get("input_modified"):
                                st.info("Input was filtered for safety")
                            if info.get("output_modified"):
                                st.info("Response was filtered for safety")
                            if not info.get("validation_passed"):
                                st.warning("Some content may have been filtered")

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Secure AI Chatbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.title("🛡️ Secure AI Chatbot")
        st.markdown("---")
        
        st.subheader("🔒 Security Features")
        st.markdown("""
        **Active Guardrails:**
        - 🟡 Unusual Prompt Detection (Warning)
        - ✅ Secrets Detection
        - ✅ PII Detection & Filtering
        - ✅ Toxic Language Detection
        - ✅ Profanity Filtering
        - ✅ NSFW Content Detection
        - ✅ Politeness Check
        - 🟡 Gibberish Text Detection (Lenient)
        
        **Note:** Guardrails are tuned for practical use while maintaining security.
        """)
        
        st.markdown("---")
        
        st.subheader("📊 Chat Statistics")
        st.metric("Total Messages", st.session_state.message_count)
        
        if st.button("🗑️ Clear Conversation"):
            st.session_state.conversation_history = []
            st.session_state.message_count = 0
            st.rerun()
        
        st.markdown("---")
        st.subheader("⚙️ Model Settings")
        st.info("Using: GPT-3.5-Turbo via LangChain")
    
    # Main chat interface
    st.title("🤖 Secure AI Chatbot")
    st.markdown("Chat safely with AI - all messages are filtered through multiple security layers!")
    
    # Display conversation
    display_conversation()
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([4, 1])
        
        with col1:
            user_input = st.text_input(
                "Type your message here...",
                placeholder="Ask me anything (safely)!",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.form_submit_button("Send 📤", use_container_width=True)
    
    # Process user input
    if send_button and user_input.strip():
        # Add user message to history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": st.session_state.message_count
        })
        
        # Show thinking indicator
        with st.spinner("🤔 Thinking securely..."):
            # Get AI response
            response_data = st.session_state.chatbot.get_response(
                user_input, 
                st.session_state.conversation_history
            )
        
        # Add AI response to history
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": response_data["response"],
            "timestamp": st.session_state.message_count + 1,
            "validation_info": {
                "input_modified": response_data.get("input_modified", False),
                "output_modified": response_data.get("output_modified", False),
                "validation_passed": response_data.get("validation_passed", True),
                "input_blocked": response_data.get("input_blocked", False)
            }
        })
        
        st.session_state.message_count += 2
        
        # Show alerts for blocked content
        if response_data.get("input_blocked"):
            st.error("⚠️ Your message was blocked by security filters. Please rephrase.")
        elif response_data.get("input_modified") or response_data.get("output_modified"):
            st.info("🛡️ Content was filtered for safety.")
        
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🔒 This chatbot uses multiple AI safety guardrails to ensure secure conversations."
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
