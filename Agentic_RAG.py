import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict, List
import re
from langchain_community.document_loaders import UnstructuredPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain.agents import Tool
from dateutil import parser, relativedelta
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU
import phonenumbers
import pytz

# Load environment variables from .env file
load_dotenv()

# Use environment variable for API key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

class ChatBot:
    def __init__(self):
        self.setup_document_qa()
        self.setup_tools()
        self.user_info = {}
        self.scheduled_calls = []  # List to store scheduled calls
        
    def setup_document_qa(self):
        try:
            # Load and process documents
            loader = DirectoryLoader(
                "./documents", 
                glob="**/*.pdf",
                loader_cls=UnstructuredPDFLoader
            )
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Setup embeddings and vectorstore
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.vectorstore = Chroma.from_documents(splits, embeddings)
            
            # Initialize LLM with correct model name
            llm = ChatGroq(
                temperature=0.7,
                model_name="llama-3.2-3b-preview",
                max_tokens=4096
            )
            
            # Setup memory with updated parameters if any
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="answer"
            )
            
            # Setup QA chain with verbose=False
            self.qa_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=self.vectorstore.as_retriever(),
                memory=self.memory,
                verbose=False,
                return_source_documents=True,
                chain_type="stuff"
            )
            
        except Exception as e:
            print(f"Error in setup_document_qa: {str(e)}")
            raise

    def setup_tools(self):
        self.tools = [
            Tool(
                name="Schedule_Appointment",
                func=self.handle_scheduling,
                description="Use this to schedule appointments and collect user information"
            ),
            Tool(
                name="Collect_User_Info",
                func=self.collect_user_info,
                description="Use this to collect user's name, email, and phone number"
            )
        ]

    def validate_email(self, email: str) -> bool:
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(pattern, email))
    
    def validate_phone(self, phone: str) -> bool:
        try:
            parsed = phonenumbers.parse(phone, "US")
            return phonenumbers.is_valid_number(parsed)
        except:
            return False
            
    def parse_date(self, date_string: str) -> str:
        try:
            # Get current date and time in local timezone
            now = datetime.now(pytz.timezone('Asia/Kathmandu'))
            
            # Handle relative dates
            lower_query = date_string.lower()
            
            # Handle "today"
            if "today" in lower_query:
                return now.strftime("%Y-%m-%d")
            
            # Handle "tomorrow"
            if "tomorrow" in lower_query:
                tomorrow = now + timedelta(days=1)
                return tomorrow.strftime("%Y-%m-%d")
            
            # Handle "next week"
            if "next week" in lower_query:
                next_week = now + timedelta(days=7)
                return next_week.strftime("%Y-%m-%d")
            
            # Handle specific days of the week
            weekday_mapping = {
                'monday': MO, 'tuesday': TU, 'wednesday': WE,
                'thursday': TH, 'friday': FR, 'saturday': SA, 'sunday': SU
            }
            
            for day, day_const in weekday_mapping.items():
                if f"next {day}" in lower_query:
                    next_day = now + relativedelta(weekday=day_const(+1))
                    return next_day.strftime("%Y-%m-%d")
                elif day in lower_query:
                    # If just the day is mentioned, get the next occurrence
                    next_occurrence = now + relativedelta(weekday=day_const(+1))
                    return next_occurrence.strftime("%Y-%m-%d")
                
            # Handle "next month"
            if "next month" in lower_query:
                next_month = now + relativedelta(months=1)
                return next_month.strftime("%Y-%m-%d")
            
            # Handle specific date formats
            try:
                # Try parsing various date formats
                parsed_date = parser.parse(date_string, dayfirst=False, fuzzy=True)
                # If the year is not specified, assume current year
                if parsed_date.year == 1900:
                    parsed_date = parsed_date.replace(year=now.year)
                # If the parsed date is in the past, assume next occurrence
                if parsed_date.date() < now.date():
                    if parsed_date.month < now.month:
                        parsed_date = parsed_date.replace(year=now.year + 1)
                return parsed_date.strftime("%Y-%m-%d")
                
            except:
                # Handle MM/DD/YYYY format
                if '/' in date_string:
                    parts = date_string.split('/')
                    if len(parts) == 3:
                        month, day, year = map(int, parts)
                        parsed_date = datetime(year, month, day)
                        return parsed_date.strftime("%Y-%m-%d")
                    elif len(parts) == 2:
                        # If year is omitted, assume current year
                        month, day = map(int, parts)
                        parsed_date = datetime(now.year, month, day)
                        if parsed_date.date() < now.date():
                            parsed_date = parsed_date.replace(year=now.year + 1)
                        return parsed_date.strftime("%Y-%m-%d")
            
            return None
            
        except Exception as e:
            print(f"Date parsing error: {str(e)}")
            return None
    
    def collect_user_info(self):
        user_info = {}
        
        # Collect name
        name = input("Please enter your name: ")
        user_info['name'] = name
        
        # Collect and validate email
        while True:
            email = input("Please enter your email: ")
            if self.validate_email(email):
                user_info['email'] = email
                break
            print("Invalid email format. Please try again.")
        
        # Collect and validate phone
        while True:
            phone = input("Please enter your phone number (with country code, e.g., +977 9818000000): ")
            if self.validate_phone(phone):
                user_info['phone'] = phone
                break
            print("Invalid phone number format. Please try again.")
            
        self.user_info = user_info
        return user_info

    def handle_scheduling(self, query: str):
        try:
            # Collect user information if not already present
            if not self.user_info:
                self.user_info = self.collect_user_info()
            
            # If query contains a date, try to parse it
            date_str = self.parse_date(query)
            if date_str:
                response = f"\nI've scheduled a call for {date_str} with:\n"
                response += f"Name: {self.user_info['name']}\n"
                response += f"Email: {self.user_info['email']}\n"
                response += f"Phone: {self.user_info['phone']}\n"
                
                # Store the scheduled call
                self.scheduled_calls.append({
                    'date': date_str,
                    'name': self.user_info['name'],
                    'email': self.user_info['email'],
                    'phone': self.user_info['phone']
                })
                
                return response
            else:
                return """When would you like to schedule the call?"""
            
        except Exception as e:
            return f"I apologize, but I encountered an error while scheduling: {str(e)}"

    def show_scheduled_calls(self):
        """Return a formatted list of scheduled calls"""
        if not self.scheduled_calls:
            return "No scheduled calls at the moment."
        
        response = "Here are your scheduled calls:\n"
        for call in self.scheduled_calls:
            response += f"- Date: {call['date']}, Name: {call['name']}, Email: {call['email']}, Phone: {call['phone']}\n"
        return response

    def clear_history(self):
        """Clear the conversation history"""
        try:
            self.memory.clear()
            self.chat_history = []
            return "Chat history has been cleared."
        except Exception as e:
            return f"Error clearing chat history: {str(e)}"

    def process_query(self, query: str):
        # Add clear history command
        if query.lower().strip() in ['clear history', 'clear chat', 'erase history', 'erase chat']:
            return self.clear_history()
        
        # Check for request to show scheduled calls
        if "show scheduled calls" in query.lower():
            return self.show_scheduled_calls()
            
        # Enhanced scheduling keywords
        scheduling_keywords = [
            'schedule', 'book', 'appointment', 'call', 'meet',
            'want to talk', 'discuss', 'consultation', 'meeting',
            'set up', 'arrange', 'plan', 'catch up', 'sync',
            'connect', 'get in touch', 'reach out'
        ]
        
        # Time-related keywords
        time_keywords = [
            'morning', 'afternoon', 'evening', 'night',
            'today', 'tomorrow', 'next', 'weekend',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
            'saturday', 'sunday', 'week', 'month',
            'am', 'pm', 'o\'clock', ':00'
        ]
        
        # Help keywords
        help_keywords = [
            'help', 'how to', 'guide', 'explain',
            'what can you do', 'capabilities', 'features'
        ]
        
        try:
            lower_query = query.lower()
            
            # Handle help requests
            if any(keyword in lower_query for keyword in help_keywords):
                return self.show_help()
            
            # Handle memory-related queries
            if any(phrase in lower_query for phrase in [
                'what did i just', 'last question', 'previous',
                'what were we talking about', 'what was i saying'
            ]):
                return self.get_conversation_history()
            
            # Handle scheduling context
            if self.user_info:
                # User has already provided contact info
                if any(word in lower_query for word in time_keywords):
                    return self.handle_scheduling(query)
                elif 'cancel' in lower_query or 'reschedule' in lower_query:
                    return self.handle_cancellation()
                elif 'change' in lower_query and ('email' in lower_query or 'phone' in lower_query or 'name' in lower_query):
                    return self.update_user_info(query)
            
            # Handle new scheduling requests
            if any(keyword in lower_query for keyword in scheduling_keywords):
                return self.handle_scheduling(query)
            
            # Handle basic commands
            if lower_query.strip() in ['call', 'schedule', 'book']:
                return self.show_scheduling_help()
            
            # Handle document QA queries
            result = self.qa_chain.invoke({"question": query, "chat_history": self.memory.chat_memory.messages})
            return result["answer"]
            
        except Exception as e:
            error_msg = f"I encountered an error processing your query. Please try again. Error: {str(e)}"
            return error_msg

    def show_help(self):
        """Show available commands and features"""
        help_text = """
I can help you with the following:

1. Schedule Appointments:
   - "Schedule a call"
   - "Book an appointment"
   - "Let's meet next week"

2. Manage Your Information:
   - Update contact details
   - Cancel or reschedule appointments
   - Change your preferences

3. Answer Questions:
   - Ask about any topic in the loaded documents
   - Get clarification on previous responses
   - Check conversation history

4. Date and Time:
   - Use natural language ("next Monday", "tomorrow afternoon")
   - Specify exact dates (MM/DD/YYYY)
   - Set flexible time ranges

5. Commands:
   - "clear chat" - Erase conversation history
   - "help" - Show this help message
   - "cancel" - Cancel current operation

How can I assist you today?
"""
        return help_text

    def show_scheduling_help(self):
        return """
Would you like to schedule a call? Here's how:

1. Say "I want to schedule a call" or "Book an appointment"
2. I'll collect your contact information
3. Then specify your preferred time:
   - "tomorrow morning"
   - "next Monday afternoon"
   - "December 25th at 2pm"
   - Or any specific date (MM/DD/YYYY)

Please let me know how you'd like to proceed!
"""

    def handle_cancellation(self):
        if not self.user_info:
            return "There's no active appointment to cancel."
        
        self.user_info = {}
        return "Your appointment has been cancelled. Let me know if you'd like to schedule a new one!"

    def update_user_info(self, query):
        """Handle updates to user information"""
        try:
            if 'email' in query.lower():
                email = input("Please enter your new email: ")
                if self.validate_email(email):
                    self.user_info['email'] = email
                    return f"Email updated successfully to: {email}"
                return "Invalid email format. No changes made."
                
            elif 'phone' in query.lower():
                phone = input("Please enter your new phone number (with country code, e.g., +977 9818000000): ")
                if self.validate_phone(phone):
                    self.user_info['phone'] = phone
                    return f"Phone number updated successfully to: {phone}"
                return "Invalid phone format. No changes made."
                
            elif 'name' in query.lower():
                name = input("Please enter your new name: ")
                self.user_info['name'] = name
                return f"Name updated successfully to: {name}"
                
            return "What information would you like to update? (name, email, or phone)"
            
        except Exception as e:
            return f"Error updating information: {str(e)}"

    def get_conversation_history(self):
        """Get a summary of recent conversation"""
        try:
            chat_history = self.memory.chat_memory.messages
            if not chat_history:
                return "This is the start of our conversation."
                
            last_exchanges = chat_history[-4:]  # Get last 2 exchanges
            summary = "Recent conversation:\n\n"
            for msg in last_exchanges:
                if "Human" in str(msg):
                    summary += f"You: {str(msg).split('Human: ')[1]}\n"
                else:
                    summary += f"Bot: {str(msg).split('Assistant: ')[1]}\n"
            return summary
            
        except Exception as e:
            return f"Error retrieving conversation history: {str(e)}"