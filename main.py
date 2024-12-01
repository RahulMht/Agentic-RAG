from Agentic_RAG import ChatBot

def main():
    chatbot = ChatBot()
    
    print("Chatbot initialized. Type 'quit' to exit.")
    while True:
        query = input("You: ")
        if query.lower() == 'quit':
            break
            
        response = chatbot.process_query(query)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()