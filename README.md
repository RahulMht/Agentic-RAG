# Agentic Chatbot

Agentic-RAG is an intelligent conversational agent designed to answer user queries from documents, schedule calls, and manage user information. It leverages LangChain, Gradio, and other modern libraries to provide a seamless user experience.

## Features

- **Document Q&A**: Ask questions about loaded documents and get accurate answers.
- **Appointment Scheduling**: Schedule calls and meetings using natural language date inputs.
- **Information Management**: Update contact details, view conversation history, and manage appointments.
- **Memory**: Maintains conversation context to provide relevant responses.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/RahulMht/Agentic-RAG.git
   cd agentic-RAG
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   - Create a `.env` file in the root directory.
   - Add your Groq API key:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```

## Usage

1. **Run the chatbot**:
   ```bash
   python main.py
   ```

2. **Interact with the chatbot**:
   - Type your queries or commands.
   - Use "quit" to exit the chatbot.

## Commands

- **Schedule a Call**: "I want to schedule a call"
- **Show Scheduled Calls**: "show scheduled calls"
- **Clear Chat History**: "clear chat"
- **Help**: "help"

## Development

- **Agentic_RAG.py**: Main logic for document processing and scheduling.
- **agent.py**: Custom agent implementation.
- **main.py**: Entry point for running the chatbot.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain)
- [Hugging Face](https://huggingface.co/)
