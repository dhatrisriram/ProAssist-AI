# ProAssist-AI

An intelligent customer support agent leveraging AI-driven multi-agent workflows to manage and resolve customer queries in real-time with categorization, sentiment analysis, response generation, and escalation support.

---

## Features
- **Query Categorization:** Classifies customer queries into categories such as Technical, Billing, Shipping, Returns, Product Inquiry, and General.  
- **Sentiment Analysis:** Analyzes the sentiment of customer queries on a scale from very negative (-1.0) to very positive (1.0).    
- **Escalation Handling:** Automatically escalates queries with strong negative sentiment or containing urgent/emergency keywords to human agents.  
- **Dynamic Conditional Routing:** Routes queries intelligently based on category and urgency using a flexible LangGraph workflow.  
- **Session Conversation Tracking:** Maintains conversation history for contextual understanding and more coherent responses.  
- **Gradio-powered Chat Interface:** Offers a user-friendly interface for interactive customer support simulation.  
- **Interactive Chatbot Experience:** Allows real-time, two-way interaction with the bot where users can type queries and receive live responses.  

---

## Installation Requirements

- Python 3.8+
- langchain
- langchain_core
- langchain_groq
- langchain_community
- langgraph
- gradio
- dotenv

Install dependencies with:
pip install -r requirements.txt

## Usage

After setting your Groq API key in a `.env` file (`GROQ_API_KEY=your_key_here`), Run the application to launch the interactive customer support assistant in a web browser. Enter queries such as:

- "I can't log into my account"

- "How can I pay my bill?"

- "My package hasn't arrived yet"

- "I want to return a product"

- The system will respond with:
  - **Category** of the query
  - **Sentiment score** and explanation
  - **Tailored support response** or escalation message

---

## Example Interaction

![Wifi not working](https://i.postimg.cc/28yWQBpN/wifi.png)
![Wifi not working](https://i.postimg.cc/VNW5QYyF/wifi2.png)
![Escalation](https://i.postimg.cc/pTVb9vmk/escalation.png)
![Shipping](https://i.postimg.cc/hPBwhgNm/shipping.png) 
![Shipping](https://i.postimg.cc/bvsV4vS7/shipping2.png)

