# **ProAssist-AI**

This project creates an intelligent customer support agent that leverages multi-agent systems to manage customer queries in real-time. The system efficiently categorizes queries, analyzes sentiment and routes queries to the appropriate support channel—technical, billing or general. Negative sentiment queries are automatically escalated to human agents, ensuring prompt and effective resolutions.

## **Features**
- **Query Categorization**: Classify customer queries into Technical, Billing or General categories.
- **Sentiment Analysis**: Determine whether customer queries are Positive, Neutral or Negative.
- **Response Generation**: Automatically generate responses based on query category and sentiment.
- **Escalation**: Queries with negative sentiment are escalated to human agents.
- **Flexible Workflow**: Powered by LangGraph to create an extensible and customizable workflow for handling customer interactions.

## **Method Details**
1. **Initialization**: Set up the environment and import necessary libraries.
2. **State Definition**: Create a structure to hold query information, category, sentiment and response.
3. **Node Functions**: Implement separate functions for categorization, sentiment analysis and response generation.
4. **Graph Construction**: Use StateGraph to define the workflow, adding nodes and edges to represent the support process.
5. **Conditional Routing**: Implement logic to route queries based on their category and sentiment.
6. **Workflow Compilation**: Compile the graph into an executable application.
7. **Execution**: Process customer queries through the workflow and retrieve results.
   
## **Installation Requirements**
- python 
- langchain
- langchain_core
- langchain_groq
- langchain_community
- langgraph
- gradio

## **Usage**

Once the application is running, you'll be able to interact with the customer support assistant through a simple Gradio interface. Enter any query (e.g., "I can't log into my account" or "How can I pay my bill?")

### **Example of Input and Output:**

- **Input**: "My internet is not working!"

- **Output**:
    - **Category**: 
      The customer query "My internet is not working" would be categorized as: **Technical**.
      This is because the query is related to a problem with the functionality or performance of the internet service, which requires technical assistance to resolve.
      
    - **Sentiment**: 
      **Negative**.
      
    - **Response**: 
      "I'm so sorry to hear that your internet is not working. I'm here to help you troubleshoot the issue. Can you please try a few things to help us get started?"
      
      1. **Restart your router**: Sometimes, a simple reboot can resolve the issue. Try unplugging the power cord from the back of your router, waiting for 30 seconds, and then plugging it back in.
      2. **Check your cables**: Ensure that all cables are securely connected to your router, modem, and devices.
      3. **Check your internet service provider's status**: Visit your ISP's website or social media to see if there are any outages or maintenance in your area.
      
      If none of these steps resolve the issue, please provide me with more details about your setup, such as:
      
      - Your internet service provider
      - The type of router and modem you're using
      - Any error messages you're seeing
      - The devices you're trying to connect to the internet with
      
      I'll do my best to help you troubleshoot the issue and get your internet up and running again.
