#Introduction to Langchain
import os
from dataclasses import Field

import export
from groq import BaseModel
from langchain.chains import llm
from langchain_core.output_parsers import PydanticOutputParser

#class PromptTemplate:
#    pass
#prompt_template = PromptTemplate.from_template(
#    "List {n} cooking/meal titles for {cuisine} cuisine."
#)
#class SequentialChain:
#    pass
#complex_chain = SequentialChain(
#    chains=[chain1, chain2],
#    input_variables=["genre"],
#    output_variables=["synopsis", "titles"],
#    verbose=True,
#)
#output = complex_chain({"genre": "comedy"})
#print(f"Output: {output}")

#Setting Up Langchain and Groq
#export GROQ_API_KEY='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'
os.environ["GROQ_API_KEY"] = "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"

#MODEL I/O
#Generate Predictions
#from langchain_groq import ChatGroq
# Initialize the Groq chat model
# We're using LLaMA 3 70B model, which is one of the latest available on Groq
#llm = ChatGroq(
#    model="llama3-70b-8192",
#    temperature=0.3,
#    max_tokens=500,
#)
# Generate the response using LangChain's invoke method
#response = llm.invoke("What are the 7 wonders of the world?")
#print(f"Response: {response.content}")
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
# Initialize the Groq chat model
#(""
 #chat_model = ChatGroq()
#      model="llama3-70b-8192",
#      temperature=0.7,  # Slightly higher temperature for more creative responses
#      max_tokens=500,
#)
# Define the system message for pirate personality with emojis
#system_message = SystemMessage(
#      content="You are a friendly pirate who loves to share knowledge. Always respond in pirate speech, use pirate slang, and include plenty of nautical references. Add relevant emojis throughout your responses to make them more engaging. Arr! ☠️🏴‍☠️"
#)
# Define the question
#question = "What are the 7 wonders of the world?"
# Create messages with the system instruction and question
#messages = [
#      system_message,
#      HumanMessage(content=question)
#]
# Get the response
#response = chat_model.invoke(messages)
# Print the response
#print("\nQuestion:", question)
#print("\nPirate Response:")
#print(response.content)


#Prompt Templates
#from langchain.prompts import PromptTemplate
# Create a prompt template for generating meal titles
#prompt_template = PromptTemplate.from_template(
#    "List {n} cooking/meal titles for {cuisine} cuisine (name only)."
#)
#prompt = prompt_template.format(n=3, cuisine="italian")
#repsonse = llm.invoke(prompt)

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Initialize the Groq chat model
#llm = ChatGroq(
#    model="llama3-70b-8192",
#    temperature=0.3,
#    max_tokens=500,
#)

# Create a prompt template for generating meal titles
#prompt_template = PromptTemplate.from_template(
#    "List {n} cooking/meal titles for {cuisine} cuisine (name only)."
#)

# Create a runnable chain using the pipe operator
#chain = prompt_template | llm

# Run the chain with specific parameters
#response = chain.invoke({
#    "n": 5,
#    "cuisine": "Italian"
#})

# Print the response
#print("\nPrompt: List 5 cooking/meal titles for Italian cuisine (name only).")
#print("\nResponse:")
#print(response.content)


#Getting Structured Output
#title: str
#genre: list[str]
#year: int
# The description helps the LLM to know what it should put in there.
#class Movie(BaseModel):
#    title: str = Field(description="The title of the movie.")
#    genre: list[str] = Field(description="The genre of the movie.")
 #   year: int = Field(description="The year the movie was released.")

#parser = PydanticOutputParser(pydantic_object=Movie)

#prompt_template_text = """
#Response with a movie recommendation based on the query:\n
#{format_instructions}\n
#{query}
#"""

#format_instructions = parser.get_format_instructions()
#prompt_template = PromptTemplate(
#    template=prompt_template_text,
#    input_variables=["query"],
#    partial_variables={"format_instructions": format_instructions},
#)
#prompt = prompt_template.format(query="A 90s movie with Nicolas Cage.")
#text_output = llm.invoke(prompt)
#print(text_output.content)  # printed in JSON format
#parsed_output = parser.parse(text_output.content)
#print(parsed_output)    # title='Con Air' genre=['Action', 'Thriller'] year=1997
# Using LangChain Expression Language (LCEL)
#chain = prompt_template | llm | parser
#response = chain.invoke({"query": "A 90s movie with Nicolas Cage."})
#print(response)


#Building an AI Agent
# pip install langchain langchain_community langchain_groq duckduckgo-search

from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.chains import LLMMathChain
from langchain.prompts import PromptTemplate
from langchain.agents import AgentExecutor, Tool
from langchain.agents.structured_chat.base import StructuredChatAgent

# Initialize the Groq chat model
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0.3,
    max_tokens=1024,
)

# Custom prompt for LLMMathChain
math_prompt = PromptTemplate.from_template(
    """You are a calculator. Return ONLY the numeric result of this calculation:
{question}

Return ONLY the number with no additional text or formatting."""
)

# Set up the math chain
llm_math_chain = LLMMathChain.from_llm(llm=llm, prompt=math_prompt, verbose=False)

# Improved calculator tool function
def calculate_expression(expression):
    try:
        # Remove any non-math characters that might cause issues
        clean_expr = ''.join(c for c in expression if c.isdigit() or c in '+-*/.() ')
        result = llm_math_chain.invoke({"question": clean_expr})
        return str(result["answer"])
    except Exception as e:
        print(f"Calculation error: {e}")
        # Fallback to direct evaluation if LLM fails
        try:
            return str(numexpr.evaluate(clean_expr))
        except:
            return "Could not calculate the result"

# Initialize tools
search = DuckDuckGoSearchRun()

calculator = Tool(
    name="calculator",
    description="Use this tool for arithmetic calculations. Input should be a mathematical expression.",
    func=calculate_expression,
)

# List of tools for the agent
tools = [
    Tool(
        name="search",
        description="Search the internet for information about current events, data, or facts.",
        func=search.run
    ),
    calculator
]

# Create and run the agent
agent = StructuredChatAgent.from_llm_and_tools(llm=llm, tools=tools)
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True  # Set to False to suppress detailed execution logs
)

# Run the agent
try:
    result = agent_executor.invoke({"input": "What is the population difference between Tunisia and Algeria?"})
    print("\nFinal Answer:", result["output"])
except Exception as e:
    print(f"An error occurred: {e}")
