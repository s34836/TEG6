import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import functools
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph
from typing import Annotated, Literal, TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.arxiv import ArxivQueryRun
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from langchain_core.runnables.graph import MermaidDrawMethod
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Getting API key
dotenv_path = os.path.join('../', 'config', '.env')
load_dotenv(dotenv_path)
api_key = os.getenv('OPENAI_API_KEY_TEG')
tavily_key = os.getenv('TAVILY_API_KEY')
if not api_key:
    raise ValueError("OpenAI API key not found. Please set either OPENAI_API_KEY or OPENAI_API_KEY_TEG in your .env file.")
os.environ["OPENAI_API_KEY_TEG_06"] = api_key
os.environ["TAVILY_API_KEY"] = tavily_key

load_dotenv()
llm = ChatOpenAI(
    model_name="gpt-4.1-mini-2025-04-14",
    openai_api_key=os.environ['OPENAI_API_KEY_TEG_06'],
    temperature=0,
    request_timeout=120,
    max_retries=3
)


tools = [
    TavilySearchResults(max_results=5),
    WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper()),
    ArxivQueryRun(api_wrapper=ArxivAPIWrapper())
]

search_template = """Your job is to search multiple sources for related information that would be relevant to generate the article described by the user.

Use the following tools:
1. Tavily Search - for general web search and news
2. Wikipedia - for background information and definitions
3. Arxiv - for academic papers and research

NOTE: Do not write the article. Just search the sources for related information if needed and then forward that information to the outliner node.
Try to use a combination of these sources to get a comprehensive view of the topic.
"""

outliner_template = """Your job is to take as input information from multiple sources (web search, Wikipedia, and academic papers) along with users instruction on what article they want to write and generate a comprehensive outline
                       for the article. Make sure to incorporate insights from all available sources to create a well-rounded outline.
                    """

writer_template = """Your job is to write an article following this EXACT format:

TITLE: [Your article title here]

BODY: [Your article content here]

IMPORTANT RULES:
1. You MUST use the exact format above with "TITLE:" and "BODY:" as shown
2. Do not add any additional formatting or markdown
3. Do not copy the outline directly - write a proper article that follows the outline's structure
4. Keep the content professional and well-structured
5. Ensure there is exactly one blank line between TITLE and BODY sections
"""


# Define method for creating agents and binding them to the tools
def create_agent(llm, tools, system_message: str):
    """Create an agent."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "{system_message}",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    prompt = prompt.partial(system_message=system_message)
    
    if tools:
      return prompt | llm.bind_tools(tools)
    else:
      return prompt | llm

# Create all three agent roles
search_agent = create_agent(llm, tools, search_template)
outliner_agent = create_agent(llm, [], outliner_template)
writer_agent = create_agent(llm, [], writer_template)


# Create nodes with each agent
def agent_node(state, agent, name):
    logger.info(f"\n{'='*50}")
    logger.info(f"Starting {name}")
    logger.info(f"{'='*50}")
    
    result = agent.invoke(state)
    
    # Special handling for writer agent to ensure proper formatting
    if name == "Writer Agent":
        content = result.content
        # Ensure the output starts with TITLE: and BODY:
        if not content.strip().startswith("TITLE:"):
            content = "TITLE: " + content
        if "BODY:" not in content:
            content = content.replace("TITLE:", "TITLE:\n\nBODY:", 1)
    
    logger.info(f"Completed {name}")
    logger.info(f"{'='*50}\n")
    
    return {
        'messages': [result]
    }

search_node = functools.partial(agent_node, agent=search_agent, name="Search Agent")
outliner_node = functools.partial(agent_node, agent=outliner_agent, name="Outliner Agent")
writer_node = functools.partial(agent_node, agent=writer_agent, name="Writer Agent")

# Define the conditional edge 

def should_search(state) -> Literal["tools", "outliner"]:
    messages = state['messages']
    last_message = messages[-1]
    
    if hasattr(last_message, 'additional_kwargs') and 'tool_calls' in last_message.additional_kwargs:
        logger.info("Routing to tools node for processing search results")
        return "tools"
    
    logger.info("Routing to outliner node to generate outline")
    return "outliner"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

workflow = StateGraph(AgentState)

# Add nodes
tool_node = ToolNode(tools)

workflow.add_node("search", search_node)
workflow.add_node("tools", tool_node)
workflow.add_node("outliner", outliner_node)
workflow.add_node("writer", writer_node)

# Add edges
workflow.set_entry_point("search")
workflow.add_conditional_edges(
    "search",
    should_search
)
workflow.add_edge("tools", "search")
workflow.add_edge("outliner", "writer")
workflow.add_edge("writer", END)

# Compile the graph
graph = workflow.compile()

# Save the graph visualization as a PNG file
graph_png = graph.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
with open("workflow_graph.png", "wb") as f:
    f.write(graph_png)
print("Graph visualization saved as 'workflow_graph.png'")

question = "What is the business potential of using generative technologies in investment banking? Propose usecases and business justficiation for them."

logger.info("\nStarting workflow execution")
logger.info(f"Question: {question}")

input_message = HumanMessage(content=question)

try:
    logger.info("Initializing graph stream")
    for event in graph.stream({"messages": [input_message]}, stream_mode="values"):
        try:
            if 'messages' in event and event['messages']:
                logger.info("Processing new message in stream")
                event['messages'][-1].pretty_print()
                print("\n\n")
            else:
                logger.warning("Received empty or invalid message")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
except Exception as e:
    logger.error(f"Stream error occurred: {str(e)}")
    if "timeout" in str(e).lower():
        logger.error("The operation timed out. This might be due to slow API responses or network issues.")

logger.info("Workflow execution completed")
