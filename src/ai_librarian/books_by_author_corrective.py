from typing import Annotated, Literal

from langchain_chroma import Chroma
from langchain_community.tools import TavilySearchResults
from langchain_core.documents import Document
from langchain_core.messages import AIMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolArg, tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ai_librarian.utils.constants import MODEL_TEMPERATURE
from ai_librarian.utils.enums import OpenAIModel
from ai_librarian.utils.vectorstore import load_vector_store

# Initialize Rich console
console = Console()


class State(BaseModel):
    """
    State for the author book search agent.

    search_results will store the agent's research as it progresses.
    """

    query: str
    search_results: str = ""
    result: str | None = None
    tool_calls: list[ToolCall] = []
    pass_count: int = 0


class BooksByAuthorAgent:
    def __init__(self):
        # Define the LLM
        self.llm = ChatOpenAI(
            model=OpenAIModel.GPT_4_O_MINI, temperature=MODEL_TEMPERATURE
        )
        # Load the vector store
        self.vector_store = load_vector_store()

        # Bind the tool to the LLM
        self.primary_tools = {"search_library_database": self.search_library_database}
        self.secondary_tools = {"tavily_search": self.tavily_search}
        self.all_tools = {**self.primary_tools, **self.secondary_tools}
        self.llm_with_primary_tools = self.llm.bind_tools(
            list(self.primary_tools.values())
        )
        self.llm_with_secondary_tools = self.llm.bind_tools(
            list(self.secondary_tools.values())
        )

        self.builder = StateGraph(State)
        self.init_graph()

    def init_graph(self):
        self.builder.add_node(
            "generate_search_calls",
            self.generate_search_calls,
        )
        self.builder.add_node(
            "invoke_tool",
            self.invoke_tool,
        )
        self.builder.add_edge(START, "generate_search_calls")
        self.builder.add_edge("invoke_tool", "generate_search_calls")
        self.builder.add_edge("generate_search_calls", END)

        self.graph = self.builder.compile()

        with open("graph.png", "wb") as f:
            f.write(self.graph.get_graph().draw_mermaid_png())

    def generate_search_calls(
        self, state: State
    ) -> Command[Literal["invoke_tool", "__end__"]] | dict:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are an agent designed to find books by an author. Use the tools provided to find the books. "
                        "You may already have valid search results. If you feel you have satisfied the user's request, "
                        "you can opt to call no tools and instead reply with the user ready answer. "
                        "Otherwise, return a list of tool calls to continue searching. "
                        "Not all books are available in the library database. Ensure your final answer indicates "
                        "which, if any, books are available in the library database. "
                        "It is okay to show the user unavailable books, just be sure to indicate that they are unavailable. "
                        "Current search results: {search_results}"
                    ),
                ),
                ("human", "Query: {input}"),
            ]
        )

        prompt = prompt_template.invoke(
            {"input": state.query, "search_results": state.search_results}
        )

        console.print(f"Search pass: {state.pass_count}")
        response: AIMessage
        if state.pass_count == 0:
            response = self.llm_with_primary_tools.invoke(prompt)
        elif state.pass_count == 1:
            response = self.llm_with_secondary_tools.invoke(prompt)
        else:
            response = self.llm.invoke(prompt)

        console.print(response)

        if response.tool_calls and state.pass_count < 2:
            return Command(
                update={"tool_calls": response.tool_calls},
                goto="invoke_tool",
            )
        else:
            return {"result": response.content}

    def invoke_tool(self, state: State) -> dict:
        """
        Invoke the tool calls and return the results.

        Args:
            state (State): The state of the agent.

        Returns:
            dict: The results of the tool calls.
        """

        console.print(state.tool_calls)

        output = ""

        tool_output = []
        for tool_call in state.tool_calls:
            selected_tool = self.all_tools[tool_call["name"]]
            if tool_call["name"] in self.primary_tools:
                output += "Books available in the library database:\n"
                tool_call["args"]["vector_store"] = self.vector_store
            else:
                output += "Books currently unavailable in the library database:\n"

            tool_output.extend(
                selected_tool.invoke(tool_call["args"])
            )  # Actually invokes the tool with the generated args

        output += "\n".join([str(item) for item in tool_output])

        return {
            "search_results": state.search_results + "\n" + str(tool_output),
            "pass_count": state.pass_count + 1,
            "tool_calls": [],  # Reset the tool calls
        }

    @staticmethod
    @tool
    def search_library_database(
        author: str,
        vector_store: Annotated[Chroma, InjectedToolArg],
    ) -> list[Document]:
        """
        Search the library database for books that match the user's request.

        Args:
            author (str): The author to search for. Ensure proper spelling and capitalization.
            vector_store (Chroma): The vector store to search. Not available to the LLM.

        Returns:
            list[Document]: The books that match the user's request.
        """
        results = vector_store.get(where={"author": author}, limit=3)
        if results["ids"]:
            documents = vector_store.get_by_ids(results["ids"])
        else:
            documents = []

        return documents

    @staticmethod
    @tool
    def tavily_search(
        web_query: str,
    ) -> list[dict]:
        """
        Search the web for books by the author the user is interested in.

        Args:
            web_query (str): The user's search request. Should be a query for a book by the author.

        Returns:
            list[dict]: Search results from Tavily.
        """
        tavily = TavilySearchResults(max_results=10)

        results = tavily.invoke({"query": web_query})

        return results

    def find_books_by_author(self, request: str) -> str:
        """
        Find books by author.

        Args:
            request (str): The user's search request.

        Returns:
            str: The books by author.
        """
        response = self.graph.invoke({"query": request})

        return response["result"]


def main():
    # Welcome message in a panel
    welcome_panel = Panel(
        "[bold cyan]👋 Hello! I'm your AI Librarian. What author are you looking for today?[/]\n\n"
        "[dim](Type 'quit' to exit)[/]",
        title="AI Librarian",
        border_style="cyan",
    )
    console.print(welcome_panel)

    agent = BooksByAuthorAgent()

    while True:
        # User input with prompt styling
        user_input = console.input("\n[bold green]You:[/] ").strip()

        if user_input.lower() in ["quit", "exit", "bye"]:
            # Goodbye message in a panel
            farewell_panel = Panel(
                "[bold cyan]Thanks for chatting! Happy reading! 📚[/]",
                border_style="cyan",
            )
            console.print("\n", farewell_panel)
            break

        # Get and display recommendations with markdown formatting
        response = agent.find_books_by_author(user_input)
        md = Markdown(response)
        console.print("\n[bold blue]AI Librarian:[/]", md)


if __name__ == "__main__":
    main()
