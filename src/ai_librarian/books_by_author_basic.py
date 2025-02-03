from typing import Annotated

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import InjectedToolArg, tool
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ai_librarian.utils.constants import MODEL_TEMPERATURE
from ai_librarian.utils.enums import OpenAIModel
from ai_librarian.utils.vectorstore import load_vector_store

# Initialize Rich console
console = Console()


class BooksByAuthorAgent:
    def __init__(self):
        # Define the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful librarian. Find 3 books by the author the user is interested in.",
                ),
                ("human", "{input}"),
            ]
        )

        # Define the LLM
        self.llm = ChatOpenAI(
            model=OpenAIModel.GPT_4_O_MINI, temperature=MODEL_TEMPERATURE
        )
        # Load the vector store
        self.vector_store = load_vector_store()

        # Bind the tool to the LLM
        self.tools = {"search_library_database": self.search_library_database}
        self.llm_with_tools = self.llm.bind_tools(
            list(self.tools.values()), tool_choice="any"
        )  # tool_choice="any" forces at least one tool to be used

    @staticmethod
    @tool
    def search_library_database(
        author: str,
        vector_store: Annotated[Chroma, InjectedToolArg],
    ) -> list[Document]:
        """
        Search the library database for books that match the user's request.

        Args:
            author (str): The user's request.
            vector_store (Chroma): The vector store to search. Not available to the LLM.

        Returns:
            list[Document]: The books that match the user's request.
        """
        results = vector_store.get(where={"author": author}, limit=3)
        documents = vector_store.get_by_ids(results["ids"])

        return documents

    def find_books_by_author(self, request: str) -> str:
        """
        Find books by author.

        Args:
            request (str): The user's search request.

        Returns:
            str: The books by author.
        """
        # Interpolate the request into the prompt template
        prompt = self.prompt_template.invoke({"input": request})

        try:
            # Invoke the LLM with the interpolated prompt
            response = self.llm_with_tools.invoke(prompt)

            console.print(
                response
            )  # Will be an AIMessage containing tool calls and generated args

            tool_output = []
            for tool_call in response.tool_calls:
                selected_tool = self.tools[tool_call["name"]]
                tool_call["args"]["vector_store"] = self.vector_store
                tool_output.extend(
                    selected_tool.invoke(tool_call["args"])
                )  # Actually invokes the tool with the generated args

            return str(tool_output)

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"


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
