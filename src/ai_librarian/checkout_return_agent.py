import asyncio
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


class CheckoutReturnAgent:
    def __init__(self):
        # Define the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a librian desk clerk. You are responsible for checking out and returning books to the user.",
                ),
                (
                    "system",
                    "Ensure all book titles are converted to their full title that would allow for a good search in the library database.",
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
        self.tools = {
            "checkout_book": self.checkout_book,
            "return_book": self.return_book,
        }
        self.llm_with_tools = self.llm.bind_tools(list(self.tools.values()))

    @staticmethod
    def search_book(
        vector_store: Chroma,
        isbn: str = None,
        title: str = None,
    ) -> Document | None:
        """
        Search for a book in the library.

        Args:
            vector_store (Chroma): The vector store to search.
            isbn (str): The ISBN of the book to search for.
            title (str): The title of the book to search for.

        Returns:
            Document | None: The retrieved book or None if not found.
        """
        if isbn:
            predicate = {"isbn": isbn}
        elif title:
            predicate = {"title": title}
        else:
            raise ValueError(
                "Please provide either an ISBN or a title to checkout a book."
            )

        results = vector_store.get(where=predicate)

        if results["ids"]:
            document = vector_store.get_by_ids(results["ids"])[0]
            return document
        else:
            return None

    @staticmethod
    @tool
    async def checkout_book(
        vector_store: Annotated[Chroma, InjectedToolArg],
        isbn: str = None,
        title: str = None,
    ) -> str:
        """
        Checkout a book from the library.

        Args:
            vector_store (Chroma): The vector store to search. Not available to the LLM.
            isbn (str): The ISBN of the book to checkout.
            title (str): The title of the book to checkout.

        Returns:
            str: The book that was checked out or a message if the book was not found.
        """
        book = CheckoutReturnAgent.search_book(vector_store, isbn, title)
        if book:
            if book.metadata["checked_out"]:
                return "Book already checked out"
            else:
                book.metadata["checked_out"] = True
                vector_store.update_document(book.id, book)

            return f"Book checked out: {book.metadata['title']}"
        else:
            return "Book not found"

    @staticmethod
    @tool
    async def return_book(
        vector_store: Annotated[Chroma, InjectedToolArg],
        isbn: str = None,
        title: str = None,
    ) -> str:
        """
            Return a book to the library.

        Args:
            vector_store (Chroma): The vector store to search. Not available to the LLM.
            isbn (str): The ISBN of the book to return.
            title (str): The title of the book to return.

        Returns:
            str: The book that was returned or a message if the book was not found.
        """
        book = CheckoutReturnAgent.search_book(vector_store, isbn, title)
        if book:
            if not book.metadata["checked_out"]:
                return "Book already returned"
            else:
                book.metadata["checked_out"] = False
                vector_store.update_document(book.id, book)

            return f"Book returned: {book.metadata['title']}"
        else:
            return "Book not found"

    async def handle_user_request(self, request: str) -> str:
        """
        Handle user requests for checking out or returning books.

        Args:
            request (str): The user's request for checking out or returning a book.

        Returns:
            str: The book(s) checked out or returned.
        """
        # Interpolate the request into the prompt template
        prompt = self.prompt_template.invoke({"input": request})

        try:
            # Invoke the LLM with the interpolated prompt
            response = await self.llm_with_tools.ainvoke(prompt)

            console.print(response)

            # Create tasks for concurrent execution
            tasks = [
                self.tools[tool_call["name"]].ainvoke(
                    tool_call["args"] | {"vector_store": self.vector_store}
                )
                for tool_call in response.tool_calls
            ]

            # Run all tasks concurrently
            tool_output = await asyncio.gather(*tasks)

            return str(tool_output)

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)} {type(e)}"


async def async_main():
    """
    Async main entry point for the AI Librarian application.
    """
    # Welcome message in a panel
    welcome_panel = Panel(
        "[bold cyan]👋 Hello! I'm your AI Librarian. How can I help you today?[/]\n\n"
        "[dim](Type 'quit' to exit)[/]",
        title="AI Librarian",
        border_style="cyan",
    )
    console.print(welcome_panel)

    agent = CheckoutReturnAgent()

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

        # Handle user requests for checking out or returning books
        response = await agent.handle_user_request(user_input)
        md = Markdown(response)
        console.print("\n[bold blue]AI Librarian:[/]", md)


def main():
    """
    Synchronous entry point that runs the async main function.
    """

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
