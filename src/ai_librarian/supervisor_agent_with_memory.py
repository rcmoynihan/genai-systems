import asyncio
from enum import Enum
from typing import Annotated, Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command
from pydantic import BaseModel, Field
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ai_librarian.checkout_return_agent import CheckoutReturnAgent
from ai_librarian.recommend_books_vector_with_filters import RecommendBooksAgent
from ai_librarian.utils.constants import MODEL_TEMPERATURE
from ai_librarian.utils.enums import OpenAIModel

# Initialize Rich console
console = Console()


class State(BaseModel):
    """
    State for the librarian supervisor agent graph.

    The graph is a form of state machine that allows the agent to transition between various nodes.

    Nodes can be LLMs, tools, or other agents.
    """

    request: str | None = None
    result: str | None = None
    response: str | None = None
    messages: Annotated[list, add_messages]


class Routes(str, Enum):
    """
    Routes for the librarian supervisor agent graph.
    """

    RECOMMEND = "recommend"
    CHECKOUT_RETURN = "checkout_return"
    RESPOND = "respond"


class RouterResponse(BaseModel):
    """
    Routes for the librarian supervisor agent graph.
    """

    route: Routes = Field(
        ...,
        description=(
            "The route to take. 'recommend' if the user wants book recommendations, "
            "'checkout_return' if the user wants to checkout or return a book, "
            "'respond' if the request does not map to a sub-agent or is a general knowledge question about the books or authors."
        ),
    )
    request: str = Field(
        ...,
        description="The user's request, restated in a way that is more amenable to the route.",
    )


class LibrarianSupervisorAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=OpenAIModel.GPT_4_O_MINI, temperature=MODEL_TEMPERATURE
        )

        self.recommend_agent = RecommendBooksAgent()
        self.checkout_return_agent = CheckoutReturnAgent()

        self.builder = StateGraph(State)
        self.init_graph()

    def init_graph(self):
        self.builder.add_node(
            "router",
            self.router,
        )
        self.builder.add_edge(START, "router")

        self.builder.add_node(
            Routes.RECOMMEND.value,
            self.recommend_books,
        )

        self.builder.add_node(
            Routes.CHECKOUT_RETURN.value,
            self.checkout_return,
        )

        self.builder.add_node(
            "respond",
            self.respond,
        )

        self.builder.add_edge(Routes.RECOMMEND.value, "respond")
        self.builder.add_edge(Routes.CHECKOUT_RETURN.value, "respond")
        self.builder.add_edge("respond", END)

        checkpointer = MemorySaver()
        self.graph = self.builder.compile(checkpointer=checkpointer)
        with open("graph.png", "wb") as f:
            f.write(self.graph.get_graph().draw_mermaid_png())

    async def router(
        self, state: State
    ) -> Command[Literal["recommend", "checkout_return", "respond"]]:
        """
        Router for the librarian supervisor agent graph.
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are a librarian, and have access to various agents that can help you with the user's request. "
                        "You are responsible for routing the user's request to the appropriate agent "
                        "and then returning the result of the agent's response. "
                        "You have the following messages in your memory: {messages}"
                    ),
                ),
                ("human", "{input}"),
            ]
        )

        prompt = prompt_template.invoke(
            {"input": state.request, "messages": state.messages}
        )

        llm_with_structured_output = self.llm.with_structured_output(RouterResponse)
        response = llm_with_structured_output.invoke(prompt)

        console.print(f"[bold red]Routing to {response.route}[/]")

        return Command(
            update={"request": response.request, "messages": response.request},
            goto=response.route,
        )

    def recommend_books(self, state: State) -> dict:
        """
        Recommend books on a topic.

        Args:
            state (State): The state of the agent.

        Returns:
            dict: The result of the agent's response.
        """
        result = self.recommend_agent.get_book_recommendations(state.request)
        return {"result": result}

    async def checkout_return(self, state: State) -> dict:
        """
        Checkout or return books.

        Args:
            state (State): The state of the agent.

        Returns:
            dict: The result of the agent's response.
        """
        result = await self.checkout_return_agent.handle_user_request(state.request)
        return {"result": result}

    def respond(self, state: State) -> dict:
        """
        Respond to the user.

        Args:
            state (State): The state of the agent.

        Returns:
            dict: The result of the agent's response.
        """
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are the final thought in a librarian agent. Inform the user of the results of the agent's response. "
                        "If the response is a list of books, format them as a markdown list. "
                        "Comment on the status of the books (checked out, available, etc.) "
                        "If the user's request is a general knowledge question about the books or authors, answer the question directly and concisely. "
                        "The user's request is: {request} "
                        "You have the following messages in your memory: {messages}"
                    ),
                ),
                ("human", "Result: {input}"),
            ]
        )
        prompt = prompt_template.invoke(
            {
                "input": state.result,
                "request": state.request,
                "messages": state.messages,
            }
        )
        response = self.llm.invoke(prompt)
        return {"response": response.content, "messages": response.content}


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

    librarian = LibrarianSupervisorAgent()

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
        config = {"configurable": {"thread_id": "1"}}
        response = await librarian.graph.ainvoke({"request": user_input}, config=config)
        md = Markdown(response["response"])
        console.print("\n[bold blue]AI Librarian:[/]", md)


def main():
    """
    Synchronous entry point that runs the async main function.
    """

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
