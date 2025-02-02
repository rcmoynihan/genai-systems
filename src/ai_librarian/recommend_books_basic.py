from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from ai_librarian.utils.constants import MODEL_TEMPERATURE
from ai_librarian.utils.enums import OpenAIModel

# Initialize Rich console
console = Console()


class RecommendBooksAgent:
    def __init__(self):
        # Define the prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful librarian. Recommend 3 books on the topic the user is interested in.",
                ),
                ("human", "{input}"),
            ]
        )

        # Define the LLM
        self.llm = ChatOpenAI(
            model=OpenAIModel.GPT_4_O_MINI, temperature=MODEL_TEMPERATURE
        )

    def get_book_recommendations(self, request: str) -> str:
        """
        Get book recommendations based on user's request.

        Args:
            request (str): The user's recommendation request.

        Returns:
            str: The book recommendations.
        """
        # Interpolate the request into the prompt template
        prompt = self.prompt_template.invoke({"input": request})

        try:
            # Invoke the LLM with the interpolated prompt
            response = self.llm.invoke(prompt)
            return response.content

        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"


def main():
    # Welcome message in a panel
    welcome_panel = Panel(
        "[bold cyan]👋 Hello! I'm your AI Librarian. What kind of books are you looking for today?[/]\n\n"
        "[dim](Type 'quit' to exit)[/]",
        title="AI Librarian",
        border_style="cyan",
    )
    console.print(welcome_panel)

    agent = RecommendBooksAgent()

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
        response = agent.get_book_recommendations(user_input)
        md = Markdown(response)
        console.print("\n[bold blue]AI Librarian:[/]", md)


if __name__ == "__main__":
    main()
