from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel

from ai_librarian.utils.constants import VECTOR_STORE_PATH


class Book(BaseModel):
    isbn: str
    title: str
    author: str
    description: str
    year: int
    checked_out: bool = False


BOOKS = [
    Book(
        isbn="1234567890",
        title="The Great Gatsby",
        author="F. Scott Fitzgerald",
        description="A classic American novel about a man who falls in love with a woman who is not his wife.",
        year=1925,
        checked_out=False,
    ),
    Book(
        isbn="0451524934",
        title="1984",
        author="George Orwell",
        description="A dystopian novel set in a totalitarian society where critical thinking is suppressed and surveillance is omnipresent.",
        year=1949,
        checked_out=False,
    ),
    Book(
        isbn="0061120084",
        title="To Kill a Mockingbird",
        author="Harper Lee",
        description="A powerful story of racial injustice and loss of innocence in the American South, told through the eyes of young Scout Finch.",
        year=1960,
        checked_out=False,
    ),
    Book(
        isbn="0743273567",
        title="Pride and Prejudice",
        author="Jane Austen",
        description="A witty romance exploring themes of marriage, social class, and reputation in early 19th-century England.",
        year=1813,
        checked_out=False,
    ),
    Book(
        isbn="0316769177",
        title="The Catcher in the Rye",
        author="J.D. Salinger",
        description="A coming-of-age novel following teenager Holden Caulfield's journey through New York City while grappling with alienation and identity.",
        year=1951,
        checked_out=False,
    ),
    Book(
        isbn="0679783261",
        title="One Hundred Years of Solitude",
        author="Gabriel García Márquez",
        description="A masterpiece of magical realism following seven generations of the Buendía family in the mythical town of Macondo, weaving together reality, dreams, and the supernatural.",
        year=1967,
        checked_out=False,
    ),
    Book(
        isbn="0452284244",
        title="The Handmaid's Tale",
        author="Margaret Atwood",
        description="A haunting dystopian novel about a woman's struggle for survival in a fundamentalist theocratic state where women are stripped of their rights and autonomy.",
        year=1985,
        checked_out=False,
    ),
    Book(
        isbn="0679720200",
        title="The Stranger",
        author="Albert Camus",
        description="A philosophical novel about an ordinary man who becomes entangled in a senseless murder on an Algerian beach, exploring themes of absurdism and alienation.",
        year=1942,
        checked_out=False,
    ),
    Book(
        isbn="0679732241",
        title="Beloved",
        author="Toni Morrison",
        description="A powerful and haunting story of a former slave confronting the trauma of her past, exploring themes of memory, family, and the lasting impact of slavery.",
        year=1987,
        checked_out=False,
    ),
    Book(
        isbn="0140283331",
        title="Things Fall Apart",
        author="Chinua Achebe",
        description="A groundbreaking novel about the life of Okonkwo, a leader and wrestling champion in a fictional Nigerian village, depicting the collapse of traditional culture under British colonialism.",
        year=1958,
        checked_out=False,
    ),
]


def create_vector_store() -> None:
    """
    Create a vector store and save it to the specified path.
    """

    vector_store = Chroma(
        collection_name="books",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=str(VECTOR_STORE_PATH),
    )

    documents = [
        Document(
            page_content=book.description,
            metadata=book.model_dump(exclude={"description"}),
        )
        for book in BOOKS
    ]
    vector_store.add_documents(documents)


def load_vector_store() -> Chroma:
    """
    Load a vector store from the specified path.

    Returns:
        InMemoryVectorStore: The vector store.
    """
    vector_store = Chroma(
        collection_name="books",
        embedding_function=OpenAIEmbeddings(),
        persist_directory=str(VECTOR_STORE_PATH),
    )
    return vector_store


if __name__ == "__main__":
    create_vector_store()
