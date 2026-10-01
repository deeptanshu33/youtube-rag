from langchain_community.vectorstores import FAISS
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.vectorstores import VectorStore, VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi


class YTRag:
    _vector_store: VectorStore
    _curr_video_id: str
    _ytt_api = None
    _retriever: VectorStoreRetriever
    _llm: BaseChatModel

    def __init__(self, llm: BaseChatModel):
        self._llm = llm
        self._ytt_api = YouTubeTranscriptApi()
        self._curr_video_id = ""

    def choose_video(self, video_id: str):
        self._curr_video_id = video_id
        self.populate_store()

    def populate_store(self):
        video_transcript = self._ytt_api.fetch(self._curr_video_id)
        transcript = "".join(chunk.text for chunk in video_transcript)

        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000)
        chunks = splitter.create_documents(texts=[transcript])

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self._vector_store = FAISS.from_documents(chunks, embeddings)
        self._retriever = self._vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

    def run(self, user_prompt: str) -> str:
        if(len(self._curr_video_id) == 0):
            return "Video not set"
        
        prompt = PromptTemplate(template="""
            You are a helpful assistant. 
            You will be provided with a transcript context, answer ONLY from the provided context.
            If context provided is insufficient, repond with "Insufficient context"
                                
            {context}
                                
            Question: {question}
        """, input_variables=['context', 'question'])

        def format_docs(retrieved_docs):
            context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
            return context_text

        parallel_chain = RunnableParallel({
            'context': self._retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })

        parser = StrOutputParser()

        main_chain = parallel_chain | prompt | self._llm | parser
        response = main_chain.invoke(user_prompt)
        return response