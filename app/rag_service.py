"""RAG сервис для генерации ответов"""
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import settings
from app.retriever import RAGRetriever
from llm_service.llm_client import LLMClient


class RAGService:
    """Сервис для генерации ответов на основе найденных документов"""
    
    def __init__(self, retriever: RAGRetriever):
        self.retriever = retriever
        self._init_llm()
        self._init_prompt()
    
    def _init_llm(self):
        """Инициализация LLM"""
        client = LLMClient(provider="openai")
        self.llm = client.create_chat(
            model=settings.openai.chat_model_name,
            temperature=0.4,
        )
    
    def _init_prompt(self):
        """Инициализация промпта"""
        system_template = """Ты — AI-ассистент, специализирующийся на машинном обучении и data science.
Твоя задача — отвечать на вопросы пользователей, используя предоставленный контекст из справочника Яндекса по машинному обучению.

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленного контекста
2. Если информации недостаточно для ответа, честно скажи об этом
3. Используй математические формулы там, где это уместно (в формате LaTeX)
4. Структурируй ответ: используй списки, подзаголовки для читаемости
5. Если в контексте есть несколько релевантных частей, объедини информацию
6. Отвечай на русском языке

КОНТЕКСТ:
{context}

Теперь ответь на вопрос пользователя, основываясь на этом контексте."""

        user_template = "{question}"
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("user", user_template)
        ])
    
    def _format_context(self, documents: List[Document]) -> str:
        """Форматирование документов в контекст"""
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            filename = doc.metadata.get("filename", "unknown")
            breadcrumbs = doc.metadata.get("breadcrumbs", "")
            
            header = f"[Документ {i}]"
            if breadcrumbs:
                header += f" {breadcrumbs}"
            header += f" (источник: {filename})"
            
            context_parts.append(f"{header}\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_answer(
        self, 
        query: str, 
        top_k: int | None = None,
        temperature: float = 0.7,
        use_hyde: bool = False
    ) -> dict:
        """
        Генерация ответа на вопрос
        
        Args:
            query: Вопрос пользователя
            top_k: Количество документов для контекста
            temperature: Температура генерации
            use_hyde: Использовать HyDE для улучшения поиска
            
        Returns:
            dict с ответом и метаданными
        """
        # Получаем контекст со скорами
        context_data = self.retriever.get_context_for_query(
            query, 
            top_k=top_k, 
            with_scores=True,
            use_hyde=use_hyde
        )
        documents = context_data["documents"]
        scores = context_data.get("scores", [])
        
        if not documents:
            return {
                "query": query,
                "answer": "К сожалению, я не нашел релевантной информации в базе знаний для ответа на ваш вопрос.",
                "documents": [],
                "sources": [],
                "num_documents_used": 0,
                "total_tokens_context": 0
            }
        
        # Форматируем контекст
        context = self._format_context(documents)
        
        # Создаем chain
        chain = self.prompt | self.llm.with_config({"temperature": temperature}) | StrOutputParser()
        
        # Генерируем ответ
        answer = chain.invoke({
            "context": context,
            "question": query
        })
        
        return {
            "query": query,
            "answer": answer,
            "documents": documents,
            "scores": scores,
            "sources": context_data["sources"],
            "num_documents_used": context_data["num_documents"],
            "total_tokens_context": context_data["total_tokens"],
            "used_hyde": context_data.get("used_hyde", False)
        }

