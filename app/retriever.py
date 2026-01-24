"""RAG Retriever с поддержкой Qdrant и Redis"""
import tiktoken
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_core.load import loads
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_community.storage import RedisStore
from qdrant_client import QdrantClient

from app.config import settings
from llm_service.llm_client import LLMClient


class RAGRetriever:
    """Retriever для поиска документов с использованием Parent-Child стратегии"""
    
    def __init__(self):
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Инициализация компонентов
        self._init_embeddings()
        self._init_qdrant()
        self._init_redis()
        self._init_hyde_llm()
    
    def _init_embeddings(self):
        """Инициализация embedding моделей"""
        client = LLMClient(provider="openai")
        self.dense_embeddings = client.create_embeddings(
            model=settings.openai.embedding_model_name
        )
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    def _init_qdrant(self):
        """Инициализация Qdrant клиента и vector store"""
        self.qdrant_client = QdrantClient(url=settings.qdrant.url)
        
        # Проверяем существование коллекции
        collections = self.qdrant_client.get_collections().collections
        collection_exists = any(
            c.name == settings.qdrant.collection_name for c in collections
        )
        
        if not collection_exists:
            raise ValueError(
                f"Коллекция '{settings.qdrant.collection_name}' не найдена в Qdrant. "
                "Сначала запустите ETL процесс для загрузки данных."
            )
        
        self.vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=settings.qdrant.collection_name,
            embedding=self.dense_embeddings,
            sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",  # Указываем имя dense вектора
            sparse_vector_name="sparse",  # Указываем имя sparse вектора
        )
    
    def _init_redis(self):
        """Инициализация Redis store для parent chunks"""
        self.parent_store = RedisStore(
            redis_url=settings.redis.url,
            namespace="rag:parents"
        )
    
    def _init_hyde_llm(self):
        """Инициализация LLM для HyDE"""
        client = LLMClient(provider="openai")
        self.hyde_llm = client.create_chat(
            model=settings.openai.chat_model_name,
            temperature=0.7,  # Выше температура для разнообразия
        )
        
        # Промпт для генерации гипотетического документа
        hyde_template = """Ты — эксперт по машинному обучению и data science.

Пользователь задал вопрос: {question}

Напиши короткий, информативный параграф (2-3 предложения), который мог бы быть ответом на этот вопрос из учебника по машинному обучению. 
Пиши так, как будто это фрагмент из справочника Яндекса по ML.
Используй технические термины и формулировки.
НЕ пиши "Ответ:", просто напиши сам текст.

Гипотетический фрагмент документа:"""

        self.hyde_prompt = ChatPromptTemplate.from_template(hyde_template)
        self.hyde_chain = self.hyde_prompt | self.hyde_llm | StrOutputParser()
    
    def _generate_hypothetical_document(self, query: str) -> str:
        """Генерирует гипотетический документ для запроса"""
        try:
            hypothetical_doc = self.hyde_chain.invoke({"question": query})
            print(f"🔮 HyDE документ: {hypothetical_doc[:100]}...")
            return hypothetical_doc
        except Exception as e:
            print(f"⚠️  Ошибка генерации HyDE документа: {e}")
            return query  # Fallback на оригинальный запрос
    
    def _tiktoken_len(self, text: str) -> int:
        """Подсчет токенов в тексте"""
        return len(self.encoding.encode(text))
    
    def _load_parent_chunk(self, parent_id: str) -> Document | None:
        """Загрузить parent chunk из Redis"""
        try:
            result = self.parent_store.mget([parent_id])
            if result and result[0]:
                return loads(result[0].decode("utf-8"))
        except Exception as e:
            print(f"Ошибка загрузки parent chunk {parent_id}: {e}")
        return None
    
    def get_parent_docs_count(self) -> int:
        """Получить количество parent документов в Redis"""
        try:
            count = 0
            # yield_keys возвращает итератор по ключам с префиксом
            for _ in self.parent_store.yield_keys():
                count += 1
            return count
        except Exception as e:
            print(f"Ошибка подсчета ключей Redis: {e}")
            return 0
    
    def search(
        self,
        query: str,
        top_k: int | None = None,
        max_tokens: int | None = None
    ) -> List[Document]:
        """
        Поиск релевантных документов
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов (по умолчанию из настроек)
            max_tokens: Максимальное количество токенов в контексте
            
        Returns:
            Список документов (parent chunks)
        """
        if top_k is None:
            top_k = settings.rag.retrieval_top_k
        if max_tokens is None:
            max_tokens = settings.rag.max_context_tokens
        
        # Поиск child chunks в Qdrant
        child_chunks = self.vector_store.similarity_search(query, k=top_k)
        
        # Извлечение parent chunks из Redis
        parent_ids = []
        for child in child_chunks:
            parent_id = child.metadata.get("parent_id")
            if parent_id and parent_id not in parent_ids:
                parent_ids.append(parent_id)
        
        # Загрузка parent chunks
        parent_chunks = []
        total_tokens = 0
        
        for parent_id in parent_ids:
            parent = self._load_parent_chunk(parent_id)
            if parent:
                chunk_tokens = self._tiktoken_len(parent.page_content)
                
                # Проверяем лимит токенов
                if total_tokens + chunk_tokens > max_tokens:
                    break
                
                parent_chunks.append(parent)
                total_tokens += chunk_tokens
        
        return parent_chunks
    
    def search_with_scores(
        self,
        query: str,
        top_k: int | None = None,
        max_tokens: int | None = None
    ) -> List[Tuple[Document, float]]:
        """
        Поиск релевантных документов со скорами
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов (по умолчанию из настроек)
            max_tokens: Максимальное количество токенов в контексте
            
        Returns:
            Список кортежей (документ, скор)
        """
        if top_k is None:
            top_k = settings.rag.retrieval_top_k
        if max_tokens is None:
            max_tokens = settings.rag.max_context_tokens
        
        # Поиск child chunks в Qdrant со скорами
        child_chunks_with_scores = self.vector_store.similarity_search_with_score(query, k=top_k)
        
        # Создаем словарь parent_id -> лучший скор
        parent_scores = {}
        parent_ids_order = []
        
        for child, score in child_chunks_with_scores:
            parent_id = child.metadata.get("parent_id")
            if parent_id:
                # Сохраняем лучший (минимальный для distance) скор для каждого parent
                if parent_id not in parent_scores:
                    parent_scores[parent_id] = score
                    parent_ids_order.append(parent_id)
                else:
                    parent_scores[parent_id] = min(parent_scores[parent_id], score)
        
        # Загрузка parent chunks со скорами
        parent_chunks_with_scores = []
        total_tokens = 0
        
        for parent_id in parent_ids_order:
            parent = self._load_parent_chunk(parent_id)
            if parent:
                chunk_tokens = self._tiktoken_len(parent.page_content)
                
                # Проверяем лимит токенов
                if total_tokens + chunk_tokens > max_tokens:
                    break
                
                parent_chunks_with_scores.append((parent, parent_scores[parent_id]))
                total_tokens += chunk_tokens
        
        return parent_chunks_with_scores
    
    def search_with_hyde(
        self,
        query: str,
        top_k: int | None = None,
        max_tokens: int | None = None,
        score_threshold: float | None = None
    ) -> List[Tuple[Document, float]]:
        """
        Поиск с использованием HyDE
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            max_tokens: Максимальное количество токенов
            score_threshold: Порог скора для фильтрации результатов
            
        Returns:
            Список кортежей (документ, скор)
        """
        if top_k is None:
            top_k = settings.rag.retrieval_top_k
        if max_tokens is None:
            max_tokens = settings.rag.max_context_tokens
        if score_threshold is None:
            score_threshold = settings.rag.score_threshold
        
        # Генерируем гипотетический документ
        search_query = self._generate_hypothetical_document(query)
        
        # Поиск child chunks в Qdrant со скорами
        child_chunks_with_scores = self.vector_store.similarity_search_with_score(
            search_query, k=top_k * 2  # Берём больше для фильтрации
        )
        
        # Создаем словарь parent_id -> лучший скор
        parent_scores = {}
        parent_ids_order = []
        
        for child, score in child_chunks_with_scores:
            # Фильтруем по порогу скора
            if score > score_threshold:
                continue
                
            parent_id = child.metadata.get("parent_id")
            if parent_id:
                if parent_id not in parent_scores:
                    parent_scores[parent_id] = score
                    parent_ids_order.append(parent_id)
                else:
                    parent_scores[parent_id] = min(parent_scores[parent_id], score)
        
        # Ограничиваем до top_k
        parent_ids_order = parent_ids_order[:top_k]
        
        # Загрузка parent chunks со скорами
        parent_chunks_with_scores = []
        total_tokens = 0
        
        for parent_id in parent_ids_order:
            parent = self._load_parent_chunk(parent_id)
            if parent:
                chunk_tokens = self._tiktoken_len(parent.page_content)
                
                if total_tokens + chunk_tokens > max_tokens:
                    break
                
                parent_chunks_with_scores.append((parent, parent_scores[parent_id]))
                total_tokens += chunk_tokens
        
        return parent_chunks_with_scores
    
    def get_context_for_query(self, query: str, top_k: int | None = None, with_scores: bool = False, use_hyde: bool = False) -> dict:
        """
        Получить контекст для запроса с метаданными
        
        Args:
            query: Поисковый запрос
            top_k: Количество результатов
            with_scores: Включить скоры в результат
            use_hyde: Использовать HyDE для улучшения поиска
        
        Returns:
            dict с полями:
                - query: исходный запрос
                - documents: список найденных документов
                - scores: список скоров (если with_scores=True)
                - total_tokens: общее количество токенов
                - sources: список источников
                - used_hyde: был ли использован HyDE
        """
        if use_hyde:
            # Используем HyDE поиск (всегда со скорами)
            documents_with_scores = self.search_with_hyde(query, top_k=top_k)
            documents = [doc for doc, _ in documents_with_scores]
            scores = [score for _, score in documents_with_scores] if with_scores else None
        elif with_scores:
            documents_with_scores = self.search_with_scores(query, top_k=top_k)
            documents = [doc for doc, _ in documents_with_scores]
            scores = [score for _, score in documents_with_scores]
        else:
            documents = self.search(query, top_k=top_k)
            scores = None
        
        total_tokens = sum(self._tiktoken_len(doc.page_content) for doc in documents)
        
        # Извлечение уникальных источников
        sources = []
        seen_files = set()
        for doc in documents:
            filename = doc.metadata.get("filename", "unknown")
            if filename not in seen_files:
                sources.append({
                    "filename": filename,
                    "breadcrumbs": doc.metadata.get("breadcrumbs", ""),
                    "url": doc.metadata.get("url", "")
                })
                seen_files.add(filename)
        
        result = {
            "query": query,
            "documents": documents,
            "total_tokens": total_tokens,
            "sources": sources,
            "num_documents": len(documents),
            "used_hyde": use_hyde
        }
        
        if with_scores:
            result["scores"] = scores
        
        return result

