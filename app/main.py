"""FastAPI приложение для RAG сервиса"""
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import tiktoken

from app.config import settings
from app.retriever import RAGRetriever
from app.rag_service import RAGService
from app.schemas import (
    SearchRequest, SearchResponse, DocumentResponse, DocumentMetadata,
    RAGRequest, RAGResponse, HealthResponse
)


# Глобальные объекты (инициализируются при старте)
retriever: RAGRetriever | None = None
rag_service: RAGService | None = None
encoding = tiktoken.get_encoding("cl100k_base")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle события приложения"""
    global retriever, rag_service
    
    # Startup
    print(f"🚀 Запуск {settings.app.name} v{settings.app.version}")
    
    try:
        print("📦 Инициализация RAG Retriever...")
        retriever = RAGRetriever()
        print("✅ RAG Retriever инициализирован")
        
        print("🤖 Инициализация RAG Service...")
        rag_service = RAGService(retriever)
        print("✅ RAG Service инициализирован")
        
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        raise
    
    yield
    
    # Shutdown
    print("👋 Остановка сервиса...")


# Создание приложения
app = FastAPI(
    title="RAG API Service",
    description="API для поиска и генерации ответов на основе справочника Яндекса по ML",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_retriever() -> RAGRetriever:
    """Dependency для получения retriever"""
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever не инициализирован")
    return retriever


def get_rag_service() -> RAGService:
    """Dependency для получения RAG service"""
    if rag_service is None:
        raise HTTPException(status_code=503, detail="RAG Service не инициализирован")
    return rag_service


@app.get("/", tags=["General"])
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "RAG API Service",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check(ret: RAGRetriever = Depends(get_retriever)):
    """Проверка здоровья сервиса"""
    try:
        # Проверка Qdrant
        collections = ret.qdrant_client.get_collections().collections
        collection_exists = any(
            c.name == settings.qdrant.collection_name for c in collections
        )
        
        vectors_count = None
        if collection_exists:
            collection_info = ret.qdrant_client.get_collection(settings.qdrant.collection_name)
            vectors_count = collection_info.points_count
        
        # Проверка Redis
        redis_connected = True
        redis_docs_count = None
        try:
            ret.parent_store.mget(["test"])
            redis_docs_count = ret.get_parent_docs_count()
        except Exception:
            redis_connected = False
        
        return HealthResponse(
            status="healthy" if collection_exists and redis_connected else "degraded",
            qdrant_connected=True,
            redis_connected=redis_connected,
            redis_parent_docs_count=redis_docs_count,
            collection_exists=collection_exists,
            collection_vectors_count=vectors_count
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service unhealthy: {str(e)}")


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_documents(
    request: SearchRequest,
    ret: RAGRetriever = Depends(get_retriever)
):
    """
    Поиск релевантных документов
    
    Использует гибридный поиск (dense + sparse embeddings) для нахождения
    наиболее релевантных фрагментов из базы знаний.
    
    Опционально можно использовать HyDE (Hypothetical Document Embeddings)
    для улучшения качества поиска на сложных запросах.
    """
    try:
        context_data = ret.get_context_for_query(
            request.query, 
            top_k=request.top_k, 
            with_scores=True,
            use_hyde=request.use_hyde
        )
        
        # Форматируем документы
        documents = []
        for i, doc in enumerate(context_data["documents"]):
            score = context_data["scores"][i] if context_data.get("scores") else None
            documents.append(DocumentResponse(
                content=doc.page_content,
                metadata=DocumentMetadata(
                    filename=doc.metadata.get("filename", "unknown"),
                    breadcrumbs=doc.metadata.get("breadcrumbs", ""),
                    url=doc.metadata.get("url", ""),
                    parent_id=doc.metadata.get("parent_id", "")
                ),
                tokens=len(encoding.encode(doc.page_content)),
                score=score
            ))
        
        return SearchResponse(
            query=context_data["query"],
            documents=documents,
            total_tokens=context_data["total_tokens"],
            num_documents=context_data["num_documents"],
            sources=context_data["sources"],
            used_hyde=context_data.get("used_hyde", False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка поиска: {str(e)}")


@app.post("/rag", response_model=RAGResponse, tags=["RAG"])
async def generate_answer(
    request: RAGRequest,
    service: RAGService = Depends(get_rag_service)
):
    """
    Генерация ответа на вопрос (RAG)
    
    Находит релевантные документы и генерирует ответ на основе контекста
    с использованием LLM.
    
    Опционально можно использовать HyDE (Hypothetical Document Embeddings)
    для улучшения качества поиска на сложных запросах.
    
    Этот эндпоинт предназначен для использования в tool-агентах.
    """
    try:
        result = service.generate_answer(
            query=request.query,
            top_k=request.top_k,
            temperature=request.temperature,
            use_hyde=request.use_hyde
        )
        
        # Форматируем документы для ответа
        documents = []
        for i, doc in enumerate(result["documents"]):
            score = result["scores"][i] if i < len(result.get("scores", [])) else None
            documents.append(DocumentResponse(
                content=doc.page_content,
                metadata=DocumentMetadata(
                    filename=doc.metadata.get("filename", "unknown"),
                    breadcrumbs=doc.metadata.get("breadcrumbs", ""),
                    url=doc.metadata.get("url", ""),
                    parent_id=doc.metadata.get("parent_id", "")
                ),
                tokens=len(encoding.encode(doc.page_content)),
                score=score
            ))
        
        return RAGResponse(
            query=result["query"],
            answer=result["answer"],
            documents=documents,
            sources=result["sources"],
            num_documents_used=result["num_documents_used"],
            total_tokens_context=result["total_tokens_context"],
            used_hyde=result.get("used_hyde", False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка генерации ответа: {str(e)}")


@app.get("/stats", tags=["General"])
async def get_stats(ret: RAGRetriever = Depends(get_retriever)):
    """Получить статистику базы знаний"""
    try:
        collection_info = ret.qdrant_client.get_collection(settings.qdrant.collection_name)
        
        return {
            "collection_name": settings.qdrant.collection_name,
            "points_count": collection_info.points_count,
            "vectors_count": collection_info.vectors_count if hasattr(collection_info, 'vectors_count') else collection_info.points_count,
            "status": collection_info.status,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка получения статистики: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

