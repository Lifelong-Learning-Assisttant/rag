"""Pydantic схемы для API"""
from pydantic import BaseModel, Field
from typing import List


class SearchRequest(BaseModel):
    """Запрос на поиск документов"""
    query: str = Field(..., description="Поисковый запрос", min_length=1)
    top_k: int | None = Field(None, description="Количество результатов", ge=1, le=50)
    use_hyde: bool = Field(False, description="Использовать HyDE для улучшения поиска")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "байесовский подход в машинном обучении",
                    "top_k": 5,
                    "use_hyde": False
                }
            ]
        }
    }


class DocumentMetadata(BaseModel):
    """Метаданные документа"""
    filename: str
    breadcrumbs: str = ""
    url: str = ""
    parent_id: str = ""


class DocumentResponse(BaseModel):
    """Ответ с документом"""
    content: str = Field(..., description="Содержимое документа")
    metadata: DocumentMetadata
    tokens: int = Field(..., description="Количество токенов")
    score: float | None = Field(None, description="Скор релевантности документа")


class SearchResponse(BaseModel):
    """Ответ на поиск"""
    query: str
    documents: List[DocumentResponse]
    total_tokens: int
    num_documents: int
    sources: List[dict]
    used_hyde: bool = Field(False, description="Был ли использован HyDE")


class RAGRequest(BaseModel):
    """Запрос для RAG (генерация ответа)"""
    query: str = Field(..., description="Вопрос пользователя", min_length=1)
    top_k: int | None = Field(None, description="Количество документов для контекста", ge=1, le=20)
    temperature: float = Field(0.7, description="Температура генерации", ge=0.0, le=2.0)
    use_hyde: bool = Field(False, description="Использовать HyDE для улучшения поиска")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Что такое байесовский подход в машинном обучении?",
                    "top_k": 5,
                    "temperature": 0.7,
                    "use_hyde": False
                }
            ]
        }
    }


class RAGResponse(BaseModel):
    """Ответ RAG системы"""
    query: str
    answer: str
    documents: List[DocumentResponse] = Field(..., description="Документы, использованные для ответа")
    sources: List[dict]
    num_documents_used: int
    total_tokens_context: int
    used_hyde: bool = Field(False, description="Был ли использован HyDE")


class HealthResponse(BaseModel):
    """Статус здоровья сервиса"""
    status: str
    qdrant_connected: bool
    redis_connected: bool
    collection_exists: bool
    collection_vectors_count: int | None = None

