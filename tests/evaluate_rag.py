"""Скрипт для оценки качества RAG системы на датасете"""
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import time
from collections import defaultdict

# Добавляем путь к модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from app.retriever import RAGRetriever
from app.rag_service import RAGService

load_dotenv()


class RAGEvaluator:
    """Класс для оценки качества RAG системы"""
    
    def __init__(self, retriever: RAGRetriever, rag_service: RAGService):
        self.retriever = retriever
        self.rag_service = rag_service
        self.results = []
    
    def evaluate_retrieval(self, question: str, expected_source: str, expected_topics: List[str]) -> Dict[str, Any]:
        """Оценка качества поиска документов"""
        # Получаем контекст со скорами
        context = self.retriever.get_context_for_query(question, top_k=5, with_scores=True)
        
        documents = context['documents']
        scores = context.get('scores', [])
        sources = context['sources']
        
        # Метрика 1: Найден ли ожидаемый источник
        source_found = any(expected_source in source['filename'] for source in sources)
        
        # Метрика 2: Релевантность по ключевым словам (в топ-3)
        top_docs = documents[:3]
        topic_matches = []
        for topic in expected_topics:
            found = any(topic.lower() in doc.page_content.lower() for doc in top_docs)
            topic_matches.append(found)
        
        topic_coverage = sum(topic_matches) / len(expected_topics) if expected_topics else 0
        
        # Метрика 3: Средний скор топ-3
        avg_score_top3 = sum(scores[:3]) / 3 if len(scores) >= 3 else (sum(scores) / len(scores) if scores else 1.0)
        
        # Метрика 4: Количество найденных документов
        num_docs = len(documents)
        
        return {
            'source_found': source_found,
            'topic_coverage': topic_coverage,
            'avg_score_top3': avg_score_top3,
            'num_docs_retrieved': num_docs,
            'sources': [s['filename'] for s in sources],
            'scores': scores[:5]  # Топ-5 скоров
        }
    
    def evaluate_generation(self, question: str, answer: str, expected_topics: List[str]) -> Dict[str, Any]:
        """Оценка качества генерации ответа"""
        # Метрика 1: Длина ответа (информативность)
        answer_length = len(answer)
        
        # Метрика 2: Покрытие ожидаемых топиков в ответе
        answer_lower = answer.lower()
        topics_in_answer = sum(1 for topic in expected_topics if topic.lower() in answer_lower)
        topic_coverage_answer = topics_in_answer / len(expected_topics) if expected_topics else 0
        
        # Метрика 3: Структурированность (наличие списков, параграфов)
        has_structure = any(marker in answer for marker in ['\n-', '\n*', '\n1.', '\n2.', '##'])
        
        # Метрика 4: Наличие русского текста
        russian_chars = sum(1 for c in answer if 'а' <= c.lower() <= 'я')
        total_chars = sum(1 for c in answer if c.isalpha())
        russian_ratio = russian_chars / total_chars if total_chars > 0 else 0
        
        return {
            'answer_length': answer_length,
            'topic_coverage_answer': topic_coverage_answer,
            'has_structure': has_structure,
            'russian_ratio': russian_ratio,
            'answer_preview': answer[:200] + '...' if len(answer) > 200 else answer
        }
    
    def evaluate_question(self, question_data: Dict[str, Any], use_hyde: bool = False) -> Dict[str, Any]:
        """Оценка одного вопроса"""
        question = question_data['question']
        expected_source = question_data.get('expected_source', '')
        expected_topics = question_data.get('expected_topics', [])
        category = question_data.get('category', 'unknown')
        difficulty = question_data.get('difficulty', 'unknown')
        
        print(f"\n{'='*80}")
        print(f"Вопрос: {question}")
        print(f"Категория: {category} | Сложность: {difficulty}")
        print(f"{'='*80}")
        
        # Оценка поиска
        print("\n🔍 Оценка поиска...")
        retrieval_metrics = self.evaluate_retrieval(question, expected_source, expected_topics)
        
        # Генерация ответа
        print("🤖 Генерация ответа...")
        start_time = time.time()
        result = self.rag_service.generate_answer(question, top_k=5, use_hyde=use_hyde)
        generation_time = time.time() - start_time
        
        # Оценка генерации
        print("📊 Оценка ответа...")
        generation_metrics = self.evaluate_generation(question, result['answer'], expected_topics)
        
        # Объединяем метрики
        evaluation = {
            'question': question,
            'category': category,
            'difficulty': difficulty,
            'expected_source': expected_source,
            'expected_topics': expected_topics,
            'retrieval': retrieval_metrics,
            'generation': generation_metrics,
            'generation_time': generation_time,
            'num_documents_used': result['num_documents_used'],
            'total_tokens_context': result['total_tokens_context'],
            'used_hyde': use_hyde,
            'answer': result['answer']
        }
        
        # Выводим краткую сводку
        print(f"\n✅ Результаты:")
        print(f"  • Источник найден: {'✓' if retrieval_metrics['source_found'] else '✗'}")
        print(f"  • Покрытие топиков (поиск): {retrieval_metrics['topic_coverage']:.1%}")
        print(f"  • Покрытие топиков (ответ): {generation_metrics['topic_coverage_answer']:.1%}")
        print(f"  • Средний скор топ-3: {retrieval_metrics['avg_score_top3']:.3f}")
        print(f"  • Длина ответа: {generation_metrics['answer_length']} символов")
        print(f"  • Время генерации: {generation_time:.2f}s")
        print(f"  • Найденные источники: {', '.join(retrieval_metrics['sources'][:3])}")
        
        return evaluation
    
    def evaluate_dataset(self, dataset_path: str, use_hyde: bool = False) -> Dict[str, Any]:
        """Оценка всего датасета"""
        print(f"\n{'#'*80}")
        print(f"# ОЦЕНКА RAG СИСТЕМЫ НА ДАТАСЕТЕ")
        print(f"# Датасет: {dataset_path}")
        print(f"# HyDE: {'включен' if use_hyde else 'выключен'}")
        print(f"{'#'*80}")
        
        # Загружаем датасет
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"\n📚 Загружено {len(dataset)} вопросов")
        
        # Оцениваем каждый вопрос
        results = []
        for i, question_data in enumerate(dataset, 1):
            print(f"\n\n{'='*80}")
            print(f"ВОПРОС {i}/{len(dataset)}")
            print(f"{'='*80}")
            
            try:
                evaluation = self.evaluate_question(question_data, use_hyde=use_hyde)
                results.append(evaluation)
            except Exception as e:
                print(f"\n❌ Ошибка при обработке вопроса: {e}")
                results.append({
                    'question': question_data['question'],
                    'error': str(e)
                })
        
        # Вычисляем агрегированные метрики
        print(f"\n\n{'#'*80}")
        print(f"# ИТОГОВЫЕ МЕТРИКИ")
        print(f"{'#'*80}")
        
        metrics = self.calculate_aggregate_metrics(results)
        self.print_metrics(metrics)
        
        return {
            'results': results,
            'aggregate_metrics': metrics,
            'dataset_size': len(dataset),
            'used_hyde': use_hyde
        }
    
    def calculate_aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Вычисление агрегированных метрик"""
        # Фильтруем результаты с ошибками
        valid_results = [r for r in results if 'error' not in r]
        
        if not valid_results:
            return {}
        
        # Метрики поиска
        source_found_rate = sum(1 for r in valid_results if r['retrieval']['source_found']) / len(valid_results)
        avg_topic_coverage_retrieval = sum(r['retrieval']['topic_coverage'] for r in valid_results) / len(valid_results)
        avg_score = sum(r['retrieval']['avg_score_top3'] for r in valid_results) / len(valid_results)
        avg_docs_retrieved = sum(r['retrieval']['num_docs_retrieved'] for r in valid_results) / len(valid_results)
        
        # Метрики генерации
        avg_answer_length = sum(r['generation']['answer_length'] for r in valid_results) / len(valid_results)
        avg_topic_coverage_answer = sum(r['generation']['topic_coverage_answer'] for r in valid_results) / len(valid_results)
        structured_answers_rate = sum(1 for r in valid_results if r['generation']['has_structure']) / len(valid_results)
        avg_russian_ratio = sum(r['generation']['russian_ratio'] for r in valid_results) / len(valid_results)
        
        # Производительность
        avg_generation_time = sum(r['generation_time'] for r in valid_results) / len(valid_results)
        avg_tokens = sum(r['total_tokens_context'] for r in valid_results) / len(valid_results)
        
        # Метрики по категориям
        by_category = defaultdict(list)
        for r in valid_results:
            by_category[r['category']].append(r)
        
        category_metrics = {}
        for category, cat_results in by_category.items():
            category_metrics[category] = {
                'count': len(cat_results),
                'source_found_rate': sum(1 for r in cat_results if r['retrieval']['source_found']) / len(cat_results),
                'avg_topic_coverage': sum(r['retrieval']['topic_coverage'] for r in cat_results) / len(cat_results),
                'avg_answer_length': sum(r['generation']['answer_length'] for r in cat_results) / len(cat_results)
            }
        
        # Метрики по сложности
        by_difficulty = defaultdict(list)
        for r in valid_results:
            by_difficulty[r['difficulty']].append(r)
        
        difficulty_metrics = {}
        for difficulty, diff_results in by_difficulty.items():
            difficulty_metrics[difficulty] = {
                'count': len(diff_results),
                'source_found_rate': sum(1 for r in diff_results if r['retrieval']['source_found']) / len(diff_results),
                'avg_topic_coverage': sum(r['retrieval']['topic_coverage'] for r in diff_results) / len(diff_results),
                'avg_score': sum(r['retrieval']['avg_score_top3'] for r in diff_results) / len(diff_results)
            }
        
        return {
            'overall': {
                'total_questions': len(results),
                'successful_questions': len(valid_results),
                'failed_questions': len(results) - len(valid_results)
            },
            'retrieval': {
                'source_found_rate': source_found_rate,
                'avg_topic_coverage': avg_topic_coverage_retrieval,
                'avg_score_top3': avg_score,
                'avg_docs_retrieved': avg_docs_retrieved
            },
            'generation': {
                'avg_answer_length': avg_answer_length,
                'avg_topic_coverage': avg_topic_coverage_answer,
                'structured_answers_rate': structured_answers_rate,
                'avg_russian_ratio': avg_russian_ratio
            },
            'performance': {
                'avg_generation_time': avg_generation_time,
                'avg_tokens_context': avg_tokens
            },
            'by_category': category_metrics,
            'by_difficulty': difficulty_metrics
        }
    
    def print_metrics(self, metrics: Dict[str, Any]):
        """Красивый вывод метрик"""
        if not metrics:
            print("❌ Нет метрик для отображения")
            return
        
        print("\n📊 ОБЩИЕ МЕТРИКИ")
        print("-" * 80)
        overall = metrics['overall']
        print(f"  Всего вопросов: {overall['total_questions']}")
        print(f"  Успешно обработано: {overall['successful_questions']}")
        print(f"  Ошибок: {overall['failed_questions']}")
        
        print("\n🔍 МЕТРИКИ ПОИСКА")
        print("-" * 80)
        retrieval = metrics['retrieval']
        print(f"  Источник найден: {retrieval['source_found_rate']:.1%}")
        print(f"  Покрытие топиков: {retrieval['avg_topic_coverage']:.1%}")
        print(f"  Средний скор топ-3: {retrieval['avg_score_top3']:.3f}")
        print(f"  Среднее кол-во документов: {retrieval['avg_docs_retrieved']:.1f}")
        
        print("\n📝 МЕТРИКИ ГЕНЕРАЦИИ")
        print("-" * 80)
        generation = metrics['generation']
        print(f"  Средняя длина ответа: {generation['avg_answer_length']:.0f} символов")
        print(f"  Покрытие топиков в ответе: {generation['avg_topic_coverage']:.1%}")
        print(f"  Структурированные ответы: {generation['structured_answers_rate']:.1%}")
        print(f"  Доля русского текста: {generation['avg_russian_ratio']:.1%}")
        
        print("\n⚡ ПРОИЗВОДИТЕЛЬНОСТЬ")
        print("-" * 80)
        performance = metrics['performance']
        print(f"  Среднее время генерации: {performance['avg_generation_time']:.2f}s")
        print(f"  Средний размер контекста: {performance['avg_tokens_context']:.0f} токенов")
        
        print("\n📂 ПО КАТЕГОРИЯМ")
        print("-" * 80)
        for category, cat_metrics in metrics['by_category'].items():
            print(f"  {category.upper()} (n={cat_metrics['count']})")
            print(f"    • Источник найден: {cat_metrics['source_found_rate']:.1%}")
            print(f"    • Покрытие топиков: {cat_metrics['avg_topic_coverage']:.1%}")
            print(f"    • Длина ответа: {cat_metrics['avg_answer_length']:.0f} символов")
        
        print("\n🎯 ПО СЛОЖНОСТИ")
        print("-" * 80)
        for difficulty, diff_metrics in metrics['by_difficulty'].items():
            print(f"  {difficulty.upper()} (n={diff_metrics['count']})")
            print(f"    • Источник найден: {diff_metrics['source_found_rate']:.1%}")
            print(f"    • Покрытие топиков: {diff_metrics['avg_topic_coverage']:.1%}")
            print(f"    • Средний скор: {diff_metrics['avg_score']:.3f}")
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Сохранение результатов в JSON"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Результаты сохранены в: {output_path}")


def main():
    """Основная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Оценка качества RAG системы')
    parser.add_argument('--dataset', type=str, default='tests/dataset.json', help='Путь к датасету')
    parser.add_argument('--output', type=str, default='tests/evaluation_results.json', help='Путь для сохранения результатов')
    parser.add_argument('--hyde', action='store_true', help='Использовать HyDE')
    parser.add_argument('--limit', type=int, default=None, help='Ограничить количество вопросов')
    
    args = parser.parse_args()
    
    print("\n🚀 Инициализация RAG системы...")
    retriever = RAGRetriever()
    rag_service = RAGService(retriever)
    
    print("✅ RAG система инициализирована")
    
    # Создаем evaluator
    evaluator = RAGEvaluator(retriever, rag_service)
    
    # Если нужно ограничить количество вопросов
    if args.limit:
        with open(args.dataset, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        dataset = dataset[:args.limit]
        temp_dataset_path = 'tests/dataset_limited.json'
        with open(temp_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        dataset_path = temp_dataset_path
    else:
        dataset_path = args.dataset
    
    # Оцениваем датасет
    results = evaluator.evaluate_dataset(dataset_path, use_hyde=args.hyde)
    
    # Сохраняем результаты
    evaluator.save_results(results, args.output)
    
    print("\n✅ Оценка завершена!")


if __name__ == '__main__':
    main()

