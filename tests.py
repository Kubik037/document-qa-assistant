import json
import logging
import os
import time
import unittest
from typing import List, Dict, Any

import numpy as np
import ragas
import ragas.llms
import requests
from datasets import Dataset
from numpy import isnan
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    LLMContextRecall,
    LLMContextPrecisionWithReference
)
from tqdm import tqdm

from config import eval_client, azure_embeddings, MODEL_NAME, chunk_size, \
    overlap, top_k, EMBED_MODEL

logger = logging.getLogger(__name__)

class RAGASEvaluationTest(unittest.TestCase):
    """Unit tests for evaluating a RAG system using RAGAS metrics."""

    # Configuration
    eval_data = None
    TEST_SIZE = "large"
    BASE_DIR ="<your_base_directory>"
    API_ENDPOINT = "http://localhost:5000/ask"
    EVALUATION_DATASET_PATH = f"{BASE_DIR}/test_data/ragas_questions_{TEST_SIZE}.json"
    OUTPUT_RESULTS_PATH = f"{BASE_DIR}/tests/test_results/{TEST_SIZE}/{EMBED_MODEL}/{MODEL_NAME}_{chunk_size}_{overlap}_{top_k}.json"

    # Thresholds for test pass/fail
    MIN_FAITHFULNESS_SCORE = 0.7
    MIN_ANSWER_RELEVANCY_SCORE = 0.7
    MIN_CONTEXT_PRECISION_SCORE = 0.6
    MIN_CONTEXT_RECALL_SCORE = 0.6
    MIN_OVERALL_SCORE = 0.7

    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(cls.OUTPUT_RESULTS_PATH), exist_ok=True)

        # Load the evaluation dataset
        cls.eval_data = cls.load_evaluation_dataset(cls.EVALUATION_DATASET_PATH)

        # Run evaluation once for all tests
        cls.metrics = cls.run_ragas_evaluation()

    @classmethod
    def load_evaluation_dataset(cls, file_path: str) -> List[Dict[str, str]]:
        """Load the evaluation dataset from a JSON file."""
        with open(file_path, 'r', encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def query_api(cls, question: str) -> Dict[str, Any]:
        """Query the RAG API with a question and return the response."""
        try:
            response = requests.post(
                cls.API_ENDPOINT,
                json={"question": question, "rerank": True},
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.exception(f"Error querying API: {str(e)}")
            return {"answer": "", "sources": [], "retrieved_chunks": "", "usage": ""}

    @classmethod
    def run_ragas_evaluation(cls) -> Dict[str, Any]:
        """Run the RAGAS evaluation on the dataset."""
        results = []

        logger.info(f"Running evaluation on {len(cls.eval_data)} questions...")
        time.sleep(0.5)
        for item in tqdm(cls.eval_data):
            question = item["question"]
            ground_truth = item["ground_truth"]

            # Query the API
            api_response = cls.query_api(question)

            generated_answer = api_response.get("answer", "")
            retrieved_chunks = api_response.get("retrieved_chunks", "")
            sources = api_response.get("sources", [])
            total_tokens = api_response.get("total_tokens", "")
            completion_tokens = api_response.get("completion_tokens", "")

            # Store results for this question
            result = {
                "question": question,
                "ground_truth": ground_truth,
                "generated_answer": generated_answer,
                "retrieved_context": retrieved_chunks,
                "retrieved_sources": sources,
                "total_tokens": total_tokens,
                "completion_tokens": completion_tokens,
            }

            results.append(result)

            time.sleep(0.5)

        # Convert to format needed for RAGAS evaluation
        ragas_data = {
            "question": [item["question"] for item in results],
            "answer": [item["generated_answer"] for item in results],
            "contexts": [[item["retrieved_context"]] for item in results],
            "ground_truth": [item["ground_truth"] for item in results],
            "ground_contexts": [item.get("context", []) for item in cls.eval_data],
        }

        # Create a Hugging Face dataset
        dataset = Dataset.from_dict(ragas_data)

        # Run RAGAS metrics
        evaluator_llm = LangchainLLMWrapper(eval_client)
        evaluator_embeddings = LangchainEmbeddingsWrapper(azure_embeddings)
        logger.info("Computing RAGAS metrics...")
        eval_result = ragas.evaluate(dataset=dataset, llm=evaluator_llm,
                                     embeddings=evaluator_embeddings,
                                     metrics=[faithfulness, answer_relevancy,
                                              LLMContextPrecisionWithReference(),
                                              LLMContextRecall()])

        # Combine metrics
        metrics = {
            "faithfulness": float(np.mean(
                [i["faithfulness"] for i in eval_result.scores if
                 not isnan(i["faithfulness"])])),
            "answer_relevancy": float(np.mean(
                [i["answer_relevancy"] for i in eval_result.scores if
                 not isnan(i["answer_relevancy"])])),
            "llm_context_precision_with_reference": float(np.mean(
                [i["llm_context_precision_with_reference"] for i in eval_result.scores if
                 not isnan(i["llm_context_precision_with_reference"])])),
            "context_recall": float(np.mean(
                [i["context_recall"] for i in eval_result.scores if
                 not isnan(i["context_recall"])])),
            "mean_total_tokens": float(
                np.mean([int(i["total_tokens"]) if i["total_tokens"] else 0 for i in results])),
            "mean_completion_tokens": float(
                np.mean([int(i["completion_tokens"]) if i["completion_tokens"] else 0 for i in results])),
            "detailed_scores": {
                "faithfulness": [i["faithfulness"] for i in eval_result.scores],
                "answer_relevancy": [i["answer_relevancy"] for i in eval_result.scores],
                "llm_context_precision_with_reference": [
                    i["llm_context_precision_with_reference"] for i in
                    eval_result.scores],
                "context_recall": [i["context_recall"] for i in eval_result.scores],
            },
            "questions_evaluated": len(results),
            "individual_results": results
        }

        # Calculate an overall score (weighted average)
        weights = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.3,
            "llm_context_precision_with_reference": 0.2,
            "context_recall": 0.2
        }

        overall_score = sum(metrics[key] * weight for key, weight in weights.items())
        metrics["overall_score"] = float(overall_score)

        # Save results
        with open(cls.OUTPUT_RESULTS_PATH, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Evaluation completed and results saved to {cls.OUTPUT_RESULTS_PATH}")
        logger.info("Summary of Results:")
        logger.info(f"Overall Score: {metrics['overall_score']:.4f}")
        logger.info(f"Faithfulness: {metrics['faithfulness']:.4f}")
        logger.info(f"Answer Relevancy: {metrics['answer_relevancy']:.4f}")
        logger.info(f"Context Precision: {metrics['llm_context_precision_with_reference']:.4f}")
        logger.info(f"Context Recall : {metrics['context_recall']:.4f}")

        return metrics

    # Test methods
    def test_overall_score(self):
        """Test if the overall score meets the minimum threshold."""
        self.assertGreaterEqual(
            self.metrics["overall_score"],
            self.MIN_OVERALL_SCORE,
            f"Overall score ({self.metrics['overall_score']:.4f}) below threshold ({self.MIN_OVERALL_SCORE})"
        )

    def test_faithfulness(self):
        """Test if the faithfulness score meets the minimum threshold."""
        self.assertGreaterEqual(
            self.metrics["faithfulness"],
            self.MIN_FAITHFULNESS_SCORE,
            f"Faithfulness score ({self.metrics['faithfulness']:.4f}) below threshold ({self.MIN_FAITHFULNESS_SCORE})"
        )

    def test_answer_relevancy(self):
        """Test if the answer relevancy score meets the minimum threshold."""
        self.assertGreaterEqual(
            self.metrics["answer_relevancy"],
            self.MIN_ANSWER_RELEVANCY_SCORE,
            f"Answer relevancy score ({self.metrics['answer_relevancy']:.4f}) below threshold ({self.MIN_ANSWER_RELEVANCY_SCORE})"
        )

    def test_context_precision(self):
        """Test if the context precision score meets the minimum threshold."""
        self.assertGreaterEqual(
            self.metrics["llm_context_precision_with_reference"],
            self.MIN_CONTEXT_PRECISION_SCORE,
            f"Context precision score ({self.metrics['llm_context_precision_with_reference']:.4f}) below threshold ({self.MIN_CONTEXT_PRECISION_SCORE})"
        )

    def test_context_recall(self):
        """Test if the context recall score meets the minimum threshold."""
        self.assertGreaterEqual(
            self.metrics["context_recall"],
            self.MIN_CONTEXT_RECALL_SCORE,
            f"Context recall score ({self.metrics['context_recall']:.4f}) below threshold ({self.MIN_CONTEXT_RECALL_SCORE})"
        )

    def test_individual_questions(self):
        """Test if each individual question meets minimum quality standards."""
        detailed_scores = self.metrics["detailed_scores"]

        # For each metric, check if any individual score is significantly below threshold
        critical_failure_threshold = 0.4  # Significantly worse than the passing threshold

        failures = []
        for i, (faith, rel, prec, rec) in enumerate(zip(
                detailed_scores["faithfulness"],
                detailed_scores["answer_relevancy"],
                detailed_scores["llm_context_precision_with_reference"],
                detailed_scores["context_recall"]
        )):
            question = self.metrics["individual_results"][i]["question"]

            # Check for critical failures
            if faith < critical_failure_threshold:
                failures.append(f"faithfulness: {faith:.2f}")
            if rel < critical_failure_threshold:
                failures.append(f"relevancy: {rel:.2f}")
            if prec < critical_failure_threshold:
                failures.append(f"precision: {prec:.2f}")
            if rec < critical_failure_threshold:
                failures.append(f"recall: {rec:.2f}")

            if failures:
                logger.error(f"Question {i + 1} has critical failures \nQuestion: {question}")
        if failures:
            self.fail("Some questions failed individual examining")


if __name__ == "__main__":
    # You can configure the test parameters here before running
    RAGASEvaluationTest.API_ENDPOINT = "http://localhost:5000/ask"
    RAGASEvaluationTest.EVALUATION_DATASET_PATH = "test_data/ragas_questions.json"
    RAGASEvaluationTest.OUTPUT_RESULTS_PATH = "test_results/ragas_evaluation_results.json"

    # Run the tests
    unittest.main()
