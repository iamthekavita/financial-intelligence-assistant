from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import LocalModel


def test_rag_response():

    local_model = LocalModel(
        model="llama3.1:latest",
        api_key="dummy",
        base_url="http://localhost:11434/v1"
    )

    metric = AnswerRelevancyMetric(
        threshold=0.7,
        model=local_model,
        include_reason=True
    )

    test_case = LLMTestCase(
        input="What is RAG?",
        actual_output="RAG stands for Retrieval Augmented Generation.",
        expected_output="Retrieval Augmented Generation"
    )


    metric.measure(test_case)

    print("\nScore:", metric.score)
    print("\nReason:", metric.reason)

    assert metric.score >= 0.7