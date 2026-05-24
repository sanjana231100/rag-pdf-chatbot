"""
Automated evaluation framework for the RAG pipeline.
Run from the project root:

    python eval/run_eval.py

Requires:
  - Knowledge base seeded: python scripts/seed_knowledge_base.py
  - GROQ_API_KEY set in .env
  - eval/golden_qa.json present

Outputs a detailed report with metrics per question and aggregate scores.
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vectorstore import load_knowledge_base
from src.rag_chain import build_rag_chain, ask
from src.confidence import score_retrieval_confidence, should_fallback, build_fallback_response
from src.citation_verifier import verify_sources
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

GOLDEN_QA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_qa.json")
LLM_MODEL = "llama-3.3-70b-versatile"


def get_llm():
    return ChatGroq(
        model=LLM_MODEL,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )


def judge_correctness(question: str, golden: str, generated: str, llm) -> float:
    """
    Use LLM-as-judge to score answer correctness against golden answer.
    Returns a score between 0.0 and 1.0.
    """
    prompt = f"""You are an objective evaluator. Score how well the generated answer 
matches the golden answer for the given question.

Question: {question}
Golden answer: {golden}
Generated answer: {generated}

Scoring criteria:
- 1.0: Generated answer is correct and complete
- 0.7: Generated answer is mostly correct with minor gaps
- 0.5: Generated answer is partially correct
- 0.3: Generated answer has the right topic but wrong details
- 0.0: Generated answer is wrong or says it cannot find the information when it should know

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

    try:
        response = llm.invoke(prompt)
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5


def judge_faithfulness(answer: str, source_chunks: list, llm) -> float:
    """
    Score whether all claims in the answer are grounded in the source chunks.
    Returns a score between 0.0 and 1.0.
    """
    if not source_chunks:
        return 0.0

    context = "\n\n".join([doc.page_content[:300] for doc in source_chunks[:3]])

    prompt = f"""You are a faithfulness evaluator. Score how well the answer is 
grounded in the provided context. Penalise any claims not supported by the context.

Context:
{context}

Answer: {answer}

Score:
- 1.0: Every claim in the answer is directly supported by the context
- 0.7: Most claims are supported, minor unsupported details
- 0.5: About half the claims are supported
- 0.3: Few claims are supported, mostly hallucinated
- 0.0: Answer is entirely unsupported by the context

Respond with ONLY a number between 0.0 and 1.0. Nothing else."""

    try:
        response = llm.invoke(prompt)
        score = float(response.content.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5


def run_eval():
    print("\n" + "=" * 60)
    print("RAG Pipeline Evaluation Framework")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    print("\nLoading golden Q&A dataset...")
    with open(GOLDEN_QA_PATH, "r") as f:
        golden_qa = json.load(f)
    print(f"Loaded {len(golden_qa)} questions")

    print("\nLoading knowledge base...")
    vectorstore = load_knowledge_base()
    if vectorstore is None:
        print("ERROR: Knowledge base not found.")
        print("Run: python scripts/seed_knowledge_base.py")
        return

    kb_data = vectorstore.get()
    chunks = [
        Document(page_content=pc, metadata=meta)
        for pc, meta in zip(kb_data["documents"], kb_data["metadatas"])
    ]
    print(f"Knowledge base loaded: {len(chunks)} chunks")

    print("\nBuilding RAG chain...")
    chain = build_rag_chain(vectorstore, chunks)
    llm = get_llm()
    print("Ready\n")

    results = []
    difficulty_groups = {}

    for i, qa in enumerate(golden_qa):
        qid = qa["id"]
        question = qa["question"]
        golden = qa["golden_answer"]
        difficulty = qa["difficulty"]

        print(f"[{i+1:02d}/{len(golden_qa)}] Q{qid} ({difficulty}): {question[:60]}...")

        start = time.time()

        confidence = score_retrieval_confidence(vectorstore, question)

        if should_fallback(confidence):
            result = build_fallback_response(question, confidence)
            generated = result["answer"]
            source_docs = []
            is_fallback = True
        else:
            try:
                result = ask(chain, question, session_id=f"eval_{qid}")
                generated = result["answer"]
                source_docs = result["source_documents"]
                is_fallback = False
            except Exception as e:
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    print(f"\n⚠️  Rate limit hit at Q{qid}. Saving partial results...")
                    break
                raise

        try:
            correctness = judge_correctness(question, golden, generated, llm)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                print(f"\n⚠️  Rate limit hit during scoring at Q{qid}. Saving partial results...")
                break
            correctness = 0.5
        faithfulness = judge_faithfulness(generated, source_docs, llm) if source_docs else (1.0 if is_fallback and "could not find" in generated.lower() else 0.0)

        if source_docs:
            verification = verify_sources(generated, source_docs, max_claims=2)
            citation_accuracy = verification["verification_rate"]
        else:
            citation_accuracy = 1.0 if is_fallback else 0.0

        retrieval_relevance = min(confidence * 1.2, 1.0) if not is_fallback else (1.0 - confidence)

        elapsed = time.time() - start

        row = {
            "id": qid,
            "question": question,
            "golden": golden,
            "generated": generated,
            "difficulty": difficulty,
            "confidence": round(confidence, 3),
            "correctness": round(correctness, 3),
            "faithfulness": round(faithfulness, 3),
            "citation_accuracy": round(citation_accuracy, 3),
            "retrieval_relevance": round(retrieval_relevance, 3),
            "is_fallback": is_fallback,
            "elapsed_s": round(elapsed, 1)
        }
        results.append(row)

        if difficulty not in difficulty_groups:
            difficulty_groups[difficulty] = []
        difficulty_groups[difficulty].append(row)

        status = "✅" if correctness >= 0.7 else "⚠️" if correctness >= 0.4 else "❌"
        print(f"       {status} correctness={correctness:.2f} faithfulness={faithfulness:.2f} citation={citation_accuracy:.2f} ({elapsed:.1f}s)")

        time.sleep(0.5)

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
    print("=" * 60)

    metrics = ["correctness", "faithfulness", "citation_accuracy", "retrieval_relevance"]
    for metric in metrics:
        avg = sum(r[metric] for r in results) / len(results)
        print(f"  {metric:<25} {avg:.3f} ({avg*100:.1f}%)")

    print(f"\n  {'total_questions':<25} {len(results)}")
    print(f"  {'fallback_triggered':<25} {sum(1 for r in results if r['is_fallback'])}")
    print(f"  {'avg_response_time':<25} {sum(r['elapsed_s'] for r in results)/len(results):.1f}s")

    print("\n" + "-" * 60)
    print("RESULTS BY DIFFICULTY")
    print("-" * 60)
    for diff, rows in difficulty_groups.items():
        avg_correctness = sum(r["correctness"] for r in rows) / len(rows)
        print(f"  {diff:<20} n={len(rows)}  correctness={avg_correctness:.2f}")

    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "aggregate": {
                metric: round(sum(r[metric] for r in results) / len(results), 3)
                for metric in metrics
            },
            "by_difficulty": {
                diff: {
                    metric: round(sum(r[metric] for r in rows) / len(rows), 3)
                    for metric in metrics
                }
                for diff, rows in difficulty_groups.items()
            },
            "results": results
        }, f, indent=2)

    print(f"\nFull report saved to: {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    run_eval()