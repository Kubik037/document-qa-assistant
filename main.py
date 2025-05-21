import json
import logging
import time

from flask import Flask, request, jsonify, Response

from config import call_llm, call_llm_stream, OLD_VERSIONS, DOCS_DIR
from database import create_embeddings, load_and_split_documents, find_relevant_chunks

app = Flask(__name__)
logger = logging.getLogger(__name__)

@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    """API endpoint to process questions with streaming response and conversation history"""
    try:
        data = request.json
        question = data.get("question", "")
        context = data.get("context", [])
        conversation_id = data.get("conversation_id", "default")
        rerank = data.get("rerank", True)

        if not question:
            return jsonify({"error": "No question provided"}), 400

        relevant_text, sources = find_relevant_chunks(question, rerank=rerank)

        system_message = {
            "role": "system",
            "content": "You are an AI assistant that helps people find information from provided document sources within <sources></sources> tags. "
                       "Provide concise, accurate responses based only on the information in the sources sorted by relevance. Choose only the information relevant to the asked question "
                       "If the information is not in the sources, acknowledge that you don't have the answer. "
                       "If asked follow-up questions, maintain context from the conversation."
        }

        messages = [system_message]
        formatted_context = format_conversation_history(context) if context else ""

        messages.append({
            "role": "user",
            "content": f"Based strictly on the following sources and the conversation so far, answer the question like you would when helping someone:\n"
                       f"<conversation>\n{formatted_context}\n</conversation>\n"
                       f"<sources>\n{relevant_text}\n</sources>\n"
                       f"Question:\n<question>\n{question}\n</question>\n"
                       f"If there's relevant information in the sources, include specific references to where more details can be found."
        })

        start_time = time.time()

        def generate():
            try:
                # Stream the local LLM response token by token
                for token in call_llm_stream(messages):
                    yield f"data: {json.dumps({'token': token})}\n\n"

                # Send the sources as a separate message
                yield f"data: {json.dumps({'sources': sources})}\n\n"

                # Send a done signal
                yield f"data: {json.dumps({'done': True})}\n\n"

                lookup_time = time.time() - start_time
                logger.debug(f"🔹 Question answer for conversation {conversation_id} took: {lookup_time:.4f} seconds")

            except Exception as e:
                logger.exception(f'Inference failed: {str(e)}')
                yield f"data: {json.dumps({'error': f'Inference failed: {str(e)}'})}\n\n"

        return Response(generate(), mimetype='text/event-stream')

    except Exception as e:
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500


def format_conversation_history(context):
    """Format the conversation history in a way the LLM can understand"""
    formatted = ""
    for message in context:
        role = "Human" if message["role"] == "user" else "Assistant"
        formatted += f"{role}: {message['content']}\n\n"
    return formatted


@app.route("/ask", methods=["POST"])
def ask():
    """API endpoint to process questions"""
    try:
        data = request.json
        question = data.get("question", "")
        rerank = data.get("rerank", True)
        if not question:
            return jsonify({"error": "No question provided"}), 400
        relevant_text, sources = find_relevant_chunks(question, rerank=rerank)
        messages = [
            {"role": "system",
             "content": "You are an AI assistant that helps people find information for "
                        "the question in <question></question> tags from provided "
                        "document sources within <sources></sources> tags that are sorted by relevance."
                        "Provide concise, accurate responses based only on the "
                        "information in the sources. Choose only relevant information to the question from the sources."
                        "If the information is not in the sources, acknowledge that you "
                        "don't have the answer."
             },
            {"role": "user",
             "content": f"Based strictly on the following sources, answer the question "
                        f"like you would when helping a coworker and include a "
                        f"recommendation of what sources to check for more "
                        f"information:\n<sources>\n"
                        f"{relevant_text}\n</sources>\nQuestion:\n<question>\n"
                        f"{question}\n</question>"}
        ]
        start_time = time.time()
        try:
             response = call_llm(messages)
        except Exception as e:
            return jsonify({"error": f"Question inference failed: {str(e)}"}), 500
        lookup_time = time.time() - start_time
        logger.debug(f"🔹 Question answer took: {lookup_time:.4f} seconds")
        return jsonify(
            {"answer": response.choices[0].message.content, "sources": sources, "retrieved_chunks": relevant_text,
             "total_tokens": response.usage.total_tokens, "completion_tokens": response.usage.completion_tokens})
    except Exception as e:
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500


@app.route("/source_retrieval", methods=["POST"])
def source_retrieval():
    """API endpoint to retrieve document sources without querying the model."""
    logger.info("Retrieval requested...")

    try:
        data = request.json
        question = data.get("question", "")
        rerank = data.get("rerank", True)

        if not question:
            return jsonify({"error": "No question provided"}), 400

        start_time = time.time()

        relevant_text, sources = find_relevant_chunks(question, rerank=rerank)

        lookup_time = time.time() - start_time
        logger.debug(f"🔹 Retrieval took: {lookup_time:.4f} seconds")

        return jsonify({
            "retrieved_chunks": relevant_text,
            "sources": sources
        })
    except Exception as e:
        return jsonify({"error": f"Unexpected server error: {str(e)}"}), 500


if __name__ == "__main__":
    documents = load_and_split_documents(DOCS_DIR)

    if documents:
        create_embeddings(documents, OLD_VERSIONS)
    logger.info("Starting application...")
    app.run(host="0.0.0.0", port=5000)
