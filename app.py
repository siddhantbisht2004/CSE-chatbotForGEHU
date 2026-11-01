import os

import gradio as gr
import spaces

from chatbot_core import CSEChatbot


chatbot = CSEChatbot()
knowledge_base_path = "knowledge_base.json"
if os.path.exists(knowledge_base_path):
    chatbot.load_knowledge_base(knowledge_base_path)


@spaces.GPU
def respond(message: str, history: list[dict[str, str]]) -> str:
    try:
        chatbot.encoder.to("cuda")
        result = chatbot.get_response(message)
        response = result["response"]
        relevant_documents = result.get("relevant_documents", [])
        if relevant_documents:
            best_match = relevant_documents[0]
            source = best_match.get("metadata", {}).get("source", "document")
            response += (
                f"\n\n**Source:** `{os.path.basename(source)}`"
                f" · **Match:** {best_match.get('similarity', 0):.0%}"
            )
        return response
    finally:
        chatbot.encoder.to("cpu")


demo = gr.ChatInterface(
    fn=respond,
    title="CSE Department AI Chatbot",
    description="Ask questions about the CSE department, courses, faculty, and research.",
    examples=[
        "Tell me about CS101",
        "Who is Dr. Sarah Johnson?",
        "What are the prerequisites for Data Structures?",
        "What research areas are available?",
    ],
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
