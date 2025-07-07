import os
import argparse
import json
import hashlib
from urllib.parse import urlparse
import gradio as gr
import git
from huggingface_hub import hf_hub_download

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.core.chat_engine.types import ChatMode
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class ChatEngineWrapper:
    def __init__(self, chat_engine_with_memory, chat_engine_stateless, repo_paths):
        self.with_memory = chat_engine_with_memory
        self.stateless = chat_engine_stateless
        self.use_memory = True
        self.repo_paths = repo_paths

    def chat(self, message):
        engine = self.with_memory if self.use_memory else self.stateless
        response = engine.chat(message)
        answer = response.response

        def get_relative_repo_path(file_path: str) -> str:
            for base in self.repo_paths:
                if file_path.startswith(base):
                    repo_name = os.path.basename(base)
                    rel_path = os.path.relpath(file_path, base)
                    return os.path.join(repo_name, rel_path).replace("\\", "/")
            return file_path  # fallback

        sources = [node.node.metadata.get("file_path") for node in response.source_nodes]
        if sources:
            rel_sources = [get_relative_repo_path(s) for s in sources if s]
            answer += "\n\nSources:\n" + "\n".join(f"- {s}" for s in rel_sources)
        return answer


def download_repo(repo_url: str, target_path: str):
    if os.path.exists(target_path):
        print(f"Repository already exists at {target_path}")
    else:
        print(f"Cloning repository from {repo_url} into {target_path}...")
        git.Repo.clone_from(repo_url, target_path)


def download_model(model_key: str, config_path: str = "models.json") -> tuple[str, dict]:
    with open(config_path, "r") as f:
        config = json.load(f)

    if model_key not in config:
        raise ValueError(f"Unknown model key: {model_key}")

    model_info = config[model_key]
    local_path = hf_hub_download(
        repo_id=model_info["repo_id"],
        filename=model_info["filename"],
        cache_dir="models"
    )
    return local_path, model_info


def load_documents_from_paths(paths: list[str]) -> list:
    all_documents = []
    for base_path in paths:
        print(f"Loading documents from {base_path}...")
        repo_name = os.path.basename(os.path.normpath(base_path))
        reader = SimpleDirectoryReader(input_dir=base_path, recursive=True)
        docs = reader.load_data()

        for doc in docs:
            abs_path = doc.metadata.get("file_path")
            try:
                rel_path = os.path.relpath(abs_path, base_path).replace("\\", "/")
                doc.metadata["file_path"] = f"{repo_name}/{rel_path}"
            except ValueError:
                doc.metadata["file_path"] = f"{repo_name}/{os.path.basename(abs_path)}"

        print(f"  Loaded {len(docs)} documents from {base_path}")
        all_documents.extend(docs)
    return all_documents


def compute_index_hash(paths: list[str], embedding_model: str) -> str:
    key = "".join(paths) + embedding_model
    return hashlib.sha256(key.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your codebases using a local LLM.")
    parser.add_argument("--repos", type=str, required=True,
                        help="Comma-separated list of GitHub repository URLs")
    parser.add_argument("--web", action="store_true", help="Run in webchat mode (Gradio)")
    parser.add_argument("--model-key", type=str, default="mistral",
                        help="Model key to use from models.json")
    parser.add_argument("--local-base", type=str, default="repos",
                        help="Base directory where repositories will be cloned")
    args = parser.parse_args()

    os.makedirs(args.local_base, exist_ok=True)

    # Step 1: Clone all repositories
    repo_urls = [r.strip() for r in args.repos.split(",")]
    local_paths = []
    for url in repo_urls:
        parsed = urlparse(url)
        repo_name = os.path.splitext(parsed.path.rstrip("/").split("/")[-1])[0]
        repo_path = os.path.join(args.local_base, repo_name)
        download_repo(url, repo_path)
        local_paths.append(repo_path)

    # Step 2: Load LLM
    print(f"Downloading and initializing model: {args.model_key}")
    model_path, model_info = download_model(args.model_key)

    llm = LlamaCPP(
        model_path=model_path,
        temperature=0.1,
        max_new_tokens=2048,
        generate_kwargs={"top_p": 0.9},
        model_kwargs={
            "n_gpu_layers": model_info.get("n_gpu_layers", 20),
            "n_ctx": model_info.get("context_window", 8192),
            "n_batch": model_info.get("n_batch", 512),
            "f16_kv": model_info.get("f16_kv", True),
            "rope_scaling": model_info.get("rope_scaling", None),
        },
        verbose=True,
    )

    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Ready.")

    if args.web:
        print("Launching web UI...")
        memory_buffer = ChatMemoryBuffer.from_defaults()

        with gr.Blocks() as demo:
            gr.Markdown("# 💬 Repo Chat")

            use_memory_checkbox = gr.Checkbox(label="Use memory", value=True)
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Your message", placeholder="Ask anything about the code or docs")
            clear = gr.Button("Clear")
            state = gr.State([])

            def start_chat(message, history, memory_enabled):
                index_id = compute_index_hash(local_paths, embed_model.model_name)
                index_path = os.path.join("index_cache", index_id)

                if os.path.exists(index_path):
                    print(f"[CACHE] Loading index from {index_path}")
                    storage = StorageContext.from_defaults(persist_dir=index_path)
                    index = load_index_from_storage(storage, embed_model=embed_model)
                else:
                    print(f"[CACHE] Creating new index in {index_path}")
                    documents = load_documents_from_paths(local_paths)
                    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
                    index.storage_context.persist(persist_dir=index_path)

                engine_mem = index.as_chat_engine(
                    chat_mode=ChatMode.CONTEXT,
                    llm=llm,
                    similarity_top_k=3,
                    memory=memory_buffer
                )

                engine_stateless = index.as_chat_engine(
                    chat_mode=ChatMode.CONTEXT,
                    llm=llm,
                    similarity_top_k=3,
                    memory=None
                )

                wrapper = ChatEngineWrapper(engine_mem, engine_stateless, local_paths)
                wrapper.use_memory = memory_enabled
                reply = wrapper.chat(message)
                return "", history + [(message, reply)]

            msg.submit(start_chat, [msg, state, use_memory_checkbox], [msg, chatbot])
            clear.click(lambda: [], None, chatbot)

        demo.launch()
    else:
        # CLI fallback
        documents = load_documents_from_paths(local_paths)
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
        engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, llm=llm, similarity_top_k=3, memory=True)

        while True:
            q = input("Question (type 'exit' to quit): ")
            if q.strip().lower() == "exit":
                break
            response = engine.chat(q)
            print("Answer:", response.response)
            for node in response.source_nodes:
                print(" ↳", node.node.metadata.get("file_path"))


if __name__ == "__main__":
    main()
