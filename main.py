import os
import argparse
import json
import git
from huggingface_hub import hf_hub_download

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.chat_engine.types import ChatMode
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def download_repo(repo_url: str, target_path: str):
    if os.path.exists(target_path):
        print(f"📂 Repository already exists at {target_path}")
    else:
        print(f"⬇️ Cloning repository from {repo_url}...")
        git.Repo.clone_from(repo_url, target_path)

def download_model(model_key: str, config_path: str = "models.json") -> tuple[str, dict]:
    with open(config_path, "r") as f:
        config = json.load(f)

    if model_key not in config:
        raise ValueError(f"❌ Unknown model key: {model_key}")

    model_info = config[model_key]
    local_path = hf_hub_download(
        repo_id=model_info["repo_id"],
        filename=model_info["filename"],
        cache_dir="models"
    )
    return local_path, model_info

def main():
    parser = argparse.ArgumentParser(description="Ask questions about your codebase using a local LLM.")
    parser.add_argument("--repo", type=str, required=True, help="GitHub repository URL")
    parser.add_argument("--model-key", type=str, default="mistral", help="Model key to use from models.json")
    parser.add_argument("--local-path", type=str, default="repo", help="Local path for the cloned repository")
    args = parser.parse_args()

    # Step 1: Clone the repository if needed
    download_repo(args.repo, args.local_path)

    # Step 2: Load and parse documents
    print("🔍 Loading files from the repository...")
    documents = SimpleDirectoryReader(args.local_path, recursive=True).load_data()
    print(f"✅ Loaded {len(documents)} documents")

    # Step 3: Load embedding model
    print("🧠 Initializing embedding model...")
    embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 4: Load LLM from Hugging Face
    print(f"🚀 Downloading and initializing model: {args.model_key}")
    model_path, model_info = download_model(args.model_key)

    llm = LlamaCPP(
        model_path=model_path,
        temperature=0.1,
        max_new_tokens=512,
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

    # Step 5: Create index and chat engine
    print("📚 Creating vector index...")
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    chat_engine = index.as_chat_engine(chat_mode=ChatMode.CONTEXT, llm=llm, similarity_top_k=3)

    print("✅ Ready. Ask your questions below.")
    while True:
        q = input("💬 Question (type 'exit' to quit): ")
        if q.strip().lower() == "exit":
            break
        response = chat_engine.chat(q)
        print("🤖 Answer:", response)

if __name__ == "__main__":
    main()
