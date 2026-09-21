"""
main.py
========
Backend principal do assistente RAG (Retrieval-Augmented Generation).

Arquitetura:
    Supabase   -> auth, storage (fotos/PDFs/.txt), logs de conversas, feedback
    Qdrant     -> embeddings vetoriais (busca semântica)
    Together AI-> LLM (completions) + embeddings
    FastAPI    -> API REST (web + bot Telegram)
    Telegram   -> webhook que reaproveita o mesmo pipeline RAG

Variáveis de ambiente esperadas (.env):
    TOGETHER_API_KEY
    TOGETHER_CHAT_MODEL        (ex: "meta-llama/Llama-3.3-70B-Instruct-Turbo")
    TOGETHER_EMBED_MODEL       (ex: "togethercomputer/m2-bert-80M-8k-retrieval")
    SUPABASE_URL
    SUPABASE_SERVICE_KEY
    QDRANT_URL
    QDRANT_API_KEY
    QDRANT_COLLECTION          (ex: "documentos")
    TELEGRAM_BOT_TOKEN
    TELEGRAM_WEBHOOK_SECRET    (string aleatória para validar o webhook)
    APP_PUBLIC_URL             (ex: https://seu-servico.up.railway.app)

Instalação:
    pip install fastapi uvicorn python-dotenv httpx supabase qdrant-client together pydantic

Rodar localmente:
    uvicorn main:app --reload --port 8000
"""

import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import httpx
from supabase import create_client, Client as SupabaseClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from together import Together

# ---------------------------------------------------------------------------
# Config / bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-backend")

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_CHAT_MODEL = os.getenv("TOGETHER_CHAT_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
TOGETHER_EMBED_MODEL = os.getenv("TOGETHER_EMBED_MODEL", "togethercomputer/m2-bert-80M-8k-retrieval")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documentos")
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
APP_PUBLIC_URL = os.getenv("APP_PUBLIC_URL", "")

REQUIRED_VARS = {
    "TOGETHER_API_KEY": TOGETHER_API_KEY,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_KEY": SUPABASE_SERVICE_KEY,
    "QDRANT_URL": QDRANT_URL,
}
missing = [k for k, v in REQUIRED_VARS.items() if not v]
if missing:
    logger.warning("Variáveis de ambiente ausentes: %s (o servidor sobe, mas essas rotas vão falhar)", missing)

# --- Clients -----------------------------------------------------------------

together_client: Optional[Together] = Together(api_key=TOGETHER_API_KEY) if TOGETHER_API_KEY else None

supabase: Optional[SupabaseClient] = (
    create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_URL and SUPABASE_SERVICE_KEY else None
)

qdrant: Optional[QdrantClient] = (
    QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_URL else None
)


def ensure_qdrant_collection() -> None:
    """Cria a coleção no Qdrant se ainda não existir."""
    if qdrant is None:
        return
    existing = [c.name for c in qdrant.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(
                size=QDRANT_VECTOR_SIZE, distance=qmodels.Distance.COSINE
            ),
        )
        logger.info("Coleção Qdrant '%s' criada.", QDRANT_COLLECTION)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrinja em produção
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def on_startup() -> None:
    try:
        ensure_qdrant_collection()
    except Exception as exc:  # noqa: BLE001
        logger.error("Falha ao inicializar Qdrant: %s", exc)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="ID do usuário (uuid do Supabase Auth ou telegram chat id)")
    message: str
    channel: str = Field("web", description="'web' ou 'telegram'")
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    conversation_id: str


class FeedbackRequest(BaseModel):
    conversation_id: str
    rating: int = Field(..., ge=-1, le=1, description="-1 ruim, 0 neutro, 1 bom")
    comment: Optional[str] = None


class IngestRequest(BaseModel):
    doc_id: Optional[str] = None
    title: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Núcleo RAG
# ---------------------------------------------------------------------------

async def embed_text(text: str) -> List[float]:
    """Gera embedding via Together AI."""
    if together_client is None:
        raise HTTPException(status_code=500, detail="Together AI não configurado (TOGETHER_API_KEY ausente).")
    resp = together_client.embeddings.create(model=TOGETHER_EMBED_MODEL, input=text)
    return resp.data[0].embedding


async def search_context(query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Busca semântica no Qdrant."""
    if qdrant is None:
        return []
    hits = qdrant.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
    )
    return [
        {
            "score": h.score,
            "text": h.payload.get("text", ""),
            "title": h.payload.get("title", ""),
            "doc_id": h.payload.get("doc_id", ""),
        }
        for h in hits
    ]


def build_prompt(user_message: str, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    context_text = "\n\n".join(
        f"[Fonte: {c['title']}]\n{c['text']}" for c in context_chunks
    ) or "Nenhum contexto relevante encontrado."

    system_prompt = (
        "Você é um assistente que responde SOMENTE com base no contexto fornecido. "
        "Se a resposta não estiver no contexto, diga que não sabe.\n\n"
        f"CONTEXTO:\n{context_text}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]


async def generate_answer(messages: List[Dict[str, str]]) -> str:
    if together_client is None:
        raise HTTPException(status_code=500, detail="Together AI não configurado (TOGETHER_API_KEY ausente).")
    resp = together_client.chat.completions.create(
        model=TOGETHER_CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


def log_conversation(
    conversation_id: str,
    user_id: str,
    channel: str,
    user_message: str,
    answer: str,
    sources: List[Dict[str, Any]],
) -> None:
    """Grava a conversa no Supabase para virar dataset de fine-tuning depois."""
    if supabase is None:
        logger.warning("Supabase não configurado; conversa não foi logada.")
        return
    try:
        supabase.table("conversations").insert(
            {
                "id": conversation_id,
                "user_id": user_id,
                "channel": channel,
                "user_message": user_message,
                "answer": answer,
                "sources": sources,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ).execute()
    except Exception as exc:  # noqa: BLE001
        logger.error("Falha ao gravar conversa no Supabase: %s", exc)


async def run_rag_pipeline(req: ChatRequest) -> ChatResponse:
    query_vector = await embed_text(req.message)
    context_chunks = await search_context(query_vector, top_k=req.top_k)
    messages = build_prompt(req.message, context_chunks)
    answer = await generate_answer(messages)

    conversation_id = str(uuid.uuid4())
    log_conversation(conversation_id, req.user_id, req.channel, req.message, answer, context_chunks)

    return ChatResponse(answer=answer, sources=context_chunks, conversation_id=conversation_id)


# ---------------------------------------------------------------------------
# Rotas — API para site / app
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "together": together_client is not None,
        "supabase": supabase is not None,
        "qdrant": qdrant is not None,
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    """Endpoint principal usado pelo site/app e também pelo bot do Telegram."""
    return await run_rag_pipeline(req)


@app.post("/feedback")
async def feedback(req: FeedbackRequest) -> Dict[str, str]:
    """Recebe feedback do usuário para alimentar o dataset de fine-tuning."""
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase não configurado.")
    try:
        supabase.table("feedback").insert(
            {
                "conversation_id": req.conversation_id,
                "rating": req.rating,
                "comment": req.comment,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ).execute()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erro ao gravar feedback: {exc}") from exc
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(req: IngestRequest) -> Dict[str, str]:
    """
    Indexa um novo documento: gera embedding via Together AI e grava no Qdrant.
    Use isso para popular a base de conhecimento (fotos já em texto/OCR, PDFs extraídos, .txt).
    """
    if qdrant is None:
        raise HTTPException(status_code=500, detail="Qdrant não configurado.")

    doc_id = req.doc_id or str(uuid.uuid4())
    vector = await embed_text(req.text)

    qdrant.upsert(
        collection_name=QDRANT_COLLECTION,
        points=[
            qmodels.PointStruct(
                id=doc_id,
                vector=vector,
                payload={"doc_id": doc_id, "title": req.title, "text": req.text, **req.metadata},
            )
        ],
    )
    return {"status": "ok", "doc_id": doc_id}


# ---------------------------------------------------------------------------
# Rotas — Bot Telegram (webhook)
# ---------------------------------------------------------------------------

TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else None


async def send_telegram_message(chat_id: int, text: str) -> None:
    if TELEGRAM_API_BASE is None:
        logger.warning("TELEGRAM_BOT_TOKEN ausente; mensagem não enviada.")
        return
    async with httpx.AsyncClient(timeout=15) as client:
        await client.post(
            f"{TELEGRAM_API_BASE}/sendMessage",
            json={"chat_id": chat_id, "text": text},
        )


async def handle_telegram_update(update: Dict[str, Any]) -> None:
    message = update.get("message") or update.get("edited_message")
    if not message:
        return

    chat_id = message["chat"]["id"]
    text = message.get("text", "")
    if not text:
        await send_telegram_message(chat_id, "Por enquanto só entendo mensagens de texto.")
        return

    req = ChatRequest(user_id=str(chat_id), message=text, channel="telegram")
    try:
        result = await run_rag_pipeline(req)
        await send_telegram_message(chat_id, result.answer)
    except Exception as exc:  # noqa: BLE001
        logger.error("Erro processando update do Telegram: %s", exc)
        await send_telegram_message(chat_id, "Desculpe, tive um problema para responder agora.")


@app.post("/telegram/webhook")
async def telegram_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_telegram_bot_api_secret_token: Optional[str] = Header(None),
) -> Dict[str, str]:
    if TELEGRAM_WEBHOOK_SECRET and x_telegram_bot_api_secret_token != TELEGRAM_WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="Token de webhook inválido.")

    update = await request.json()
    # processa em background para responder rápido ao Telegram (evita retries/timeout)
    background_tasks.add_task(handle_telegram_update, update)
    return {"status": "received"}


@app.post("/telegram/set-webhook")
async def set_telegram_webhook() -> Dict[str, Any]:
    """Configura o webhook do bot apontando para APP_PUBLIC_URL/telegram/webhook."""
    if TELEGRAM_API_BASE is None:
        raise HTTPException(status_code=500, detail="TELEGRAM_BOT_TOKEN não configurado.")
    if not APP_PUBLIC_URL:
        raise HTTPException(status_code=500, detail="APP_PUBLIC_URL não configurado.")

    webhook_url = f"{APP_PUBLIC_URL.rstrip('/')}/telegram/webhook"
    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.post(
            f"{TELEGRAM_API_BASE}/setWebhook",
            json={"url": webhook_url, "secret_token": TELEGRAM_WEBHOOK_SECRET or None},
        )
    return resp.json()


# ---------------------------------------------------------------------------
# Entrypoint local
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
