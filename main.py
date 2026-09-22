"""
main.py
========
Backend principal do assistente RAG (Retrieval-Augmented Generation).

Arquitetura:
    Supabase   -> logs de conversas, feedback e configuração do modelo ativo
    Qdrant     -> embeddings vetoriais (busca semântica)
    Gemini API -> LLM (conversa) + embeddings (gemini-embedding-001)
    FastAPI    -> API REST (web + bot Telegram) + painel administrativo em /painel
    Telegram   -> webhook que reaproveita o mesmo pipeline RAG

Variáveis de ambiente (.env / Railway -> Variables):

  Obrigatórias
    GEMINI_API_KEY             chave do Google AI Studio (aistudio.google.com/apikey)
    SUPABASE_URL
    SUPABASE_SERVICE_KEY
    QDRANT_URL
    QDRANT_API_KEY             (Qdrant Cloud)

  Telegram (só se usar o bot)
    TELEGRAM_BOT_TOKEN
    TELEGRAM_WEBHOOK_SECRET    (string aleatória para validar o webhook)
    APP_PUBLIC_URL             (ex: https://seu-servico.up.railway.app)

  Opcionais (têm padrão)
    GEMINI_CHAT_MODEL            padrão "gemini-2.5-flash-lite" (preferido no Supabase model_config)
    GEMINI_FALLBACK_MODELS       padrão "gemini-2.5-flash-lite,gemini-flash-lite-latest,gemini-2.5-flash,gemini-flash-latest"
                                  Lista (separada por vírgula) de modelos que competem entre si
                                  quando é preciso descobrir qual está respondendo rápido agora.
    GEMINI_MAX_RETRIES            padrão "1" (tentativas extras no MESMO modelo antes de descartá-lo)
    GEMINI_RETRY_BASE_DELAY_SECONDS  padrão "1.0" (espera entre tentativas, dobra a cada tentativa)
    GEMINI_EMBED_MODEL          padrão "gemini-embedding-001"
    QDRANT_COLLECTION          padrão "documentos"
    QDRANT_VECTOR_SIZE         padrão "768" (dimensão do embedding; precisa bater com a coleção)
    QDRANT_SCORE_THRESHOLD     padrão "0.55" (abaixo disso o trecho é ignorado no prompt)
    CHAT_HISTORY_TURNS         padrão "6"   (quantas trocas anteriores o modelo enxerga)
    CHAT_HISTORY_WINDOW_MINUTES padrão "360" (só usa trocas dos últimos N minutos)
    MODEL_CACHE_TTL_SECONDS    padrão "300" (cache do modelo lido de model_config)
    PORT                       padrão "8000" (só usado ao rodar `python main.py`)

Instalação (requirements.txt):
    fastapi uvicorn python-dotenv httpx supabase qdrant-client google-genai pydantic

Rodar localmente:
    uvicorn main:app --reload --port 8000

Painel administrativo:
    Coloque o arquivo do painel em  painel/index.html  (ao lado deste main.py).
    Ele fica em  /painel/  (com a barra no final). A raiz "/" redireciona para lá.

Novidades desta versão — "corrida entre modelos" (fast-model race):
    Em vez de usar sempre um único modelo e ficar tentando de novo nele quando ele dá
    erro (o que causava demoras de 20-60s quando um modelo específico ficava instável,
    como aconteceu com 503 "Service Unavailable"), o backend agora:

    1. Mantém um "modelo rápido" (sticky) em memória. Se ele existe, usa DIRETO nele —
       sem gastar chamadas extras — porque ele já provou que está respondendo bem.
    2. Se não existe um modelo rápido ainda, OU se o modelo rápido atual falhar
       (erro transitório: 503/429/timeout), o backend dispara TODOS os modelos da
       lista GEMINI_FALLBACK_MODELS em paralelo ("corrida"). O primeiro que responder
       com sucesso vence; os outros são ignorados (as chamadas de rede que já saíram
       não podem ser "mortas" de fato, mas a resposta delas é simplesmente descartada).
    3. O vencedor da corrida passa a ser o novo "modelo rápido" e é usado direto nas
       próximas perguntas — inclusive de outros usuários/canais — até falhar de novo.

    Isso resolve o padrão visto nos logs: um modelo instável fazia o usuário esperar
    minutos por causa de retries sequenciais; agora, ao primeiro erro transitório, o
    sistema já parte para a corrida e volta a responder rápido com outro modelo.

Novidades de versões anteriores (mantidas):
    - Troca do Together AI pelo Gemini API (chat + embeddings), com plano gratuito.
    - Memória de conversa: o modelo recebe as últimas trocas do mesmo usuário e canal,
      então mantém o assunto ("e quanto custa?", "explica isso melhor").
      A busca no Qdrant também considera a pergunta anterior em mensagens curtas.
    - Comando /novo no Telegram (e "new_topic" no /chat) para começar um assunto novo.
    - Chamadas bloqueantes (Gemini, Qdrant, Supabase) rodam em threads, sem travar o servidor.
    - Erros da Gemini viram mensagens claras (chave inválida, limite do plano gratuito).
    - Busca compatível com versões novas do qdrant-client (query_points).

Tabelas no Supabase:
    model_config, conversations e feedback (SQL no final deste arquivo).
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import httpx
from google import genai
from google.genai import types as gtypes
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from supabase import Client as SupabaseClient
from supabase import create_client

# ---------------------------------------------------------------------------
# Config / bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag-backend")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")
GEMINI_EMBED_MODEL = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")

_DEFAULT_FALLBACK_MODELS = "gemini-2.5-flash-lite,gemini-flash-lite-latest,gemini-2.5-flash,gemini-flash-latest"
GEMINI_FALLBACK_MODELS = [
    m.strip() for m in os.getenv("GEMINI_FALLBACK_MODELS", _DEFAULT_FALLBACK_MODELS).split(",") if m.strip()
]

GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "1"))
GEMINI_RETRY_BASE_DELAY = float(os.getenv("GEMINI_RETRY_BASE_DELAY_SECONDS", "1.0"))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documentos")
QDRANT_VECTOR_SIZE = int(os.getenv("QDRANT_VECTOR_SIZE", "768"))
QDRANT_SCORE_THRESHOLD = float(os.getenv("QDRANT_SCORE_THRESHOLD", "0.55"))

CHAT_HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "6"))
CHAT_HISTORY_WINDOW_MINUTES = int(os.getenv("CHAT_HISTORY_WINDOW_MINUTES", "360"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")
APP_PUBLIC_URL = os.getenv("APP_PUBLIC_URL", "")

# O gemini-embedding-001 aceita ~2048 tokens por texto; 7000 caracteres é um teto seguro.
MAX_INGEST_CHARS = 7000

REQUIRED_VARS = {
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "SUPABASE_URL": SUPABASE_URL,
    "SUPABASE_SERVICE_KEY": SUPABASE_SERVICE_KEY,
    "QDRANT_URL": QDRANT_URL,
}
missing = [k for k, v in REQUIRED_VARS.items() if not v]
if missing:
    logger.warning("Variáveis de ambiente ausentes: %s (o servidor sobe, mas essas rotas vão falhar)", missing)

# --- Clients -----------------------------------------------------------------

gemini_client: Optional[genai.Client] = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

supabase: Optional[SupabaseClient] = (
    create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY) if SUPABASE_URL and SUPABASE_SERVICE_KEY else None
)

qdrant: Optional[QdrantClient] = (
    QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY) if QDRANT_URL else None
)

# --- Modelo ativo (com hot-swap via Supabase) --------------------------------
# Permite trocar o modelo de chat preferido sem reiniciar o servidor: a tabela
# `model_config` é consultada a cada MODEL_CACHE_TTL segundos. Esse valor entra
# como primeiro candidato na corrida entre modelos (ver mais abaixo). Se a
# tabela não existir ou a consulta falhar, usa GEMINI_CHAT_MODEL.

MODEL_CACHE_TTL = int(os.getenv("MODEL_CACHE_TTL_SECONDS", "300"))  # 5 min

_model_cache: Dict[str, Any] = {"value": GEMINI_CHAT_MODEL, "fetched_at": 0.0}


def _is_valid_gemini_model_name(name: str) -> bool:
    """Rejeita nomes de outros provedores (ex.: 'meta-llama/...') que sobraram no banco."""
    if not name or "/" in name and not name.startswith(("models/", "tunedModels/")):
        return False
    return True


def get_active_model() -> str:
    """Retorna o modelo de chat preferido agora (com cache em memória, vindo do Supabase)."""
    now = time.time()
    if now - _model_cache["fetched_at"] < MODEL_CACHE_TTL:
        return _model_cache["value"]

    if supabase is not None:
        try:
            resp = supabase.table("model_config").select("active_model").eq("id", 1).single().execute()
            value = (resp.data or {}).get("active_model")
            if value and _is_valid_gemini_model_name(value):
                _model_cache["value"] = value
                _model_cache["fetched_at"] = now
                return value
            if value:
                logger.warning(
                    "model_config.active_model='%s' não parece um modelo Gemini; ignorando e usando '%s'.",
                    value,
                    _model_cache["value"],
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Não foi possível consultar model_config no Supabase (%s); usando fallback.", exc)

    _model_cache["fetched_at"] = now
    return _model_cache["value"]


def ensure_qdrant_collection() -> None:
    """Cria a coleção no Qdrant se ainda não existir e avisa se a dimensão não bater."""
    if qdrant is None:
        return
    existing = [c.name for c in qdrant.get_collections().collections]
    if QDRANT_COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=qmodels.VectorParams(size=QDRANT_VECTOR_SIZE, distance=qmodels.Distance.COSINE),
        )
        logger.info("Coleção Qdrant '%s' criada (dimensão %s).", QDRANT_COLLECTION, QDRANT_VECTOR_SIZE)
        return
    try:
        info = qdrant.get_collection(QDRANT_COLLECTION)
        size = getattr(info.config.params.vectors, "size", None)
        if size is not None and size != QDRANT_VECTOR_SIZE:
            logger.error(
                "A coleção '%s' tem dimensão %s, mas QDRANT_VECTOR_SIZE=%s. Apague a coleção no Qdrant "
                "(ou use outro QDRANT_COLLECTION) e reinicie, senão /ingest e /chat vão falhar.",
                QDRANT_COLLECTION,
                size,
                QDRANT_VECTOR_SIZE,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Não foi possível conferir a dimensão da coleção: %s", exc)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG Backend", version="1.3.0")

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
        await asyncio.to_thread(ensure_qdrant_collection)
    except Exception as exc:  # noqa: BLE001
        logger.error("Falha ao inicializar Qdrant: %s", exc)
    logger.info("Modelos candidatos para a corrida: %s", GEMINI_FALLBACK_MODELS)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str = Field(..., description="ID do usuário (uuid do Supabase Auth ou telegram chat id)")
    message: str
    channel: str = Field("web", description="'web' ou 'telegram'")
    top_k: int = 5
    new_topic: bool = Field(False, description="Se true, ignora o histórico anterior e começa um assunto novo")


class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    conversation_id: str
    model: Optional[str] = Field(None, description="Modelo Gemini que respondeu esta mensagem")


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
# Erros amigáveis da Gemini
# ---------------------------------------------------------------------------

def _gemini_http_error(exc: Exception, what: str) -> HTTPException:
    code = getattr(exc, "code", None)
    msg = str(exc)
    if code in (401, 403) or "API key not valid" in msg or "API_KEY_INVALID" in msg:
        return HTTPException(status_code=502, detail=f"{what}: a chave da Gemini foi rejeitada (confira GEMINI_API_KEY).")
    if code == 429 or "RESOURCE_EXHAUSTED" in msg:
        return HTTPException(
            status_code=429,
            detail=f"{what}: limite do plano gratuito da Gemini atingido. Tente de novo em instantes.",
        )
    if code == 404:
        return HTTPException(status_code=502, detail=f"{what}: modelo não encontrado ({msg[:200]}).")
    return HTTPException(status_code=502, detail=f"{what}: {msg[:300]}")


def _is_retryable_gemini_error(exc: Exception) -> bool:
    """Erros transitórios (vale tentar de novo ou trocar de modelo), em vez de erros
    definitivos como chave inválida ou modelo inexistente."""
    code = getattr(exc, "code", None)
    if code in (429, 500, 503, 504):
        return True
    msg = str(exc)
    transient_markers = (
        "UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "DEADLINE_EXCEEDED",
        "INTERNAL",
        "Service Unavailable",
        "overloaded",
        "timeout",
        "Timeout",
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectError",
    )
    return any(marker in msg for marker in transient_markers)


async def _retry_delay(attempt: int) -> None:
    delay = min(GEMINI_RETRY_BASE_DELAY * (2 ** attempt), 8.0)
    await asyncio.sleep(delay)


# ---------------------------------------------------------------------------
# Núcleo RAG
# ---------------------------------------------------------------------------

def _embed_sync(text: str, task_type: str) -> List[float]:
    resp = gemini_client.models.embed_content(  # type: ignore[union-attr]
        model=GEMINI_EMBED_MODEL,
        contents=text,
        config=gtypes.EmbedContentConfig(task_type=task_type, output_dimensionality=QDRANT_VECTOR_SIZE),
    )
    return list(resp.embeddings[0].values)


async def embed_text(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> List[float]:
    """Gera embedding via Gemini. Use RETRIEVAL_DOCUMENT ao indexar e RETRIEVAL_QUERY ao buscar.
    Faz algumas tentativas em caso de erro transitório (503/429), sem trocar de modelo —
    só existe um modelo de embedding configurado."""
    if gemini_client is None:
        raise HTTPException(status_code=500, detail="Gemini não configurado (GEMINI_API_KEY ausente).")

    last_exc: Optional[Exception] = None
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(_embed_sync, text, task_type)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_retryable_gemini_error(exc) and attempt < GEMINI_MAX_RETRIES:
                logger.warning("Embedding falhou (tentativa %s/%s); tentando de novo...", attempt + 1, GEMINI_MAX_RETRIES)
                await _retry_delay(attempt)
                continue
            break
    raise _gemini_http_error(last_exc or RuntimeError("erro desconhecido"), "Falha ao gerar embedding")


def _search_sync(query_vector: List[float], top_k: int) -> List[Any]:
    if hasattr(qdrant, "query_points"):
        res = qdrant.query_points(  # type: ignore[union-attr]
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        return list(res.points)
    return list(
        qdrant.search(  # type: ignore[union-attr]
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
        )
    )


async def search_context(query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
    """Busca semântica no Qdrant."""
    if qdrant is None:
        return []
    top_k = max(1, min(int(top_k), 20))
    try:
        hits = await asyncio.to_thread(_search_sync, query_vector, top_k)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Erro na busca do Qdrant: {str(exc)[:300]}") from exc
    results = []
    for h in hits:
        payload = h.payload or {}
        results.append(
            {
                "score": h.score,
                "text": payload.get("text", ""),
                "title": payload.get("title", ""),
                "doc_id": payload.get("doc_id", ""),
            }
        )
    return results


# --- Memória da conversa -----------------------------------------------------

# "Assunto novo": o histórico anterior a este instante é ignorado (fica em memória do processo).
_history_cutoff: Dict[Tuple[str, str], datetime] = {}


def reset_history(user_id: str, channel: str) -> None:
    _history_cutoff[(user_id, channel)] = datetime.now(timezone.utc)


def _history_sync(user_id: str, channel: str) -> List[Dict[str, str]]:
    """Últimas trocas do mesmo usuário/canal, da mais antiga para a mais nova."""
    if supabase is None or CHAT_HISTORY_TURNS <= 0:
        return []
    since = datetime.now(timezone.utc) - timedelta(minutes=CHAT_HISTORY_WINDOW_MINUTES)
    cutoff = _history_cutoff.get((user_id, channel))
    if cutoff and cutoff > since:
        since = cutoff
    try:
        rows = (
            supabase.table("conversations")
            .select("user_message, answer, created_at")
            .eq("user_id", user_id)
            .eq("channel", channel)
            .gte("created_at", since.isoformat())
            .order("created_at", desc=True)
            .limit(CHAT_HISTORY_TURNS)
            .execute()
        ).data or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Não foi possível ler o histórico de conversa (%s); seguindo sem memória.", exc)
        return []
    history: List[Dict[str, str]] = []
    for r in reversed(rows):
        history.append({"role": "user", "content": r.get("user_message") or ""})
        history.append({"role": "assistant", "content": r.get("answer") or ""})
    return history


def build_retrieval_query(message: str, history: List[Dict[str, str]]) -> str:
    """
    Mensagens curtas ("e quanto custa?") perdem o sentido sozinhas. Nesse caso a busca
    no Qdrant usa também a pergunta anterior do usuário.
    """
    if history and len(message) < 80:
        last_user = next((h["content"] for h in reversed(history) if h["role"] == "user"), "")
        if last_user:
            return f"{last_user}\n{message}"[:1500]
    return message


def build_system_prompt(context_chunks: List[Dict[str, Any]]) -> str:
    """
    Kanda: assistente virtual da empresa. Fala como a voz da empresa, sem citar
    manual, documento, fonte ou foto.
    """
    relevant_chunks = [c for c in context_chunks if c.get("score", 0) >= QDRANT_SCORE_THRESHOLD]

    identidade = (
        "Você é a Kanda, a assistente virtual da empresa. Sua função é explicar e ajudar as "
        "pessoas a realizarem as atividades na plataforma. Você é a voz da empresa: fale em "
        "nome dela, com segurança, de forma simpática, clara e direta. Quando fizer sentido, "
        "use 'nós' e 'nossa plataforma'.\n\n"
    )

    regras_voz = (
        "REGRAS DE VOZ (obrigatórias):\n"
        "- Nunca diga de onde a informação vem. Proibido usar: 'manual', 'documento', 'material', "
        "'guia', 'fonte', 'foto', 'imagem', 'arquivo', 'contexto', 'informações fornecidas', "
        "'base de conhecimento', 'segundo', 'de acordo com', 'conforme consta', 'o texto diz'.\n"
        "- Nunca escreva 'o manual afirma', 'no manual', 'com base no manual' ou parecidos. "
        "Diga a informação diretamente, como algo que você mesma sabe.\n"
        "- Se a informação disser 'o manual explica X', transforme em 'X'. Se disser 'os links "
        "que estão no manual', transforme em 'aqui estão os links'. Se citar uma foto ou imagem, "
        "descreva o passo em texto, sem mencionar a imagem.\n"
        "- Quando houver links, grupos ou contatos, entregue-os diretamente na resposta.\n\n"
        "EXEMPLOS:\n"
        "Errado: 'De acordo com o manual, o grupo do WhatsApp serve para tirar dúvidas.'\n"
        "Certo: 'O nosso grupo do WhatsApp serve para tirar dúvidas. Aqui está o link: ...'\n"
        "Errado: 'Com base nas informações do documento, você deve clicar em Entrar.'\n"
        "Certo: 'Para começar, clique em Entrar.'\n\n"
        "Use o histórico da conversa para entender referências como 'isso' ou 'e o outro?' e "
        "manter o assunto. Responda sempre no mesmo idioma do usuário.\n"
    )

    conversa_social = (
        "CONVERSA SOCIAL: cumprimentos, 'tudo bem?', agradecimentos, despedidas e perguntas "
        "sobre quem você é podem ser respondidos normalmente, de forma calorosa. Se perguntarem "
        "quem você é, diga que é a Kanda, a assistente virtual que explica e ajuda a realizar "
        "as atividades.\n\n"
    )

    if relevant_chunks:
        context_text = "\n\n".join(c["text"] for c in relevant_chunks)
        return (
            identidade
            + conversa_social
            + "INFORMAÇÕES FACTUAIS: responda usando exclusivamente o CONHECIMENTO abaixo. Nunca "
            "use conhecimento geral, nunca invente e nunca deduza algo que não esteja escrito. Se "
            "o CONHECIMENTO não tiver a resposta, diga com simpatia que não tem essa informação "
            "no momento. Se cobrir só parte da pergunta, responda essa parte e diga que não tem "
            "o restante.\n\n"
            + regras_voz
            + f"\nCONHECIMENTO (é seu, não o mencione como fonte):\n{context_text}"
        )

    return (
        identidade
        + conversa_social
        + "Para esta mensagem você não tem nenhuma informação específica. Se ela pedir qualquer "
        "informação factual (sobre a empresa ou qualquer outro assunto), diga com simpatia que "
        "não tem essa informação no momento e ofereça ajuda com outra dúvida sobre a plataforma. "
        "Nunca responda com conhecimento geral, mesmo que a pergunta seja simples.\n\n"
        + regras_voz
    )


def _to_gemini_contents(history: List[Dict[str, str]], user_message: str) -> List[Any]:
    contents = []
    for turn in history:
        role = "user" if turn["role"] == "user" else "model"
        if not turn["content"]:
            continue
        contents.append(gtypes.Content(role=role, parts=[gtypes.Part(text=turn["content"])]))
    contents.append(gtypes.Content(role="user", parts=[gtypes.Part(text=user_message)]))
    return contents


def _generate_sync(model: str, system_prompt: str, history: List[Dict[str, str]], user_message: str) -> str:
    resp = gemini_client.models.generate_content(  # type: ignore[union-attr]
        model=model,
        contents=_to_gemini_contents(history, user_message),
        config=gtypes.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.5,
            max_output_tokens=2048,
        ),
    )
    return (resp.text or "").strip()


# --- Corrida entre modelos + "modelo rápido" (sticky) ------------------------
#
# _fast_model["name"] guarda o último modelo que respondeu com sucesso. Enquanto
# ele existir, é usado direto (sem corrida) — é o caminho rápido e barato.
# Quando ele falha (erro transitório) ou ainda não existe, disparamos todos os
# candidatos em paralelo e o primeiro a responder com sucesso vira o novo sticky.
#
# Limitação conhecida: como as chamadas de rede correm dentro de threads
# (asyncio.to_thread), cancelar a tarefa asyncio não interrompe de fato uma
# chamada HTTP já em andamento — ela só é ignorada quando termina. Isso é
# aceitável aqui porque a corrida só acontece ocasionalmente (na primeira vez
# e sempre que o modelo sticky falha), não em toda mensagem.

_fast_model: Dict[str, Optional[str]] = {"name": None}
_fast_model_lock = asyncio.Lock()


def _candidate_models() -> List[str]:
    """Lista de modelos a considerar na corrida: o preferido do Supabase primeiro,
    depois os da lista de fallback, sem repetir."""
    preferred = get_active_model()
    ordered = [preferred] + [m for m in GEMINI_FALLBACK_MODELS if m != preferred]
    seen = set()
    result = []
    for m in ordered:
        if m and _is_valid_gemini_model_name(m) and m not in seen:
            seen.add(m)
            result.append(m)
    return result or [GEMINI_CHAT_MODEL]


async def _try_model_with_retries(model: str, system_prompt: str, history: List[Dict[str, str]], user_message: str) -> str:
    """Tenta um único modelo, com pequenas re-tentativas em erros transitórios."""
    last_exc: Optional[Exception] = None
    for attempt in range(GEMINI_MAX_RETRIES + 1):
        try:
            return await asyncio.to_thread(_generate_sync, model, system_prompt, history, user_message)
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if _is_retryable_gemini_error(exc) and attempt < GEMINI_MAX_RETRIES:
                await _retry_delay(attempt)
                continue
            raise
    raise last_exc or RuntimeError("erro desconhecido")


async def _race_models(
    models: List[str], system_prompt: str, history: List[Dict[str, str]], user_message: str
) -> Tuple[str, str]:
    """Dispara todos os `models` em paralelo. Retorna (modelo_vencedor, resposta) do
    primeiro que responder com sucesso. Se todos falharem, levanta a última exceção."""

    async def _run(model: str) -> Tuple[str, Optional[str], Optional[Exception]]:
        try:
            text = await _try_model_with_retries(model, system_prompt, history, user_message)
            return model, text, None
        except Exception as exc:  # noqa: BLE001
            return model, None, exc

    task_to_model = {asyncio.create_task(_run(m)): m for m in models}
    pending = set(task_to_model.keys())
    last_exc: Optional[Exception] = None
    winner: Optional[Tuple[str, str]] = None

    try:
        while pending and winner is None:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                model, text, exc = d.result()
                if exc is None and text:
                    logger.info("Corrida entre modelos: '%s' respondeu primeiro e venceu.", model)
                    winner = (model, text)
                    break
                else:
                    last_exc = exc
                    logger.warning("Corrida entre modelos: '%s' falhou (%s).", model, exc)
    finally:
        for p in pending:
            p.cancel()

    if winner:
        return winner
    raise last_exc or RuntimeError("Todos os modelos falharam na corrida.")


async def generate_answer(system_prompt: str, history: List[Dict[str, str]], user_message: str) -> Tuple[str, str]:
    """Gera a resposta usando o modelo 'rápido' atual, ou corre todos os candidatos
    em paralelo se ainda não houver um definido / ele tiver falhado. Retorna
    (resposta, nome_do_modelo_usado)."""
    if gemini_client is None:
        raise HTTPException(status_code=500, detail="Gemini não configurado (GEMINI_API_KEY ausente).")

    sticky = _fast_model["name"]
    if sticky:
        try:
            text = await _try_model_with_retries(sticky, system_prompt, history, user_message)
            return (text or "Não consegui gerar uma resposta agora. Pode reformular a pergunta?"), sticky
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Modelo rápido '%s' parou de responder bem (%s); disparando corrida entre modelos de novo.",
                sticky,
                exc,
            )
            async with _fast_model_lock:
                if _fast_model["name"] == sticky:
                    _fast_model["name"] = None

    candidates = _candidate_models()
    try:
        winner_model, text = await _race_models(candidates, system_prompt, history, user_message)
    except Exception as exc:  # noqa: BLE001
        raise _gemini_http_error(exc, "Falha ao gerar a resposta") from exc

    async with _fast_model_lock:
        _fast_model["name"] = winner_model

    return (text or "Não consegui gerar uma resposta agora. Pode reformular a pergunta?"), winner_model


def log_conversation(
    conversation_id: str,
    user_id: str,
    channel: str,
    user_message: str,
    answer: str,
    sources: List[Dict[str, Any]],
) -> None:
    """Grava a conversa no Supabase (memória de conversa e dataset de fine-tuning)."""
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
    if req.new_topic:
        reset_history(req.user_id, req.channel)

    history = await asyncio.to_thread(_history_sync, req.user_id, req.channel)
    query_text = build_retrieval_query(req.message, history)

    query_vector = await embed_text(query_text, "RETRIEVAL_QUERY")
    context_chunks = await search_context(query_vector, top_k=req.top_k)
    system_prompt = build_system_prompt(context_chunks)
    answer, model_used = await generate_answer(system_prompt, history, req.message)

    conversation_id = str(uuid.uuid4())
    await asyncio.to_thread(
        log_conversation, conversation_id, req.user_id, req.channel, req.message, answer, context_chunks
    )

    return ChatResponse(answer=answer, sources=context_chunks, conversation_id=conversation_id, model=model_used)


# ---------------------------------------------------------------------------
# Rotas — API para site / app
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "llm": gemini_client is not None,
        "supabase": supabase is not None,
        "qdrant": qdrant is not None,
        "score_threshold": QDRANT_SCORE_THRESHOLD,
        "preferred_model": await asyncio.to_thread(get_active_model),
        "fast_model": _fast_model["name"],
        "candidate_models": GEMINI_FALLBACK_MODELS,
        "embed_model": GEMINI_EMBED_MODEL,
        "vector_size": QDRANT_VECTOR_SIZE,
        "history_turns": CHAT_HISTORY_TURNS,
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

    def _insert() -> None:
        supabase.table("feedback").insert(  # type: ignore[union-attr]
            {
                "conversation_id": req.conversation_id,
                "rating": req.rating,
                "comment": req.comment,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
        ).execute()

    try:
        await asyncio.to_thread(_insert)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erro ao gravar feedback: {exc}") from exc
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(req: IngestRequest) -> Dict[str, str]:
    """
    Indexa um novo trecho: gera o embedding via Gemini e grava no Qdrant.
    Use isso para popular a base de conhecimento (texto de PDFs, .txt, OCR de fotos).
    Divida textos longos em trechos de ~500 a 2000 caracteres antes de enviar.
    """
    if qdrant is None:
        raise HTTPException(status_code=500, detail="Qdrant não configurado.")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="O texto está vazio.")
    if len(req.text) > MAX_INGEST_CHARS:
        raise HTTPException(
            status_code=422,
            detail=f"Texto com {len(req.text)} caracteres. O limite por trecho é {MAX_INGEST_CHARS}; divida em trechos menores.",
        )

    doc_id = req.doc_id or str(uuid.uuid4())
    try:
        uuid.UUID(doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="doc_id precisa ser um UUID válido.") from exc

    vector = await embed_text(req.text, "RETRIEVAL_DOCUMENT")

    # Os campos reservados vêm por último para os metadados não sobrescreverem text/title/doc_id.
    payload = {**req.metadata, "doc_id": doc_id, "title": req.title, "text": req.text}

    def _upsert() -> None:
        qdrant.upsert(  # type: ignore[union-attr]
            collection_name=QDRANT_COLLECTION,
            points=[qmodels.PointStruct(id=doc_id, vector=vector, payload=payload)],
        )

    try:
        await asyncio.to_thread(_upsert)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Erro ao gravar no Qdrant: {str(exc)[:300]}") from exc
    return {"status": "ok", "doc_id": doc_id}


# ---------------------------------------------------------------------------
# Rotas — Bot Telegram (webhook)
# ---------------------------------------------------------------------------

TELEGRAM_API_BASE = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}" if TELEGRAM_BOT_TOKEN else None
TELEGRAM_MAX_LEN = 4000  # o limite do Telegram é 4096 caracteres por mensagem
NEW_TOPIC_COMMANDS = {"/novo", "/reset", "/limpar", "/start"}


async def send_telegram_message(chat_id: int, text: str) -> None:
    if TELEGRAM_API_BASE is None:
        logger.warning("TELEGRAM_BOT_TOKEN ausente; mensagem não enviada.")
        return
    parts = [text[i : i + TELEGRAM_MAX_LEN] for i in range(0, len(text), TELEGRAM_MAX_LEN)] or [""]
    async with httpx.AsyncClient(timeout=15) as client:
        for part in parts:
            await client.post(f"{TELEGRAM_API_BASE}/sendMessage", json={"chat_id": chat_id, "text": part})


async def handle_telegram_update(update: Dict[str, Any]) -> None:
    message = update.get("message") or update.get("edited_message")
    if not message:
        return

    chat_id = message["chat"]["id"]
    text = message.get("text", "")
    if not text:
        await send_telegram_message(chat_id, "Por enquanto só entendo mensagens de texto.")
        return

    if text.strip().lower().split("@")[0] in NEW_TOPIC_COMMANDS:
        reset_history(str(chat_id), "telegram")
        await send_telegram_message(chat_id, "Certo, vamos começar um assunto novo. Como posso ajudar?")
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
# Painel administrativo (arquivos estáticos em ./painel)
# ---------------------------------------------------------------------------
# Fica no fim de propósito: as rotas da API acima têm prioridade.
# Acesse em  https://SEU-DOMINIO/painel/  (com a barra no final).

PAINEL_DIR = Path(__file__).parent / "painel"

if PAINEL_DIR.is_dir():
    app.mount("/painel", StaticFiles(directory=PAINEL_DIR, html=True), name="painel")
else:
    logger.warning("Pasta do painel não encontrada em %s; /painel não estará disponível.", PAINEL_DIR)


@app.get("/", include_in_schema=False)
async def raiz() -> RedirectResponse:
    """A raiz abre o painel."""
    return RedirectResponse(url="/painel/")


# ---------------------------------------------------------------------------
# Entrypoint local
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)


# ---------------------------------------------------------------------------
# SQL do Supabase (rode uma vez no SQL Editor)
# ---------------------------------------------------------------------------
"""
create table if not exists model_config (
  id int primary key default 1,
  active_model text not null,
  updated_at timestamptz default now(),
  constraint singleton check (id = 1)
);
insert into model_config (id, active_model)
values (1, 'gemini-2.5-flash-lite')
on conflict (id) do update set active_model = excluded.active_model;

create table if not exists conversations (
  id uuid primary key,
  user_id text not null,
  channel text not null default 'web',
  user_message text not null,
  answer text not null,
  sources jsonb default '[]'::jsonb,
  created_at timestamptz default now()
);
create index if not exists conversations_user_idx
  on conversations (user_id, channel, created_at desc);

create table if not exists feedback (
  id bigint generated always as identity primary key,
  conversation_id uuid not null references conversations(id) on delete cascade,
  rating smallint not null check (rating between -1 and 1),
  comment text,
  created_at timestamptz default now()
);
create index if not exists feedback_conv_idx on feedback (conversation_id);

alter table model_config  enable row level security;
alter table conversations enable row level security;
alter table feedback      enable row level security;

notify pgrst, 'reload schema';
"""