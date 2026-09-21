"""
finetune_export.py
==================
Exporta as conversas guardadas no Supabase para um dataset .jsonl no formato
de chat (OpenAI/Together), sobe o arquivo para a Together AI e (opcionalmente)
dispara o job de fine-tuning.

Este módulo é importado por:
    - main.py                     -> from finetune_export import export as export_dataset, upload_and_train, BASE_MODEL
    - auto_learning_pipeline.py   -> mesma coisa

Uso direto pela linha de comando:
    python finetune_export.py                        # exporta tudo para dataset.jsonl
    python finetune_export.py --min-rating 1         # só conversas com feedback positivo
    python finetune_export.py --out meu_dataset.jsonl
    python finetune_export.py --train                # exporta, sobe e dispara o treino
    python finetune_export.py --train --suffix meu-modelo-v1

Variáveis de ambiente (.env):
    SUPABASE_URL             (obrigatória)
    SUPABASE_SERVICE_KEY     (obrigatória)
    TOGETHER_API_KEY         (obrigatória só para upload/treino)
    BASE_MODEL               (opcional, sobrescreve o modelo base padrão)
    FT_SYSTEM_PROMPT         (opcional, system prompt gravado em cada exemplo)
    FT_MIN_EXAMPLES          (opcional, mínimo de exemplos para gerar o dataset)

Esquema esperado da tabela `conversations` (o parser é tolerante):
    - id, created_at
    - mensagens em UM destes formatos:
        a) coluna `messages` (jsonb) = [{"role": "...", "content": "..."}, ...]
        b) colunas separadas: user_message / assistant_message
           (aceita também: pergunta/resposta, question/answer, input/output,
            prompt/completion, user_input/bot_response)
    - feedback/rating opcional em: rating, feedback, score ou vote
"""

import os
import json
import argparse

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Modelo base usado no fine-tuning. Pode ser trocado pelo .env sem mexer no código.
BASE_MODEL = os.getenv(
    "BASE_MODEL",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference",
)

SYSTEM_PROMPT = os.getenv(
    "FT_SYSTEM_PROMPT",
    "Você é um assistente simpático, direto e conversacional. "
    "Responde sempre em português, de forma clara e útil.",
)

MIN_EXAMPLES = int(os.getenv("FT_MIN_EXAMPLES", "10"))

# Nomes de colunas aceitos para os pares pergunta/resposta.
USER_FIELDS = (
    "user_message", "pergunta", "question", "input", "prompt", "user_input", "mensagem"
)
ASSISTANT_FIELDS = (
    "assistant_message", "resposta", "answer", "output", "completion",
    "bot_response", "reply", "response"
)
RATING_FIELDS = ("rating", "feedback", "score", "vote")

PAGE_SIZE = 1000


# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------

def get_supabase():
    if not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
        raise SystemExit(
            "SUPABASE_URL / SUPABASE_SERVICE_KEY não configurados no .env."
        )
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


def fetch_conversations(supabase, min_rating=None) -> list:
    """
    Busca todas as conversas, paginando de PAGE_SIZE em PAGE_SIZE
    (o Supabase limita a resposta por padrão em 1000 linhas).
    """
    rows = []
    start = 0

    while True:
        query = supabase.table("conversations").select("*")

        if min_rating is not None:
            # Tenta filtrar no banco; se a coluna não existir, filtra em memória depois.
            try:
                query = query.gte("rating", min_rating)
            except Exception:  # noqa: BLE001
                pass

        try:
            resp = query.order("created_at", desc=False).range(start, start + PAGE_SIZE - 1).execute()
        except Exception:  # noqa: BLE001
            # Alguma tabela pode não ter created_at — tenta sem ordenação.
            resp = query.range(start, start + PAGE_SIZE - 1).execute()

        batch = resp.data or []
        rows.extend(batch)

        if len(batch) < PAGE_SIZE:
            break
        start += PAGE_SIZE

    return rows


# ---------------------------------------------------------------------------
# Conversão linha -> exemplo de treino
# ---------------------------------------------------------------------------

def _first_field(row: dict, candidates) -> str:
    for name in candidates:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def get_rating(row: dict):
    for name in RATING_FIELDS:
        if name in row and row[name] is not None:
            try:
                return int(row[name])
            except (TypeError, ValueError):
                continue
    return None


def row_to_messages(row: dict) -> list:
    """
    Converte uma linha da tabela em uma lista de mensagens no formato de chat.
    Retorna [] se a linha não tiver conteúdo aproveitável.
    """
    raw = row.get("messages")

    # Formato A: coluna jsonb com a conversa inteira
    if raw:
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = None

        if isinstance(raw, list):
            messages = []
            for item in raw:
                if not isinstance(item, dict):
                    continue
                role = item.get("role")
                content = item.get("content")
                if role in ("system", "user", "assistant") and isinstance(content, str) and content.strip():
                    messages.append({"role": role, "content": content.strip()})

            # Precisa ter pelo menos um turno user + assistant para servir de treino
            roles = {m["role"] for m in messages}
            if "user" in roles and "assistant" in roles:
                if not any(m["role"] == "system" for m in messages):
                    messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                return messages

    # Formato B: colunas separadas de pergunta e resposta
    user_text = _first_field(row, USER_FIELDS)
    assistant_text = _first_field(row, ASSISTANT_FIELDS)

    if user_text and assistant_text:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]

    return []


# ---------------------------------------------------------------------------
# export()  <- usado pelo main.py e pelo auto_learning_pipeline.py
# ---------------------------------------------------------------------------

def export(min_rating=None, out_path: str = "dataset.jsonl"):
    """
    Exporta as conversas para um arquivo .jsonl.

    Args:
        min_rating: se informado (-1, 0 ou 1), só inclui conversas com
                    feedback maior ou igual a esse valor.
        out_path:   caminho do arquivo .jsonl a ser gerado.

    Returns:
        O caminho do arquivo gerado, ou None se não houver exemplos suficientes.
    """
    supabase = get_supabase()

    if min_rating in ("", None):
        min_rating = None
    else:
        min_rating = int(min_rating)

    rows = fetch_conversations(supabase, min_rating)
    print(f"Conversas encontradas no Supabase: {len(rows)}")

    examples = []
    descartadas = 0

    for row in rows:
        if min_rating is not None:
            rating = get_rating(row)
            if rating is None or rating < min_rating:
                descartadas += 1
                continue

        messages = row_to_messages(row)
        if not messages:
            descartadas += 1
            continue

        examples.append({"messages": messages})

    print(f"Exemplos válidos: {len(examples)} (descartadas: {descartadas})")

    if len(examples) < MIN_EXAMPLES:
        print(
            f"Exemplos insuficientes para treinar (mínimo: {MIN_EXAMPLES}). "
            "Nenhum arquivo foi gerado."
        )
        return None

    with open(out_path, "w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Dataset gravado em: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# upload_and_train()  <- usado pelo auto_learning_pipeline.py
# ---------------------------------------------------------------------------

def upload_and_train(path: str, do_train: bool = True, suffix: str = None, n_epochs: int = 3):
    """
    Sobe o dataset para a Together AI e, se do_train=True, já dispara o treino.

    Args:
        path:     caminho do .jsonl gerado por export().
        do_train: se False, apenas sobe o arquivo e devolve o file_id
                  (é assim que o auto_learning_pipeline.py chama esta função).
        suffix:   sufixo/nome do modelo resultante.
        n_epochs: número de épocas do treino.

    Returns:
        file_id (str) se do_train=False; caso contrário, o job de fine-tuning.
    """
    if not TOGETHER_API_KEY:
        raise SystemExit("TOGETHER_API_KEY não configurado no .env.")

    if not path or not os.path.exists(path):
        raise SystemExit(f"Arquivo de dataset não encontrado: {path}")

    from together import Together

    client = Together(api_key=TOGETHER_API_KEY)

    print(f"Subindo {path} para a Together AI...")
    uploaded = client.files.upload(file=path, check=True)

    file_id = getattr(uploaded, "id", None)
    if file_id is None and isinstance(uploaded, dict):
        file_id = uploaded.get("id")
    if not file_id:
        raise RuntimeError(f"Upload não retornou um id de arquivo: {uploaded!r}")

    print(f"Arquivo enviado. file_id: {file_id}")

    if not do_train:
        return file_id

    print(f"Disparando fine-tuning sobre o modelo base: {BASE_MODEL}")
    job = client.fine_tuning.create(
        training_file=file_id,
        model=BASE_MODEL,
        n_epochs=n_epochs,
        suffix=suffix or "assistente-ft",
    )

    job_id = getattr(job, "id", None) or (job.get("id") if isinstance(job, dict) else None)
    print(f"Job de fine-tuning criado: {job_id}")
    print("Acompanhe o progresso no painel da Together AI ou pelo auto_learning_pipeline.py.")
    return job


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exporta conversas do Supabase para dataset de fine-tuning."
    )
    parser.add_argument("--min-rating", default=None,
                        help="Só exporta conversas com feedback >= este valor (-1, 0, 1).")
    parser.add_argument("--out", default="dataset.jsonl",
                        help="Caminho do arquivo .jsonl de saída.")
    parser.add_argument("--train", action="store_true",
                        help="Depois de exportar, sobe o arquivo e dispara o treino.")
    parser.add_argument("--suffix", default=None,
                        help="Sufixo/nome do modelo resultante.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Número de épocas do treino.")
    args = parser.parse_args()

    path = export(args.min_rating, args.out)
    if not path:
        return

    if args.train:
        upload_and_train(path, do_train=True, suffix=args.suffix, n_epochs=args.epochs)
    else:
        print("Para treinar, rode de novo com --train (ou use o auto_learning_pipeline.py).")


if __name__ == "__main__":
    main()
