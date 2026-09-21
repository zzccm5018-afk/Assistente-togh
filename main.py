"""
auto_learning_pipeline.py
==========================
Automatiza o ciclo de "aprendizado contínuo" do assistente:

    1. Verifica se há conversas novas o suficiente desde o último treino.
    2. Exporta um dataset .jsonl a partir do Supabase (reaproveita a lógica
       de finetune_export.py).
    3. Sobe o dataset e dispara um job de fine-tuning na Together AI.
    4. Espera o job terminar (polling).
    5. Roda um sanity check simples no modelo novo antes de publicá-lo.
    6. Se passar no sanity check, atualiza a tabela `model_config` no
       Supabase — o main.py detecta a troca sozinho (cache de alguns
       minutos), SEM precisar reiniciar o servidor.
    7. Se falhar em qualquer etapa, não publica nada e o modelo em produção
       continua o mesmo (fail-safe).

Isso é pensado para ser chamado periodicamente por um agendador externo:
    - cron (ex: toda semana)
    - GitHub Actions (workflow agendado, "schedule: cron")
    - qualquer scheduler do seu provedor de hospedagem

Pré-requisitos:
    - Tudo que finetune_export.py já precisa (SUPABASE_URL, SUPABASE_SERVICE_KEY,
      TOGETHER_API_KEY).
    - Tabela `model_config` criada no Supabase (veja instrução no topo do main.py).
    - Opcional: tabela `training_runs` para manter histórico dos treinos
      (criada automaticamente na primeira execução via SQL manual — ver nota
      abaixo, o script tenta gravar mas não falha se a tabela não existir).

Uso:
    python auto_learning_pipeline.py                     # roda o ciclo completo
    python auto_learning_pipeline.py --dry-run            # só mostra o que faria
    python auto_learning_pipeline.py --min-new-examples 50
    python auto_learning_pipeline.py --min-rating 1 --poll-interval 60

Tabela opcional de histórico (recomendado criar):
    create table training_runs (
        id uuid primary key default gen_random_uuid(),
        started_at timestamptz default now(),
        finished_at timestamptz,
        status text,                 -- 'success' | 'failed' | 'skipped'
        examples_count int,
        together_job_id text,
        model_id text,
        notes text
    );
"""

import os
import time
import argparse
import uuid as uuid_lib
from datetime import datetime, timezone

from dotenv import load_dotenv
from supabase import create_client

# Reaproveita a lógica de exportação já existente
from finetune_export import export as export_dataset, upload_and_train, BASE_MODEL

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

DEFAULT_MIN_NEW_EXAMPLES = int(os.getenv("AUTO_FT_MIN_NEW_EXAMPLES", "50"))
DEFAULT_MIN_RATING = os.getenv("AUTO_FT_MIN_RATING")  # pode vir vazio
DEFAULT_POLL_INTERVAL = int(os.getenv("AUTO_FT_POLL_INTERVAL_SECONDS", "60"))
DEFAULT_POLL_TIMEOUT = int(os.getenv("AUTO_FT_POLL_TIMEOUT_SECONDS", "3600"))  # 1h


def get_supabase():
    if not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
        raise SystemExit("SUPABASE_URL / SUPABASE_SERVICE_KEY não configurados no .env.")
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ---------------------------------------------------------------------------
# Etapa 1: decidir se vale a pena treinar agora
# ---------------------------------------------------------------------------

def count_new_conversations_since_last_run(supabase) -> int:
    """
    Conta quantas conversas existem desde o último treino bem-sucedido
    (baseado na tabela training_runs, se existir). Se a tabela não existir
    ou não houver treino anterior, conta todas as conversas.
    """
    since = None
    try:
        resp = (
            supabase.table("training_runs")
            .select("finished_at")
            .eq("status", "success")
            .order("finished_at", desc=True)
            .limit(1)
            .execute()
        )
        if resp.data:
            since = resp.data[0]["finished_at"]
    except Exception:
        pass  # tabela pode não existir ainda — segue sem filtro de data

    query = supabase.table("conversations").select("id", count="exact")
    if since:
        query = query.gt("created_at", since)
    resp = query.execute()
    return resp.count or 0


def log_training_run(supabase, run: dict) -> None:
    try:
        supabase.table("training_runs").insert(run).execute()
    except Exception as exc:  # noqa: BLE001
        print(f"[aviso] Não foi possível gravar em training_runs (tabela existe?): {exc}")


# ---------------------------------------------------------------------------
# Etapa 4: aguardar o job de fine-tuning terminar
# ---------------------------------------------------------------------------

def wait_for_job(together_client, job_id: str, poll_interval: int, timeout: int) -> str:
    """
    Faz polling do status do job até ele terminar (sucesso, erro ou timeout).
    Retorna: 'completed', 'failed', 'cancelled' ou 'timeout'.
    """
    elapsed = 0
    while elapsed < timeout:
        job = together_client.fine_tuning.retrieve(job_id)
        status = getattr(job, "status", None) or job.get("status")
        print(f"  status do job {job_id}: {status} (elapsed={elapsed}s)")

        if status in ("completed", "succeeded"):
            return "completed"
        if status in ("failed", "error"):
            return "failed"
        if status in ("cancelled", "canceled"):
            return "cancelled"

        time.sleep(poll_interval)
        elapsed += poll_interval

    return "timeout"


def extract_output_model_name(together_client, job_id: str) -> str:
    job = together_client.fine_tuning.retrieve(job_id)
    model_name = getattr(job, "output_name", None) or (job.get("output_name") if isinstance(job, dict) else None)
    if not model_name:
        raise RuntimeError(
            f"Job {job_id} terminou mas não retornou output_name. "
            "Confira manualmente no painel da Together AI e atualize model_config à mão."
        )
    return model_name


# ---------------------------------------------------------------------------
# Etapa 5: sanity check simples antes de publicar
# ---------------------------------------------------------------------------

def sanity_check(together_client, model_name: str) -> bool:
    """
    Testa o modelo novo com algumas perguntas simples antes de publicar.
    É deliberadamente básico: só confirma que o modelo responde de forma
    coerente e não quebra. Não substitui uma avaliação de qualidade real.
    """
    test_prompts = [
        "Oi, tudo bem?",
        "Obrigado pela ajuda!",
    ]
    try:
        for prompt in test_prompts:
            resp = together_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "Você é um assistente simpático e conversacional."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
                temperature=0.5,
            )
            answer = resp.choices[0].message.content
            if not answer or len(answer.strip()) < 2:
                print(f"  [sanity check] resposta vazia/curta demais para: {prompt!r}")
                return False
            print(f"  [sanity check] '{prompt}' -> '{answer[:80]}...'")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [sanity check] erro ao testar o modelo novo: {exc}")
        return False


# ---------------------------------------------------------------------------
# Etapa 6: publicar o modelo novo (hot-swap sem reiniciar o main.py)
# ---------------------------------------------------------------------------

def publish_model(supabase, model_name: str) -> None:
    supabase.table("model_config").upsert(
        {
            "id": 1,
            "active_model": model_name,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
    ).execute()
    print(f"Modelo publicado em model_config: {model_name}")
    print("O main.py vai detectar a troca automaticamente em até "
          f"{os.getenv('MODEL_CACHE_TTL_SECONDS', '300')}s (sem precisar reiniciar).")


# ---------------------------------------------------------------------------
# Orquestração
# ---------------------------------------------------------------------------

def run_pipeline(
    min_new_examples: int,
    min_rating,
    poll_interval: int,
    poll_timeout: int,
    dry_run: bool,
    suffix: str,
) -> None:
    supabase = get_supabase()
    run_id = str(uuid_lib.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()

    print("=== Pipeline de aprendizado contínuo ===")
    print(f"run_id: {run_id}")

    new_count = count_new_conversations_since_last_run(supabase)
    print(f"Conversas novas desde o último treino bem-sucedido: {new_count}")

    if new_count < min_new_examples:
        print(f"Menos que o mínimo exigido ({min_new_examples}). Pulando ciclo.")
        log_training_run(supabase, {
            "id": run_id, "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "skipped", "examples_count": new_count,
            "notes": "abaixo do mínimo de exemplos novos",
        })
        return

    if dry_run:
        print("[--dry-run] Pararia aqui: exportaria, treinaria, validaria e publicaria.")
        return

    if not TOGETHER_API_KEY:
        raise SystemExit("TOGETHER_API_KEY não configurado no .env.")

    from together import Together
    together_client = Together(api_key=TOGETHER_API_KEY)

    # 2) exportar dataset
    dataset_path = f"dataset_{run_id}.jsonl"
    rating_filter = int(min_rating) if min_rating not in (None, "") else None
    path = export_dataset(rating_filter, dataset_path)
    if not path:
        log_training_run(supabase, {
            "id": run_id, "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "skipped", "examples_count": 0,
            "notes": "export não gerou exemplos",
        })
        return

    # 3) upload + disparo do treino
    job_suffix = suffix or f"autoft-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}"
    file_id = upload_and_train(path, do_train=False)  # só sobe o arquivo aqui
    print("Disparando job de fine-tuning...")
    job = together_client.fine_tuning.create(
        training_file=file_id,
        model=BASE_MODEL,
        n_epochs=3,
        suffix=job_suffix,
    )
    job_id = getattr(job, "id", None) or job["id"]
    print(f"Job criado: {job_id}")

    # 4) esperar terminar
    result = wait_for_job(together_client, job_id, poll_interval, poll_timeout)

    if result != "completed":
        print(f"Treino não completou com sucesso (status final: {result}). Modelo em produção NÃO foi alterado.")
        log_training_run(supabase, {
            "id": run_id, "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed", "examples_count": new_count,
            "together_job_id": job_id, "notes": f"status final: {result}",
        })
        return

    model_name = extract_output_model_name(together_client, job_id)
    print(f"Treino concluído. Modelo gerado: {model_name}")

    # 5) sanity check antes de publicar
    if not sanity_check(together_client, model_name):
        print("Sanity check falhou. Modelo em produção NÃO foi alterado.")
        log_training_run(supabase, {
            "id": run_id, "started_at": started_at,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "status": "failed", "examples_count": new_count,
            "together_job_id": job_id, "model_id": model_name,
            "notes": "reprovado no sanity check",
        })
        return

    # 6) publicar (hot-swap)
    publish_model(supabase, model_name)
    log_training_run(supabase, {
        "id": run_id, "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "status": "success", "examples_count": new_count,
        "together_job_id": job_id, "model_id": model_name,
        "notes": "publicado com sucesso",
    })
    print("Ciclo concluído com sucesso.")


def main():
    parser = argparse.ArgumentParser(description="Pipeline automático de aprendizado contínuo.")
    parser.add_argument("--min-new-examples", type=int, default=DEFAULT_MIN_NEW_EXAMPLES,
                         help="Mínimo de conversas novas para justificar um novo treino.")
    parser.add_argument("--min-rating", default=DEFAULT_MIN_RATING,
                         help="Só usa conversas com feedback >= este valor no dataset (-1, 0, 1).")
    parser.add_argument("--poll-interval", type=int, default=DEFAULT_POLL_INTERVAL,
                         help="Segundos entre verificações de status do treino.")
    parser.add_argument("--poll-timeout", type=int, default=DEFAULT_POLL_TIMEOUT,
                         help="Tempo máximo de espera pelo treino (segundos).")
    parser.add_argument("--dry-run", action="store_true",
                         help="Só mostra o que faria, sem treinar nem publicar nada.")
    parser.add_argument("--suffix", default=None, help="Nome/sufixo do modelo resultante.")
    args = parser.parse_args()

    run_pipeline(
        min_new_examples=args.min_new_examples,
        min_rating=args.min_rating,
        poll_interval=args.poll_interval,
        poll_timeout=args.poll_timeout,
        dry_run=args.dry_run,
        suffix=args.suffix,
    )


if __name__ == "__main__":
    main()