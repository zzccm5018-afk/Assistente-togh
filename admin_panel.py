"""
admin_panel.py
==============
Painel administrativo para o ciclo de aprendizado contínuo
(auto_learning_pipeline.py + finetune_export.py + main.py).

O que ele cobre
---------------
1.  Visão geral      : modelo em produção, conversas totais/novas, último treino,
                       saúde das integrações (Supabase / Together / tabelas).
2.  Conversas        : listar, filtrar por nota, buscar, ver a conversa inteira,
                       corrigir a nota (curadoria) e excluir.
3.  Arquivos         : exportar dataset .jsonl, subir .jsonl pronto do seu
                       computador, pré-visualizar, validar formato, baixar e excluir.
4.  Treinos          : enviar o arquivo para a Together, disparar o fine-tuning com
                       os hiperparâmetros escolhidos, acompanhar status e cancelar.
5.  Modelos          : listar modelos treinados, rodar o sanity check, publicar em
                       produção (hot-swap via model_config) e voltar para o anterior.
6.  Pipeline         : rodar o ciclo completo (ou em modo de teste) com os logs
                       aparecendo ao vivo na tela.
7.  Histórico        : tabela training_runs com todos os ciclos já executados.

Como rodar
----------
    pip install fastapi uvicorn python-multipart python-dotenv supabase together
    python admin_panel.py                 # sobe em http://localhost:8800/admin

Ou acoplado ao seu main.py já existente:

    from admin_panel import router as admin_router
    app.include_router(admin_router)      # painel vai para /admin

Variáveis de ambiente (.env)
----------------------------
    SUPABASE_URL, SUPABASE_SERVICE_KEY, TOGETHER_API_KEY   (as que você já usa)
    ADMIN_TOKEN            senha de acesso ao painel. Se ficar em branco, o painel
                           gera uma e mostra no terminal ao iniciar.
    ADMIN_DATASETS_DIR     pasta dos .jsonl (padrão: ./datasets)
    ADMIN_PORT             porta quando rodado sozinho (padrão: 8800)
"""

from __future__ import annotations

import io
import json
import os
import re
import secrets
import threading
import time
import traceback
import uuid as uuid_lib
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

load_dotenv()

# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

DATASETS_DIR = Path(os.getenv("ADMIN_DATASETS_DIR", "./datasets")).resolve()
DATASETS_DIR.mkdir(parents=True, exist_ok=True)

STATIC_DIR = Path(__file__).parent / "admin_static"

ADMIN_TOKEN = os.getenv("ADMIN_TOKEN") or secrets.token_urlsafe(18)
_TOKEN_WAS_GENERATED = not os.getenv("ADMIN_TOKEN")

MAX_UPLOAD_BYTES = int(os.getenv("ADMIN_MAX_UPLOAD_MB", "200")) * 1024 * 1024
SAFE_NAME = re.compile(r"^[A-Za-z0-9._-]{1,180}$")

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Clientes (carregados sob demanda, para o painel abrir mesmo com algo faltando)
# ---------------------------------------------------------------------------

_supabase = None
_together = None


def get_supabase():
    global _supabase
    if _supabase is None:
        if not (SUPABASE_URL and SUPABASE_SERVICE_KEY):
            raise HTTPException(503, "SUPABASE_URL ou SUPABASE_SERVICE_KEY não estão no .env.")
        from supabase import create_client

        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    return _supabase


def get_together():
    global _together
    if _together is None:
        if not TOGETHER_API_KEY:
            raise HTTPException(503, "TOGETHER_API_KEY não está no .env.")
        from together import Together

        _together = Together(api_key=TOGETHER_API_KEY)
    return _together


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def as_dict(obj: Any) -> dict:
    """Together devolve ora objeto, ora dict. Normaliza para dict."""
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    for attr in ("model_dump", "dict"):
        if hasattr(obj, attr):
            try:
                return getattr(obj, attr)()
            except Exception:
                pass
    return {k: v for k, v in vars(obj).items() if not k.startswith("_")}


def safe_path(name: str) -> Path:
    if not SAFE_NAME.match(name or ""):
        raise HTTPException(400, "Nome de arquivo inválido.")
    path = (DATASETS_DIR / name).resolve()
    if DATASETS_DIR not in path.parents:
        raise HTTPException(400, "Caminho fora da pasta de datasets.")
    return path


# ---------------------------------------------------------------------------
# Autenticação (token único de administrador)
# ---------------------------------------------------------------------------

def require_admin(request: Request) -> bool:
    sent = request.headers.get("x-admin-token") or request.cookies.get("admin_token") or ""
    if not secrets.compare_digest(sent, ADMIN_TOKEN):
        raise HTTPException(401, "Token de administrador inválido.")
    return True


Auth = Depends(require_admin)


@router.post("/api/login")
async def login(request: Request):
    body = await request.json()
    token = (body or {}).get("token", "")
    if not secrets.compare_digest(token, ADMIN_TOKEN):
        raise HTTPException(401, "Token incorreto.")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(
        "admin_token", token, httponly=True, samesite="lax",
        secure=os.getenv("ADMIN_COOKIE_SECURE", "0") == "1", max_age=60 * 60 * 12,
    )
    return resp


@router.post("/api/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("admin_token")
    return resp


# ---------------------------------------------------------------------------
# Tarefas em segundo plano com log ao vivo
# ---------------------------------------------------------------------------

class Task:
    def __init__(self, kind: str):
        self.id = str(uuid_lib.uuid4())[:8]
        self.kind = kind
        self.status = "running"          # running | success | error
        self.started_at = now_iso()
        self.finished_at: Optional[str] = None
        self.lines: list[str] = []
        self.result: Any = None
        self._lock = threading.Lock()

    def log(self, text: str) -> None:
        with self._lock:
            for line in str(text).rstrip().splitlines() or [""]:
                self.lines.append(f"{datetime.now().strftime('%H:%M:%S')}  {line}")

    def snapshot(self, since: int = 0) -> dict:
        with self._lock:
            return {
                "id": self.id, "kind": self.kind, "status": self.status,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "result": self.result, "total_lines": len(self.lines),
                "lines": self.lines[since:],
            }


TASKS: dict[str, Task] = {}


class _StreamToTask(io.TextIOBase):
    """Redireciona os print() do pipeline para o buffer da tarefa."""

    def __init__(self, task: Task):
        self.task = task
        self.buf = ""

    def write(self, s: str) -> int:
        self.buf += s
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            self.task.log(line)
        return len(s)

    def flush(self) -> None:
        if self.buf:
            self.task.log(self.buf)
            self.buf = ""


def run_in_task(kind: str, fn) -> Task:
    """Executa fn(task) numa thread, capturando stdout/stderr no log da tarefa."""
    task = Task(kind)
    TASKS[task.id] = task

    def _runner():
        stream = _StreamToTask(task)
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                task.result = fn(task)
            stream.flush()
            task.status = "success"
        except Exception as exc:  # noqa: BLE001
            stream.flush()
            task.log(f"ERRO: {exc}")
            task.log(traceback.format_exc())
            task.status = "error"
            task.result = {"error": str(exc)}
        finally:
            task.finished_at = now_iso()
            # mantém só as 40 tarefas mais recentes na memória
            if len(TASKS) > 40:
                for old in sorted(TASKS.values(), key=lambda t: t.started_at)[:-40]:
                    TASKS.pop(old.id, None)

    threading.Thread(target=_runner, daemon=True).start()
    return task


@router.get("/api/tasks", dependencies=[Auth])
def list_tasks():
    items = sorted(TASKS.values(), key=lambda t: t.started_at, reverse=True)
    return [
        {"id": t.id, "kind": t.kind, "status": t.status,
         "started_at": t.started_at, "finished_at": t.finished_at}
        for t in items
    ]


@router.get("/api/tasks/{task_id}", dependencies=[Auth])
def get_task(task_id: str, since: int = 0):
    task = TASKS.get(task_id)
    if not task:
        raise HTTPException(404, "Tarefa não encontrada (ou já saiu da memória).")
    return task.snapshot(since)


# ---------------------------------------------------------------------------
# Visão geral
# ---------------------------------------------------------------------------

def _active_model() -> Optional[dict]:
    try:
        resp = get_supabase().table("model_config").select("*").eq("id", 1).limit(1).execute()
        return resp.data[0] if resp.data else None
    except Exception:
        return None


def _count(table: str, **filters) -> Optional[int]:
    try:
        q = get_supabase().table(table).select("id", count="exact")
        for op, col, val in filters.get("conds", []):
            q = getattr(q, op)(col, val)
        return q.execute().count or 0
    except Exception:
        return None


@router.get("/api/overview", dependencies=[Auth])
def overview():
    sb_ok, sb_err = True, None
    try:
        get_supabase()
    except HTTPException as exc:
        sb_ok, sb_err = False, exc.detail

    data: dict[str, Any] = {
        "generated_at": now_iso(),
        "integrations": {
            "supabase": {"ok": sb_ok, "detail": sb_err},
            "together": {"ok": bool(TOGETHER_API_KEY),
                         "detail": None if TOGETHER_API_KEY else "TOGETHER_API_KEY ausente"},
        },
        "active_model": None,
        "conversations_total": None,
        "conversations_new": None,
        "last_run": None,
        "runs_recent": [],
        "tables": {"model_config": False, "training_runs": False, "conversations": False},
        "datasets_count": len(list(DATASETS_DIR.glob("*.jsonl"))),
        "cache_ttl": os.getenv("MODEL_CACHE_TTL_SECONDS", "300"),
    }
    if not sb_ok:
        return data

    sb = get_supabase()

    cfg = _active_model()
    if cfg:
        data["active_model"] = cfg
        data["tables"]["model_config"] = True

    try:
        data["conversations_total"] = sb.table("conversations").select("id", count="exact").execute().count or 0
        data["tables"]["conversations"] = True
    except Exception:
        pass

    since = None
    try:
        resp = (sb.table("training_runs").select("*")
                .order("started_at", desc=True).limit(10).execute())
        data["tables"]["training_runs"] = True
        data["runs_recent"] = resp.data or []
        for run in resp.data or []:
            if run.get("status") == "success" and run.get("finished_at"):
                data["last_run"] = run
                since = run["finished_at"]
                break
    except Exception:
        pass

    try:
        q = sb.table("conversations").select("id", count="exact")
        if since:
            q = q.gt("created_at", since)
        data["conversations_new"] = q.execute().count or 0
    except Exception:
        pass

    return data


# ---------------------------------------------------------------------------
# Conversas
# ---------------------------------------------------------------------------

@router.get("/api/conversations", dependencies=[Auth])
def list_conversations(
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
    rating: Optional[str] = None,
    search: Optional[str] = None,
    order: str = "created_at",
):
    sb = get_supabase()
    q = sb.table("conversations").select("*", count="exact")
    if rating not in (None, "", "todas"):
        try:
            q = q.eq("rating", int(rating))
        except ValueError:
            pass
    if search:
        cleaned = search.replace(",", " ").strip()
        try:
            q = q.or_(f"user_message.ilike.%{cleaned}%,assistant_message.ilike.%{cleaned}%")
        except Exception:
            pass
    try:
        q = q.order(order, desc=True)
    except Exception:
        pass
    resp = q.range(offset, offset + limit - 1).execute()
    return {"items": resp.data or [], "total": resp.count or 0, "limit": limit, "offset": offset}


@router.get("/api/conversations/{conv_id}", dependencies=[Auth])
def get_conversation(conv_id: str):
    resp = get_supabase().table("conversations").select("*").eq("id", conv_id).limit(1).execute()
    if not resp.data:
        raise HTTPException(404, "Conversa não encontrada.")
    return resp.data[0]


@router.patch("/api/conversations/{conv_id}", dependencies=[Auth])
async def update_conversation(conv_id: str, request: Request):
    """Curadoria: corrigir a nota de uma conversa antes de ela entrar no dataset."""
    body = await request.json()
    patch = {k: v for k, v in (body or {}).items() if k in ("rating", "notes", "tags")}
    if not patch:
        raise HTTPException(400, "Nada para atualizar. Campos aceitos: rating, notes, tags.")
    resp = get_supabase().table("conversations").update(patch).eq("id", conv_id).execute()
    return {"ok": True, "updated": resp.data}


@router.delete("/api/conversations/{conv_id}", dependencies=[Auth])
def delete_conversation(conv_id: str):
    get_supabase().table("conversations").delete().eq("id", conv_id).execute()
    return {"ok": True}


@router.post("/api/conversations/bulk-delete", dependencies=[Auth])
async def bulk_delete(request: Request):
    body = await request.json()
    ids = (body or {}).get("ids") or []
    if not ids:
        raise HTTPException(400, "Envie a lista de ids.")
    get_supabase().table("conversations").delete().in_("id", ids).execute()
    return {"ok": True, "deleted": len(ids)}


# ---------------------------------------------------------------------------
# Arquivos / datasets
# ---------------------------------------------------------------------------

def _file_info(path: Path) -> dict:
    stat = path.stat()
    lines = 0
    try:
        with path.open("rb") as fh:
            lines = sum(1 for line in fh if line.strip())
    except Exception:
        pass
    return {
        "name": path.name,
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        "examples": lines,
    }


@router.get("/api/files", dependencies=[Auth])
def list_files():
    files = [_file_info(p) for p in DATASETS_DIR.glob("*.jsonl") if p.is_file()]
    files.sort(key=lambda f: f["modified_at"], reverse=True)
    return {"dir": str(DATASETS_DIR), "files": files}


@router.post("/api/files/upload", dependencies=[Auth])
async def upload_file(file: UploadFile = File(...), overwrite: bool = Form(False)):
    name = Path(file.filename or "").name
    if not name.endswith(".jsonl"):
        raise HTTPException(400, "Envie um arquivo .jsonl (uma conversa por linha).")
    path = safe_path(name)
    if path.exists() and not overwrite:
        raise HTTPException(409, f"Já existe um arquivo chamado {name}. Marque 'substituir' para trocar.")

    written = 0
    with path.open("wb") as out:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                out.close()
                path.unlink(missing_ok=True)
                raise HTTPException(413, f"Arquivo maior que o limite de {MAX_UPLOAD_BYTES // (1024*1024)} MB.")
            out.write(chunk)

    return {"ok": True, "file": _file_info(path), "validation": _validate(path)}


@router.get("/api/files/{name}/download", dependencies=[Auth])
def download_file(name: str):
    path = safe_path(name)
    if not path.exists():
        raise HTTPException(404, "Arquivo não encontrado.")
    return FileResponse(path, filename=name, media_type="application/x-ndjson")


@router.get("/api/files/{name}/preview", dependencies=[Auth])
def preview_file(name: str, limit: int = Query(5, ge=1, le=50)):
    path = safe_path(name)
    if not path.exists():
        raise HTTPException(404, "Arquivo não encontrado.")
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                rows.append({"_erro": f"linha inválida: {exc}"})
            if len(rows) >= limit:
                break
    return {"name": name, "rows": rows}


def _validate(path: Path) -> dict:
    """Confere se o .jsonl está no formato de chat esperado pela Together."""
    total, ok, problems = 0, 0, []
    roles = {"system", "user", "assistant", "tool"}
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            total += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                if len(problems) < 20:
                    problems.append(f"linha {i}: JSON inválido ({exc.msg})")
                continue
            msgs = obj.get("messages")
            if not isinstance(msgs, list) or not msgs:
                if len(problems) < 20:
                    problems.append(f"linha {i}: falta a lista 'messages'")
                continue
            bad = False
            for m in msgs:
                if not isinstance(m, dict) or m.get("role") not in roles or not isinstance(m.get("content"), str):
                    bad = True
                    break
            if bad:
                if len(problems) < 20:
                    problems.append(f"linha {i}: alguma mensagem sem 'role' válido ou sem 'content' em texto")
                continue
            if not any(m["role"] == "assistant" for m in msgs):
                if len(problems) < 20:
                    problems.append(f"linha {i}: nenhuma resposta do assistente")
                continue
            ok += 1
    return {"total": total, "valid": ok, "invalid": total - ok, "problems": problems}


@router.post("/api/files/{name}/validate", dependencies=[Auth])
def validate_file(name: str):
    path = safe_path(name)
    if not path.exists():
        raise HTTPException(404, "Arquivo não encontrado.")
    return _validate(path)


@router.delete("/api/files/{name}", dependencies=[Auth])
def delete_file(name: str):
    path = safe_path(name)
    if not path.exists():
        raise HTTPException(404, "Arquivo não encontrado.")
    path.unlink()
    return {"ok": True}


@router.post("/api/files/export", dependencies=[Auth])
async def export_dataset_endpoint(request: Request):
    """Gera um novo .jsonl a partir do Supabase usando o finetune_export.py."""
    body = await request.json() if await request.body() else {}
    min_rating = (body or {}).get("min_rating")
    name = (body or {}).get("name") or f"dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    if not name.endswith(".jsonl"):
        name += ".jsonl"
    path = safe_path(name)

    def job(task: Task):
        from finetune_export import export as export_dataset

        rating = int(min_rating) if min_rating not in (None, "", "todas") else None
        task.log(f"Exportando conversas do Supabase (nota mínima: {rating if rating is not None else 'sem filtro'})")
        result = export_dataset(rating, str(path))
        if not result or not path.exists():
            raise RuntimeError("A exportação não gerou nenhum exemplo.")
        info = _file_info(path)
        task.log(f"Arquivo criado: {info['name']} — {info['examples']} exemplos")
        validation = _validate(path)
        task.log(f"Validação: {validation['valid']} válidos, {validation['invalid']} com problema")
        return {"file": info, "validation": validation}

    return {"task_id": run_in_task("export", job).id}


# ---------------------------------------------------------------------------
# Treinos (Together AI)
# ---------------------------------------------------------------------------

def _default_base_model() -> str:
    try:
        from finetune_export import BASE_MODEL

        return BASE_MODEL
    except Exception:
        return os.getenv("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference")


@router.get("/api/training/defaults", dependencies=[Auth])
def training_defaults():
    return {
        "base_model": _default_base_model(),
        "n_epochs": 3,
        "learning_rate": 1e-5,
        "batch_size": None,
        "suffix": f"ft-{datetime.now().strftime('%Y%m%d%H%M')}",
    }


@router.post("/api/training/start", dependencies=[Auth])
async def start_training(request: Request):
    """Sobe o arquivo escolhido para a Together e dispara o fine-tuning."""
    body = await request.json()
    name = (body or {}).get("file")
    if not name:
        raise HTTPException(400, "Escolha o arquivo de treino.")
    path = safe_path(name)
    if not path.exists():
        raise HTTPException(404, "Arquivo não encontrado.")

    validation = _validate(path)
    if validation["valid"] == 0:
        raise HTTPException(400, "Esse arquivo não tem nenhum exemplo válido. Valide antes de treinar.")
    if validation["invalid"] and not (body or {}).get("ignore_invalid"):
        raise HTTPException(
            400,
            f"{validation['invalid']} linha(s) com problema. Corrija o arquivo ou marque "
            "'treinar mesmo assim'.",
        )

    base_model = (body or {}).get("base_model") or _default_base_model()
    n_epochs = int((body or {}).get("n_epochs") or 3)
    suffix = (body or {}).get("suffix") or f"ft-{datetime.now().strftime('%Y%m%d%H%M')}"
    learning_rate = (body or {}).get("learning_rate")
    batch_size = (body or {}).get("batch_size")

    def job(task: Task):
        client = get_together()
        task.log(f"Enviando {path.name} ({_file_info(path)['examples']} exemplos) para a Together…")
        uploaded = as_dict(client.files.upload(file=str(path), check=True))
        file_id = uploaded.get("id")
        task.log(f"Arquivo recebido: {file_id}")

        kwargs: dict[str, Any] = {
            "training_file": file_id, "model": base_model,
            "n_epochs": n_epochs, "suffix": suffix,
        }
        if learning_rate:
            kwargs["learning_rate"] = float(learning_rate)
        if batch_size:
            kwargs["batch_size"] = int(batch_size)

        task.log(f"Criando o treino sobre {base_model} ({n_epochs} épocas)…")
        created = as_dict(client.fine_tuning.create(**kwargs))
        job_id = created.get("id")
        task.log(f"Treino criado: {job_id}. Acompanhe na aba Treinos.")
        return {"job_id": job_id, "file_id": file_id, "base_model": base_model}

    return {"task_id": run_in_task("treino", job).id}


@router.get("/api/training/jobs", dependencies=[Auth])
def list_jobs(limit: int = Query(25, ge=1, le=100)):
    client = get_together()
    try:
        raw = client.fine_tuning.list()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"A Together não respondeu: {exc}")
    items = as_dict(raw).get("data") if not isinstance(raw, list) else raw
    items = items or []
    out = []
    for item in items:
        d = as_dict(item)
        out.append({
            "id": d.get("id"),
            "status": str(d.get("status", "")).lower().replace("status.", ""),
            "model": d.get("model"),
            "output_name": d.get("output_name"),
            "created_at": d.get("created_at"),
            "n_epochs": d.get("n_epochs"),
            "training_file": d.get("training_file"),
        })
    out.sort(key=lambda j: str(j.get("created_at") or ""), reverse=True)
    return {"jobs": out[:limit]}


@router.get("/api/training/jobs/{job_id}", dependencies=[Auth])
def get_job(job_id: str):
    client = get_together()
    d = as_dict(client.fine_tuning.retrieve(job_id))
    events = []
    try:
        raw = client.fine_tuning.list_events(job_id)
        raw = as_dict(raw).get("data") if not isinstance(raw, list) else raw
        events = [as_dict(e) for e in (raw or [])][-40:]
    except Exception:
        pass
    d["_events"] = events
    return d


@router.post("/api/training/jobs/{job_id}/cancel", dependencies=[Auth])
def cancel_job(job_id: str):
    get_together().fine_tuning.cancel(job_id)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Modelos e publicação
# ---------------------------------------------------------------------------

@router.get("/api/models", dependencies=[Auth])
def list_models():
    """Modelos já treinados, extraídos dos treinos concluídos."""
    client = get_together()
    try:
        raw = client.fine_tuning.list()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(502, f"A Together não respondeu: {exc}")
    items = as_dict(raw).get("data") if not isinstance(raw, list) else raw
    models = []
    for item in items or []:
        d = as_dict(item)
        status = str(d.get("status", "")).lower()
        if d.get("output_name") and ("completed" in status or "succeeded" in status):
            models.append({
                "name": d["output_name"], "job_id": d.get("id"),
                "base_model": d.get("model"), "created_at": d.get("created_at"),
            })
    models.sort(key=lambda m: str(m.get("created_at") or ""), reverse=True)
    active = _active_model() or {}
    return {"models": models, "active": active.get("active_model")}


@router.post("/api/models/sanity-check", dependencies=[Auth])
async def run_sanity_check(request: Request):
    body = await request.json()
    model = (body or {}).get("model")
    if not model:
        raise HTTPException(400, "Escolha o modelo a testar.")
    prompts = (body or {}).get("prompts") or ["Oi, tudo bem?", "Obrigado pela ajuda!"]
    system = (body or {}).get("system") or "Você é um assistente simpático e conversacional."

    def job(task: Task):
        client = get_together()
        results = []
        for prompt in prompts:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                max_tokens=200, temperature=0.5,
            )
            answer = resp.choices[0].message.content or ""
            passed = len(answer.strip()) >= 2
            task.log(f"{'ok  ' if passed else 'falha'}  {prompt!r} -> {answer[:120]!r}")
            results.append({"prompt": prompt, "answer": answer, "passed": passed})
        approved = all(r["passed"] for r in results)
        task.log("Aprovado no teste." if approved else "Reprovado no teste.")
        return {"model": model, "approved": approved, "results": results}

    return {"task_id": run_in_task("teste", job).id}


@router.post("/api/models/publish", dependencies=[Auth])
async def publish_model_endpoint(request: Request):
    """Troca o modelo em produção. O main.py pega a mudança pelo cache, sem reiniciar."""
    body = await request.json()
    model = (body or {}).get("model")
    if not model:
        raise HTTPException(400, "Escolha o modelo a publicar.")
    sb = get_supabase()
    previous = (_active_model() or {}).get("active_model")
    sb.table("model_config").upsert({
        "id": 1, "active_model": model, "updated_at": now_iso(),
    }).execute()
    try:
        sb.table("training_runs").insert({
            "id": str(uuid_lib.uuid4()), "started_at": now_iso(), "finished_at": now_iso(),
            "status": "success", "model_id": model,
            "notes": f"publicado pelo painel (anterior: {previous or 'nenhum'})",
        }).execute()
    except Exception:
        pass
    return {"ok": True, "active_model": model, "previous": previous,
            "cache_ttl": os.getenv("MODEL_CACHE_TTL_SECONDS", "300")}


@router.get("/api/models/history", dependencies=[Auth])
def model_history(limit: int = Query(30, ge=1, le=200)):
    try:
        resp = (get_supabase().table("training_runs").select("*")
                .order("started_at", desc=True).limit(limit).execute())
        return {"runs": resp.data or [], "table_exists": True}
    except Exception as exc:  # noqa: BLE001
        return {"runs": [], "table_exists": False, "detail": str(exc)}


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

@router.post("/api/pipeline/run", dependencies=[Auth])
async def run_pipeline_endpoint(request: Request):
    body = await request.json() if await request.body() else {}
    body = body or {}
    dry_run = bool(body.get("dry_run"))
    min_new = int(body.get("min_new_examples") or os.getenv("AUTO_FT_MIN_NEW_EXAMPLES", "50"))
    min_rating = body.get("min_rating")
    poll_interval = int(body.get("poll_interval") or 60)
    poll_timeout = int(body.get("poll_timeout") or 3600)
    suffix = body.get("suffix") or None

    for task in TASKS.values():
        if task.kind == "pipeline" and task.status == "running":
            raise HTTPException(409, f"Já existe um ciclo rodando (tarefa {task.id}).")

    def job(task: Task):
        from auto_learning_pipeline import run_pipeline

        task.log("Iniciando o ciclo" + (" em modo de teste (nada será publicado)." if dry_run else "."))
        run_pipeline(
            min_new_examples=min_new,
            min_rating=min_rating if min_rating not in ("", "todas") else None,
            poll_interval=poll_interval, poll_timeout=poll_timeout,
            dry_run=dry_run, suffix=suffix,
        )
        return {"dry_run": dry_run}

    return {"task_id": run_in_task("pipeline", job).id}


# ---------------------------------------------------------------------------
# Configuração visível no painel
# ---------------------------------------------------------------------------

@router.get("/api/config", dependencies=[Auth])
def get_config():
    def mask(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return value[:6] + "…" + value[-4:] if len(value) > 12 else "•" * len(value)

    return {
        "supabase_url": SUPABASE_URL,
        "supabase_key": mask(SUPABASE_SERVICE_KEY),
        "together_key": mask(TOGETHER_API_KEY),
        "datasets_dir": str(DATASETS_DIR),
        "base_model": _default_base_model(),
        "min_new_examples": os.getenv("AUTO_FT_MIN_NEW_EXAMPLES", "50"),
        "min_rating": os.getenv("AUTO_FT_MIN_RATING") or "",
        "poll_interval": os.getenv("AUTO_FT_POLL_INTERVAL_SECONDS", "60"),
        "poll_timeout": os.getenv("AUTO_FT_POLL_TIMEOUT_SECONDS", "3600"),
        "model_cache_ttl": os.getenv("MODEL_CACHE_TTL_SECONDS", "300"),
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
    }


@router.get("/api/health")
def health():
    return {"ok": True, "time": now_iso()}


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------

@router.get("", response_class=HTMLResponse, include_in_schema=False)
@router.get("/", response_class=HTMLResponse, include_in_schema=False)
def admin_page():
    index = STATIC_DIR / "admin.html"
    if not index.exists():
        raise HTTPException(500, f"admin.html não encontrado em {STATIC_DIR}.")
    return HTMLResponse(index.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Execução isolada
# ---------------------------------------------------------------------------

def create_app():
    from fastapi import FastAPI

    app = FastAPI(title="Painel de aprendizado contínuo", docs_url="/admin/docs")
    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root():
        from fastapi.responses import RedirectResponse

        return RedirectResponse("/admin")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("ADMIN_PORT", "8800"))
    print("\n" + "=" * 64)
    print(f"  Painel disponível em  http://localhost:{port}/admin")
    if _TOKEN_WAS_GENERATED:
        print(f"  Token de acesso desta sessão: {ADMIN_TOKEN}")
        print("  Defina ADMIN_TOKEN no .env para fixar uma senha.")
    else:
        print("  Token de acesso: o valor de ADMIN_TOKEN no seu .env")
    print(f"  Datasets em: {DATASETS_DIR}")
    print("=" * 64 + "\n")
    uvicorn.run(app, host=os.getenv("ADMIN_HOST", "127.0.0.1"), port=port)
