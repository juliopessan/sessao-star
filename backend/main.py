import io
import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
import ml

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"

app = FastAPI(title="Sessão STAR")


@app.on_event("startup")
def _startup():
    db.init_db()


ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "pm_alex")
WHISPER_SIZE = os.getenv("WHISPER_MODEL", "small")


# ============================================================
# Kokoro TTS
# ============================================================

_kokoro_pipeline = None
_kokoro_lock = threading.Lock()


def get_kokoro_pipeline():
    """Carrega o pipeline Kokoro (idioma 'p' = português do Brasil) uma única vez."""
    global _kokoro_pipeline
    if _kokoro_pipeline is None:
        from kokoro import KPipeline

        _kokoro_pipeline = KPipeline(lang_code="p")
    return _kokoro_pipeline


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None


@app.post("/api/tts")
def tts(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "Texto vazio.")
    voice = req.voice or DEFAULT_VOICE

    try:
        # o pipeline do Kokoro não é seguro para chamadas concorrentes (ex.: dois
        # cliques rápidos em "ouvir de novo") — serializa com um lock.
        with _kokoro_lock:
            pipeline = get_kokoro_pipeline()
            chunks = [audio for _graphemes, _phonemes, audio in pipeline(text, voice=voice)]
    except Exception as exc:  # pragma: no cover - erro de infraestrutura local
        raise HTTPException(500, f"Falha ao gerar voz com Kokoro: {exc}") from exc

    if not chunks:
        raise HTTPException(500, "Kokoro não gerou áudio para este texto.")

    audio = np.concatenate(chunks)
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


# ============================================================
# Whisper STT (faster-whisper)
# ============================================================

_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(WHISPER_SIZE, device="auto", compute_type="int8")
    return _whisper_model


@app.post("/api/extract-resume")
async def extract_resume(file: UploadFile = File(...)):
    """Extrai o texto de um currículo enviado em PDF, DOCX ou TXT."""
    name = file.filename or "curriculo"
    suffix = Path(name).suffix.lower()
    data = await file.read()
    if not data:
        raise HTTPException(400, "Arquivo vazio.")

    try:
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif suffix == ".docx":
            from docx import Document

            doc = Document(io.BytesIO(data))
            text = "\n".join(p.text for p in doc.paragraphs)
        elif suffix in (".txt", ".md", ""):
            text = data.decode("utf-8", errors="ignore")
        else:
            raise HTTPException(415, f"Formato '{suffix}' não suportado. Envie PDF, DOCX ou TXT.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Falha ao extrair texto do arquivo: {exc}") from exc

    text = text.strip()
    if not text:
        raise HTTPException(422, "Não consegui extrair texto deste arquivo — ele pode ser um PDF escaneado (imagem). Cole o currículo manualmente.")

    return {"text": text, "filename": name}


@app.post("/api/stt")
async def stt(audio: UploadFile = File(...)):
    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Áudio vazio.")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        with _whisper_lock:
            model = get_whisper_model()
            segments, _info = model.transcribe(tmp_path, language="pt", beam_size=5)
            text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"Falha ao transcrever com Whisper: {exc}") from exc
    finally:
        os.unlink(tmp_path)

    return {"text": text}


# ============================================================
# Geração de perguntas STAR e relatório final (Claude API, com fallback local)
# ============================================================

SKILLS = [
    "liderança", "dados", "vendas", "atendimento", "projeto", "logística",
    "financeiro", "marketing", "TI", "engenharia", "produto", "operações",
    "suporte", "design", "recursos humanos", "qualidade",
]


def detect_area(text: str) -> str:
    lower = (text or "").lower()
    for skill in SKILLS:
        if skill.lower() in lower:
            return skill
    return "sua área"


def local_questions(resume: str, role: str, jd: str = "") -> List[dict]:
    area = detect_area(f"{resume}\n{jd}")
    alvo = f" para a vaga de {role}" if role else ""
    questions = [
        {"theme": "Liderança", "question": f"Conte sobre uma vez em que você liderou uma pessoa ou um time em {area}{alvo}. Qual era a situação, o que estava sob sua responsabilidade, o que você fez e qual foi o resultado?"},
        {"theme": "Conflito", "question": "Descreva um conflito real que você teve com um colega, cliente ou gestor. O que causou o desentendimento, que ação você tomou e como isso terminou?"},
        {"theme": "Prazo apertado", "question": "Fale sobre um projeto com prazo muito curto que você precisou entregar. Como era o cenário, o que ficou sob sua responsabilidade, o que você fez para dar conta e qual foi o resultado final?"},
        {"theme": "Falha e aprendizado", "question": "Conte sobre um erro ou uma decisão que não deu certo. Qual era o contexto, o que você fez diante disso e o que mudou depois?"},
        {"theme": "Iniciativa", "question": "Dê um exemplo de algo que você propôs sem que ninguém tivesse pedido. Qual era a situação, sua tarefa, a ação tomada e o impacto medido?"},
        {"theme": "Trabalho em equipe", "question": "Descreva uma entrega que dependeu fortemente de outras pessoas ou áreas. Qual era o objetivo comum, seu papel específico, o que você fez para alinhar todo mundo e o resultado alcançado?"},
    ]
    # prioriza, no fallback local, os temas onde o histórico do candidato mostra mais lacunas de STAR
    weak_names = {w["theme"] for w in ml.weak_theme_profile(limit=3)}
    if weak_names:
        questions.sort(key=lambda q: 0 if q["theme"] in weak_names else 1)
    return questions


STAR_HINTS = {
    "situação": ["quando", "na época", "situação", "cenário", "empresa", "projeto", "time"],
    "tarefa": ["responsável", "minha tarefa", "precisava", "objetivo", "meta"],
    "ação": ["eu fiz", "decidi", "implementei", "propus", "organizei", "conversei", "criei", "liderei"],
    "resultado": ["resultado", "consegui", "reduziu", "aumentou", "%", "impacto", "economizou", "melhorou"],
}


def star_coverage(answer: str) -> List[str]:
    """Heurística de palavras-chave — o 'professor' que rotula os dados de treino do modelo em ml.py."""
    lower = (answer or "").lower()
    return [key for key, words in STAR_HINTS.items() if any(w in lower for w in words)]


def coverage_for(answer: str) -> List[str]:
    """Cobertura STAR de uma resposta: usa o modelo treinado (ml.py) quando disponível,
    senão cai na heurística de palavras-chave."""
    predicted = ml.predict_coverage(answer)
    return predicted if predicted is not None else star_coverage(answer)


def local_report(transcripts: List[dict]) -> str:
    answered = [t for t in transcripts if not t.get("skipped") and t.get("answer")]
    lines = [f"Respostas registradas: {len(answered)} de {len(transcripts)}.", ""]
    for i, t in enumerate(answered, start=1):
        covered = coverage_for(t["answer"])
        words = len(t["answer"].split())
        cov_text = ", ".join(covered) if covered else "nenhum sinal claro de STAR"
        lines.append(f"{i}. [{t['theme']}] {words} palavras — cobriu: {cov_text}")
    lines.append("")
    lines.append("Dica: respostas curtas raramente cobrem Situação, Tarefa, Ação e Resultado. Tente sempre fechar com um número ou um resultado concreto.")
    return "\n".join(lines)


def local_tutor_feedback(answer: str) -> str:
    labels = {"situação": "Situação", "tarefa": "Tarefa", "ação": "Ação", "resultado": "Resultado"}
    covered = coverage_for(answer)
    missing = [labels[k] for k in labels if k not in covered]
    words = len(answer.split())

    parts = []
    if covered:
        parts.append("Boa! Sua resposta já trouxe sinais de " + ", ".join(labels[k] for k in covered) + ".")
    else:
        parts.append("Sua resposta ainda não deixou claro nenhum dos quatro elementos do método STAR.")
    if missing:
        parts.append("Para a entrevista de verdade, tente incluir também: " + ", ".join(missing) + ".")
    if words < 25:
        parts.append("Respostas muito curtas raramente convencem — tente detalhar um pouco mais cada parte.")
    parts.append("Agora vamos para as perguntas reais.")
    return " ".join(parts)


def get_anthropic_client():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None
    from anthropic import Anthropic

    return Anthropic(api_key=key)


def extract_json(text: str):
    text = text.strip()
    try:
        return json.loads(text)
    except ValueError:
        pass
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1).strip())
        except ValueError:
            pass
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    end = max(text.rfind("}"), text.rfind("]"))
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            pass
    raise ValueError("Resposta do Claude não contém JSON válido.")


class TutorFeedbackRequest(BaseModel):
    question: str
    answer: str


@app.post("/api/tutor-feedback")
def tutor_feedback(req: TutorFeedbackRequest):
    answer = (req.answer or "").strip()
    if not answer:
        return {"feedback": "Você pulou a resposta de treino — sem problemas, vamos direto para a entrevista real.", "source": "none"}

    client = get_anthropic_client()
    if client is None:
        return {"feedback": local_tutor_feedback(answer), "source": "local"}

    prompt = (
        "Você é um tutor de entrevistas simpático e direto, ajudando um candidato a treinar o "
        "método STAR (Situação, Tarefa, Ação, Resultado) antes da entrevista de verdade. Ele acabou "
        "de responder UMA pergunta de treino (a resposta não vale nota). Dê um feedback curto (3 a "
        "4 frases), em português do Brasil, tom encorajador mas honesto: diga o que a resposta já "
        "cobriu bem do método STAR, aponte no máximo dois pontos concretos para melhorar, e termine "
        "incentivando o candidato a seguir para a entrevista real. Não use markdown nem listas.\n\n"
        f"PERGUNTA DE TREINO: {req.question}\n\nRESPOSTA DO CANDIDATO: {answer[:3000]}"
    )
    try:
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text").strip()
        if not text:
            raise ValueError("Resposta vazia.")
        return {"feedback": text, "source": "claude"}
    except Exception:
        return {"feedback": local_tutor_feedback(answer), "source": "local"}


def weak_theme_hint() -> str:
    """Trecho de prompt com os temas onde o histórico do candidato mostra mais lacunas de STAR."""
    weak = ml.weak_theme_profile(limit=2)
    if not weak:
        return ""
    names = ", ".join(w["theme"] for w in weak)
    return (
        f"HISTÓRICO DO CANDIDATO: em sessões de prática anteriores, os temas onde ele mais deixou "
        f"de cobrir Situação, Tarefa, Ação ou Resultado foram: {names}. Inclua pelo menos uma "
        f"pergunta nesses temas, se fizer sentido com o currículo e a vaga.\n\n"
    )


class QuestionsRequest(BaseModel):
    resume: str
    role: Optional[str] = ""
    jd: Optional[str] = ""
    linkedin: Optional[str] = ""


@app.post("/api/questions")
def questions(req: QuestionsRequest):
    resume = (req.resume or "").strip()
    jd = (req.jd or "").strip()
    if not resume:
        raise HTTPException(400, "Currículo vazio.")

    client = get_anthropic_client()
    if client is None:
        return {"questions": local_questions(resume, req.role or "", jd), "source": "local"}

    prompt = (
        "Você é um recrutador experiente conduzindo uma entrevista comportamental pelo método "
        "STAR (Situação, Tarefa, Ação, Resultado). Com base no currículo do candidato e na "
        "descrição da vaga (JD) abaixo, gere EXATAMENTE 6 perguntas de entrevista comportamental "
        "em português do Brasil. Cruze as duas fontes: para cada requisito importante da JD, "
        "prefira uma pergunta que investigue uma experiência real do candidato relacionada a esse "
        "requisito (usando o que está no currículo); onde o currículo não cobrir um requisito "
        "central da JD, inclua uma pergunta que investigue se o candidato tem experiência "
        "equivalente em outro contexto. Cubra temas variados (liderança, conflito, prazo apertado, "
        "falha/aprendizado, iniciativa, trabalho em equipe) e mencione, quando possível, algo "
        "concreto do currículo (empresa, projeto, número) ou da vaga (tecnologia, responsabilidade "
        "exigida). Responda SOMENTE com um JSON no formato: "
        '{"questions":[{"theme":"string curto","question":"pergunta completa"}]}, sem nenhum outro '
        "texto.\n\n"
        f"CURRÍCULO DO CANDIDATO:\n{resume[:6000]}\n\n"
        + (f"DESCRIÇÃO DA VAGA (JD):\n{jd[:4000]}\n\n" if jd else "")
        + (f"TÍTULO DA VAGA: {req.role}\n\n" if req.role else "")
        + (f"URL DO LINKEDIN (apenas contexto, ignore se não ajudar): {req.linkedin}\n\n" if req.linkedin else "")
        + weak_theme_hint()
    )

    try:
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = "".join(block.text for block in msg.content if block.type == "text")
        data = extract_json(raw)
        qs = data.get("questions") or []
        cleaned = [
            {"theme": q.get("theme", "Pergunta"), "question": q.get("question", "")}
            for q in qs
            if q.get("question")
        ]
        if not cleaned:
            raise ValueError("Nenhuma pergunta válida retornada.")
        return {"questions": cleaned, "source": "claude"}
    except Exception:
        return {"questions": local_questions(resume, req.role or "", jd), "source": "local"}


class ReportRequest(BaseModel):
    transcripts: List[dict]
    jd: Optional[str] = ""


@app.post("/api/report")
def report(req: ReportRequest):
    transcripts = req.transcripts or []
    jd = (req.jd or "").strip()
    answered = [t for t in transcripts if not t.get("skipped") and t.get("answer")]
    if not answered:
        return {"report": "Nenhuma resposta foi registrada nesta sessão.", "source": "none"}

    client = get_anthropic_client()
    if client is None:
        return {"report": local_report(transcripts), "source": "local"}

    transcript_text = "\n\n".join(
        f"{i}. Tema: {t.get('theme')}\nPergunta: {t.get('question')}\n"
        f"Resposta: {'(pulada)' if t.get('skipped') else (t.get('answer') or '(vazia)')}"
        for i, t in enumerate(transcripts, start=1)
    )
    prompt = (
        "Você é um recrutador sênior avaliando uma entrevista simulada pelo método STAR "
        "(Situação, Tarefa, Ação, Resultado). Abaixo está a transcrição de perguntas e respostas "
        "faladas pelo candidato"
        + (", e a descrição da vaga (JD) para a qual ele está sendo avaliado" if jd else "")
        + ". Escreva uma avaliação objetiva em português do Brasil, em texto "
        "corrido (sem markdown, sem listas com marcadores), com no máximo 160 palavras, cobrindo: "
        "pontos fortes específicos, uma ou duas respostas em que faltou Situação/Tarefa/Ação/"
        "Resultado"
        + (", o quão aderente o candidato parece estar aos requisitos da vaga" if jd else "")
        + ", e uma recomendação prática para a próxima entrevista.\n\n"
        + (f"DESCRIÇÃO DA VAGA (JD):\n{jd[:4000]}\n\n" if jd else "")
        + "TRANSCRIÇÃO:\n"
        + transcript_text[:6000]
    )
    try:
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text").strip()
        if not text:
            raise ValueError("Resposta vazia.")
        return {"report": text, "source": "claude"}
    except Exception:
        return {"report": local_report(transcripts), "source": "local"}


# ============================================================
# Histórico local (SQLite) e modelo preditivo (ml.py)
# ============================================================

class SessionSaveRequest(BaseModel):
    role: Optional[str] = ""
    resume: Optional[str] = ""
    jd: Optional[str] = ""
    questions_source: Optional[str] = ""
    report: Optional[str] = ""
    report_source: Optional[str] = ""
    transcripts: List[dict]


@app.post("/api/session/save")
def session_save(req: SessionSaveRequest):
    session_id = db.save_session(
        role=(req.role or "").strip(),
        resume_excerpt=(req.resume or "").strip()[:2000],
        jd_excerpt=(req.jd or "").strip()[:2000],
        questions_source=req.questions_source or "",
        report_text=req.report or "",
        report_source=req.report_source or "",
        transcripts=req.transcripts or [],
        coverage_fn=star_coverage,
    )
    model = ml.train()
    return {
        "session_id": session_id,
        "sessions_total": db.session_count(),
        "model_trained": model is not None,
    }


@app.get("/api/insights")
def insights():
    return {
        "sessions_total": db.session_count(),
        "answers_total": db.answer_count(),
        "weak_themes": ml.weak_theme_profile(),
        "model": ml.model_status(),
    }


# ============================================================
# Frontend estático
# ============================================================

app.mount("/assets", StaticFiles(directory=FRONTEND_DIR), name="assets")


@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico")
def favicon_ico():
    return FileResponse(FRONTEND_DIR / "favicon.ico")


@app.get("/favicon.svg")
def favicon_svg():
    return FileResponse(FRONTEND_DIR / "favicon.svg")


@app.get("/apple-touch-icon.png")
def apple_touch_icon():
    return FileResponse(FRONTEND_DIR / "apple-touch-icon.png")
