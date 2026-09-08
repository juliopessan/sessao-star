import io
import json
import os
import re
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import db
import ml

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "frontend"

@asynccontextmanager
async def lifespan(_app: FastAPI):
    db.init_db()
    yield


app = FastAPI(title="STAR Session", lifespan=lifespan)


ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
DEFAULT_VOICE_PT = os.getenv("KOKORO_VOICE", "pm_alex")
DEFAULT_VOICE_EN = os.getenv("KOKORO_VOICE_EN", "af_heart")
WHISPER_SIZE = os.getenv("WHISPER_MODEL", "small")


# ============================================================
# Kokoro TTS
# ============================================================

_kokoro_pipelines = {}
_kokoro_lock = threading.Lock()


def get_kokoro_pipeline(lang_code: str):
    """Loads the Kokoro pipeline for the requested language ('p' = Brazilian
    Portuguese, 'a' = American English) once, keeping one instance per language."""
    if lang_code not in _kokoro_pipelines:
        from kokoro import KPipeline

        _kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code)
    return _kokoro_pipelines[lang_code]


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    lang: Optional[str] = "pt"


@app.post("/api/tts")
def tts(req: TTSRequest):
    text = (req.text or "").strip()
    if not text:
        raise HTTPException(400, "Empty text.")
    is_en = req.lang == "en"
    lang_code = "a" if is_en else "p"
    voice = req.voice or (DEFAULT_VOICE_EN if is_en else DEFAULT_VOICE_PT)

    try:
        # Kokoro's pipeline isn't safe for concurrent calls (e.g. two quick
        # clicks on "play again") — serialize with a lock.
        with _kokoro_lock:
            pipeline = get_kokoro_pipeline(lang_code)
            chunks = [audio for _graphemes, _phonemes, audio in pipeline(text, voice=voice)]
    except Exception as exc:  # pragma: no cover - local infra failure
        raise HTTPException(500, f"Failed to generate voice with Kokoro: {exc}") from exc

    if not chunks:
        raise HTTPException(500, "Kokoro didn't generate audio for this text.")

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


EXTRACT_ERRORS = {
    "empty_file": {"pt": "Arquivo vazio.", "en": "Empty file."},
    "unsupported_format": {
        "pt": "Formato '{suffix}' não suportado. Envie PDF, DOCX ou TXT.",
        "en": "Format '{suffix}' not supported. Upload PDF, DOCX, or TXT.",
    },
    "extract_failed": {
        "pt": "Falha ao extrair texto do arquivo: {exc}",
        "en": "Failed to extract text from the file: {exc}",
    },
    "empty_text": {
        "pt": "Não consegui extrair texto deste arquivo — ele pode ser um PDF escaneado (imagem). Cole o currículo manualmente.",
        "en": "Couldn't extract text from this file — it might be a scanned (image-only) PDF. Paste the résumé manually.",
    },
}


@app.post("/api/extract-resume")
async def extract_resume(file: UploadFile = File(...), lang: str = Form("pt")):
    """Extracts the text from a résumé uploaded as PDF, DOCX, or TXT."""
    ui_lang = "en" if lang == "en" else "pt"
    name = file.filename or "resume"
    suffix = Path(name).suffix.lower()
    data = await file.read()
    if not data:
        raise HTTPException(400, EXTRACT_ERRORS["empty_file"][ui_lang])

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
            raise HTTPException(415, EXTRACT_ERRORS["unsupported_format"][ui_lang].format(suffix=suffix))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, EXTRACT_ERRORS["extract_failed"][ui_lang].format(exc=exc)) from exc

    text = text.strip()
    if not text:
        raise HTTPException(422, EXTRACT_ERRORS["empty_text"][ui_lang])

    return {"text": text, "filename": name}


@app.post("/api/stt")
async def stt(audio: UploadFile = File(...), language: str = Form("pt")):
    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    data = await audio.read()
    if not data:
        raise HTTPException(400, "Empty audio.")
    whisper_lang = "en" if language == "en" else "pt"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        with _whisper_lock:
            model = get_whisper_model()
            segments, _info = model.transcribe(tmp_path, language=whisper_lang, beam_size=5)
            text = " ".join(seg.text.strip() for seg in segments).strip()
    except Exception as exc:  # pragma: no cover
        raise HTTPException(500, f"Failed to transcribe with Whisper: {exc}") from exc
    finally:
        os.unlink(tmp_path)

    return {"text": text}


# ============================================================
# STAR question generation and final report (Claude API, with local fallback)
# ============================================================

SKILLS = {
    "pt": [
        "liderança", "dados", "vendas", "atendimento", "projeto", "logística",
        "financeiro", "marketing", "TI", "engenharia", "produto", "operações",
        "suporte", "design", "recursos humanos", "qualidade",
    ],
    "en": [
        "leadership", "data", "sales", "customer service", "project", "logistics",
        "finance", "marketing", "IT", "engineering", "product", "operations",
        "support", "design", "human resources", "quality",
    ],
}


def detect_area(text: str, lang: str = "pt") -> str:
    lower = (text or "").lower()
    for skill in SKILLS.get(lang, SKILLS["pt"]):
        if skill.lower() in lower:
            return skill
    return "sua área" if lang == "pt" else "your field"


def local_questions(resume: str, role: str, jd: str = "", lang: str = "pt") -> List[dict]:
    area = detect_area(f"{resume}\n{jd}", lang)

    if lang == "en":
        for_role = f" for the {role} role" if role else ""
        questions = [
            {"theme": "Leadership", "question": f"Tell me about a time you led a person or a team in {area}{for_role}. What was the situation, what was your responsibility, what did you do, and what was the result?"},
            {"theme": "Conflict", "question": "Describe a real conflict you had with a colleague, client, or manager. What caused the disagreement, what action did you take, and how did it end?"},
            {"theme": "Tight deadline", "question": "Talk about a project with a very tight deadline you had to deliver. What was the scenario, what was your responsibility, what did you do to pull it off, and what was the final result?"},
            {"theme": "Failure and learning", "question": "Tell me about a mistake or a decision that didn't work out. What was the context, what did you do about it, and what changed afterward?"},
            {"theme": "Initiative", "question": "Give an example of something you proposed without anyone asking for it. What was the situation, your task, the action taken, and the measured impact?"},
            {"theme": "Teamwork", "question": "Describe a deliverable that depended heavily on other people or teams. What was the shared goal, your specific role, what did you do to align everyone, and what result was achieved?"},
        ]
    else:
        target = f" para a vaga de {role}" if role else ""
        questions = [
            {"theme": "Liderança", "question": f"Conte sobre uma vez em que você liderou uma pessoa ou um time em {area}{target}. Qual era a situação, o que estava sob sua responsabilidade, o que você fez e qual foi o resultado?"},
            {"theme": "Conflito", "question": "Descreva um conflito real que você teve com um colega, cliente ou gestor. O que causou o desentendimento, que ação você tomou e como isso terminou?"},
            {"theme": "Prazo apertado", "question": "Fale sobre um projeto com prazo muito curto que você precisou entregar. Como era o cenário, o que ficou sob sua responsabilidade, o que você fez para dar conta e qual foi o resultado final?"},
            {"theme": "Falha e aprendizado", "question": "Conte sobre um erro ou uma decisão que não deu certo. Qual era o contexto, o que você fez diante disso e o que mudou depois?"},
            {"theme": "Iniciativa", "question": "Dê um exemplo de algo que você propôs sem que ninguém tivesse pedido. Qual era a situação, sua tarefa, a ação tomada e o impacto medido?"},
            {"theme": "Trabalho em equipe", "question": "Descreva uma entrega que dependeu fortemente de outras pessoas ou áreas. Qual era o objetivo comum, seu papel específico, o que você fez para alinhar todo mundo e o resultado alcançado?"},
        ]

    # In the local fallback, prioritize the themes where the candidate's history shows the most STAR gaps.
    weak_names = {w["theme"] for w in ml.weak_theme_profile(limit=3, lang=lang)}
    if weak_names:
        questions.sort(key=lambda q: 0 if q["theme"] in weak_names else 1)
    return questions


# Internal keys (situation/task/action/result) are fixed regardless of
# language — only the keyword list used to detect each one changes.
STAR_HINTS = {
    "pt": {
        "situation": ["quando", "na época", "situação", "cenário", "empresa", "projeto", "time"],
        "task": ["responsável", "minha tarefa", "precisava", "objetivo", "meta"],
        "action": ["eu fiz", "decidi", "implementei", "propus", "organizei", "conversei", "criei", "liderei"],
        "result": ["resultado", "consegui", "reduziu", "aumentou", "%", "impacto", "economizou", "melhorou"],
    },
    "en": {
        "situation": ["when", "at the time", "situation", "context", "company", "project", "team"],
        "task": ["responsible", "my task", "i needed to", "goal", "objective"],
        "action": ["i did", "i decided", "i implemented", "i proposed", "i organized", "i talked", "i created", "i led"],
        "result": ["result", "achieved", "reduced", "increased", "%", "impact", "saved", "improved"],
    },
}

STAR_LABELS = {
    "pt": {"situation": "Situação", "task": "Tarefa", "action": "Ação", "result": "Resultado"},
    "en": {"situation": "Situation", "task": "Task", "action": "Action", "result": "Result"},
}


def star_coverage(answer: str, lang: str = "pt") -> List[str]:
    """Keyword heuristic — the 'teacher' that labels the training data for the model in ml.py."""
    hints = STAR_HINTS.get(lang, STAR_HINTS["pt"])
    lower = (answer or "").lower()
    return [key for key, words in hints.items() if any(w in lower for w in words)]


def coverage_for(answer: str, lang: str = "pt") -> List[str]:
    """STAR coverage for an answer: uses the trained model (ml.py) when available,
    otherwise falls back to the keyword heuristic."""
    predicted = ml.predict_coverage(answer, lang)
    return predicted if predicted is not None else star_coverage(answer, lang)


def local_report(transcripts: List[dict], lang: str = "pt") -> str:
    labels = STAR_LABELS.get(lang, STAR_LABELS["pt"])
    answered = [t for t in transcripts if not t.get("skipped") and t.get("answer")]
    if lang == "en":
        lines = [f"Answers recorded: {len(answered)} of {len(transcripts)}.", ""]
        for i, t in enumerate(answered, start=1):
            covered = coverage_for(t["answer"], lang)
            words = len(t["answer"].split())
            cov_text = ", ".join(labels[k] for k in covered) if covered else "no clear STAR signal"
            lines.append(f"{i}. [{t['theme']}] {words} words — covered: {cov_text}")
        lines.append("")
        lines.append("Tip: short answers rarely cover Situation, Task, Action and Result. Always try to close with a concrete number or outcome.")
    else:
        lines = [f"Respostas registradas: {len(answered)} de {len(transcripts)}.", ""]
        for i, t in enumerate(answered, start=1):
            covered = coverage_for(t["answer"], lang)
            words = len(t["answer"].split())
            cov_text = ", ".join(labels[k] for k in covered) if covered else "nenhum sinal claro de STAR"
            lines.append(f"{i}. [{t['theme']}] {words} palavras — cobriu: {cov_text}")
        lines.append("")
        lines.append("Dica: respostas curtas raramente cobrem Situação, Tarefa, Ação e Resultado. Tente sempre fechar com um número ou um resultado concreto.")
    return "\n".join(lines)


def local_tutor_feedback(answer: str, lang: str = "pt") -> str:
    labels = STAR_LABELS.get(lang, STAR_LABELS["pt"])
    covered = coverage_for(answer, lang)
    missing = [labels[k] for k in labels if k not in covered]
    words = len(answer.split())

    parts = []
    if lang == "en":
        if covered:
            parts.append("Nice! Your answer already showed signs of " + ", ".join(labels[k] for k in covered) + ".")
        else:
            parts.append("Your answer didn't clearly show any of the four STAR elements yet.")
        if missing:
            parts.append("For the real interview, try to also include: " + ", ".join(missing) + ".")
        if words < 25:
            parts.append("Very short answers rarely land well — try fleshing out each part a bit more.")
        parts.append("Now let's move on to the real questions.")
    else:
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
    raise ValueError("Claude's response doesn't contain valid JSON.")


class TutorFeedbackRequest(BaseModel):
    question: str
    answer: str
    lang: Optional[str] = "pt"


@app.post("/api/tutor-feedback")
def tutor_feedback(req: TutorFeedbackRequest):
    answer = (req.answer or "").strip()
    lang = "en" if req.lang == "en" else "pt"
    if not answer:
        skip_msg = (
            "You skipped the practice answer — no problem, let's go straight to the real interview."
            if lang == "en"
            else "Você pulou a resposta de treino — sem problemas, vamos direto para a entrevista real."
        )
        return {"feedback": skip_msg, "source": "none"}

    client = get_anthropic_client()
    if client is None:
        return {"feedback": local_tutor_feedback(answer, lang), "source": "local"}

    if lang == "en":
        prompt = (
            "You are a warm, direct interview tutor helping a candidate practice the STAR method "
            "(Situation, Task, Action, Result) before the real interview. They just answered ONE "
            "practice question (it does not count toward any score). Give short feedback (3 to 4 "
            "sentences), in English, encouraging but honest tone: say what the answer already "
            "covered well of the STAR method, point out at most two concrete things to improve, and "
            "end by encouraging the candidate to move on to the real interview. No markdown, no "
            "lists.\n\n"
            f"PRACTICE QUESTION: {req.question}\n\nCANDIDATE'S ANSWER: {answer[:3000]}"
        )
    else:
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
            raise ValueError("Empty response.")
        return {"feedback": text, "source": "claude"}
    except Exception:
        return {"feedback": local_tutor_feedback(answer, lang), "source": "local"}


FOLLOW_UP_QUESTIONS = {
    "pt": {
        "situation": "Qual era o contexto específico e quem estava envolvido nessa situação?",
        "task": "Qual era exatamente a sua responsabilidade ou o objetivo que precisava cumprir?",
        "action": "Quais foram as ações concretas que você tomou, passo a passo?",
        "result": "Qual foi o resultado concreto? Se possível, traga um número, prazo ou impacto mensurável.",
    },
    "en": {
        "situation": "What was the specific context, and who was involved in that situation?",
        "task": "What exactly was your responsibility or the goal you needed to achieve?",
        "action": "What concrete actions did you take, step by step?",
        "result": "What was the concrete result? If possible, share a number, timeframe, or measurable impact.",
    },
}


class FollowUpRequest(BaseModel):
    question: str
    answer: str
    theme: Optional[str] = ""
    lang: Optional[str] = "pt"


@app.post("/api/follow-up")
def follow_up(req: FollowUpRequest):
    answer = (req.answer or "").strip()
    lang = "en" if req.lang == "en" else "pt"
    covered = coverage_for(answer, lang)
    missing = [label for label in ("situation", "task", "action", "result") if label not in covered]
    should_follow_up = bool(answer) and (len(covered) < 3 or len(answer.split()) < 20)
    if not should_follow_up:
        return {"should_follow_up": False, "missing": [], "source": "heuristic"}

    client = get_anthropic_client()
    fallback = FOLLOW_UP_QUESTIONS[lang][missing[0]] if missing else FOLLOW_UP_QUESTIONS[lang]["result"]
    if client is None:
        return {"should_follow_up": True, "question": fallback, "missing": missing, "source": "local"}

    if lang == "en":
        prompt = (
            "You are an adaptive behavioral interviewer using the STAR method. The candidate's "
            "answer is incomplete. Write exactly ONE concise follow-up question in English that "
            "asks for the most important missing STAR detail. Do not explain, do not use markdown.\n\n"
            f"MAIN QUESTION: {req.question}\nTHEME: {req.theme}\n"
            f"MISSING ELEMENTS: {', '.join(missing)}\nANSWER: {answer[:3000]}"
        )
    else:
        prompt = (
            "Você é um recrutador adaptativo usando o método STAR. A resposta do candidato está "
            "incompleta. Escreva EXATAMENTE UMA pergunta curta de follow-up em português do Brasil "
            "pedindo o detalhe STAR mais importante que faltou. Não explique e não use markdown.\n\n"
            f"PERGUNTA PRINCIPAL: {req.question}\nTEMA: {req.theme}\n"
            f"ELEMENTOS AUSENTES: {', '.join(missing)}\nRESPOSTA: {answer[:3000]}"
        )
    try:
        msg = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=180,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in msg.content if block.type == "text").strip()
        if not text:
            raise ValueError("Empty follow-up question.")
        return {"should_follow_up": True, "question": text, "missing": missing, "source": "claude"}
    except Exception:
        return {"should_follow_up": True, "question": fallback, "missing": missing, "source": "local"}

def weak_theme_hint(lang: str = "pt") -> str:
    """Prompt snippet with the themes where the candidate's history shows the most STAR gaps."""
    weak = ml.weak_theme_profile(limit=2, lang=lang)
    if not weak:
        return ""
    names = ", ".join(w["theme"] for w in weak)
    if lang == "en":
        return (
            f"CANDIDATE HISTORY: in previous practice sessions, the themes where they most often "
            f"failed to cover Situation, Task, Action or Result were: {names}. Include at least one "
            f"question on those themes, if it makes sense given the résumé and the job.\n\n"
        )
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
    lang: Optional[str] = "pt"


@app.post("/api/questions")
def questions(req: QuestionsRequest):
    resume = (req.resume or "").strip()
    jd = (req.jd or "").strip()
    lang = "en" if req.lang == "en" else "pt"
    if not resume:
        raise HTTPException(400, "Currículo vazio." if lang == "pt" else "Résumé is empty.")

    client = get_anthropic_client()
    if client is None:
        return {"questions": local_questions(resume, req.role or "", jd, lang), "source": "local"}

    if lang == "en":
        prompt = (
            "You are an experienced recruiter conducting a behavioral interview using the STAR "
            "method (Situation, Task, Action, Result). Based on the candidate's résumé and the job "
            "description (JD) below, generate EXACTLY 6 behavioral interview questions in English. "
            "Cross-reference both sources: for each important JD requirement, prefer a question that "
            "probes a real candidate experience related to that requirement (using what's in the "
            "résumé); where the résumé does not cover a central JD requirement, include a question "
            "that probes whether the candidate has equivalent experience in another context. Cover "
            "varied themes (leadership, conflict, tight deadline, failure/learning, initiative, "
            "teamwork) and mention, when possible, something concrete from the résumé (company, "
            "project, number) or the job (technology, required responsibility). Reply ONLY with a "
            'JSON in the format: {"questions":[{"theme":"short string","question":"full question"}]}, '
            "with no other text.\n\n"
            f"CANDIDATE'S RÉSUMÉ:\n{resume[:6000]}\n\n"
            + (f"JOB DESCRIPTION (JD):\n{jd[:4000]}\n\n" if jd else "")
            + (f"JOB TITLE: {req.role}\n\n" if req.role else "")
            + (f"LINKEDIN URL (context only, ignore if not helpful): {req.linkedin}\n\n" if req.linkedin else "")
            + weak_theme_hint(lang)
        )
    else:
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
            + weak_theme_hint(lang)
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
            {"theme": q.get("theme", "Pergunta" if lang == "pt" else "Question"), "question": q.get("question", "")}
            for q in qs
            if q.get("question")
        ]
        if not cleaned:
            raise ValueError("No valid questions returned.")
        return {"questions": cleaned, "source": "claude"}
    except Exception:
        return {"questions": local_questions(resume, req.role or "", jd, lang), "source": "local"}


class ReportRequest(BaseModel):
    transcripts: List[dict]
    jd: Optional[str] = ""
    lang: Optional[str] = "pt"


@app.post("/api/report")
def report(req: ReportRequest):
    transcripts = req.transcripts or []
    jd = (req.jd or "").strip()
    lang = "en" if req.lang == "en" else "pt"
    answered = [t for t in transcripts if not t.get("skipped") and t.get("answer")]
    if not answered:
        msg = "No answers were recorded in this session." if lang == "en" else "Nenhuma resposta foi registrada nesta sessão."
        return {"report": msg, "source": "none"}

    client = get_anthropic_client()
    if client is None:
        return {"report": local_report(transcripts, lang), "source": "local"}

    if lang == "en":
        transcript_text = "\n\n".join(
            f"{i}. Theme: {t.get('theme')}\nQuestion: {t.get('question')}\n"
            f"Answer: {'(skipped)' if t.get('skipped') else (t.get('answer') or '(empty)')}"
            for i, t in enumerate(transcripts, start=1)
        )
        prompt = (
            "You are a senior recruiter evaluating a simulated interview using the STAR method "
            "(Situation, Task, Action, Result). Below is the transcript of questions and answers "
            "spoken by the candidate"
            + (", and the job description (JD) they are being evaluated against" if jd else "")
            + ". Write an objective evaluation in English, in flowing prose "
            "(no markdown, no bullet lists), at most 160 words, covering: specific strengths, one or "
            "two answers that missed Situation/Task/Action/Result"
            + (", how well the candidate seems to fit the job requirements" if jd else "")
            + ", and one practical recommendation for the next interview.\n\n"
            + (f"JOB DESCRIPTION (JD):\n{jd[:4000]}\n\n" if jd else "")
            + "TRANSCRIPT:\n"
            + transcript_text[:6000]
        )
    else:
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
            raise ValueError("Empty response.")
        return {"report": text, "source": "claude"}
    except Exception:
        return {"report": local_report(transcripts, lang), "source": "local"}


# ============================================================
# Local history (SQLite) and predictive model (ml.py)
# ============================================================

class SessionSaveRequest(BaseModel):
    role: Optional[str] = ""
    resume: Optional[str] = ""
    jd: Optional[str] = ""
    questions_source: Optional[str] = ""
    report: Optional[str] = ""
    report_source: Optional[str] = ""
    transcripts: List[dict]
    lang: Optional[str] = "pt"


@app.post("/api/session/save")
def session_save(req: SessionSaveRequest):
    lang = "en" if req.lang == "en" else "pt"
    session_id = db.save_session(
        role=(req.role or "").strip(),
        resume_excerpt=(req.resume or "").strip()[:2000],
        jd_excerpt=(req.jd or "").strip()[:2000],
        questions_source=req.questions_source or "",
        report_text=req.report or "",
        report_source=req.report_source or "",
        transcripts=req.transcripts or [],
        coverage_fn=lambda a: star_coverage(a, lang),
        lang=lang,
    )
    model = ml.train(lang)
    return {
        "session_id": session_id,
        "sessions_total": db.session_count(lang),
        "model_trained": model is not None or ml.model_status(lang)["trained"],
    }


@app.get("/api/insights")
def insights(lang: str = "pt"):
    ui_lang = "en" if lang == "en" else "pt"
    return {
        "sessions_total": db.session_count(ui_lang),
        "answers_total": db.answer_count(ui_lang),
        "weak_themes": ml.weak_theme_profile(lang=ui_lang),
        "model": ml.model_status(ui_lang),
        "coverage": db.coverage_summary(ui_lang),
        "timeline": db.history(lang=ui_lang),
    }


# ============================================================
# Static frontend
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
