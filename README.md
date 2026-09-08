# Sessão STAR

Simulador de entrevista por voz: envie o currículo do candidato (PDF, DOCX ou texto colado) e a
descrição da vaga (JD), o Claude cruza os dois para gerar perguntas comportamentais no método STAR
(Situação, Tarefa, Ação, Resultado) — inclusive perguntas que sondam requisitos da vaga que não
aparecem no currículo — e um recrutador simulado conduz a entrevista falando com você — voz gerada
pelo **Kokoro TTS** e transcrita pelo **Whisper**, ambos rodando localmente no seu computador.

## Requisitos do sistema

- Python 3.10+
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) — usado pelo Kokoro para fonemizar o português:
  ```bash
  brew install espeak-ng
  ```
- ffmpeg — usado para decodificar o áudio gravado no navegador antes de mandar ao Whisper:
  ```bash
  brew install ffmpeg
  ```

## Instalação

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edite `.env` e adicione sua `ANTHROPIC_API_KEY` se quiser perguntas STAR personalizadas e
relatório final gerados pelo Claude. Sem a chave, o app funciona normalmente, mas usa perguntas
e relatório heurísticos (mais genéricos).

## Rodando

```bash
cd backend
source .venv/bin/activate
uvicorn main:app --reload
```

Abra http://localhost:8000 no navegador (Chrome ou Edge recomendados, para permissão de
microfone mais confiável).

## Primeira execução

- O **Kokoro** baixa os pesos do modelo (~centenas de MB) na primeira chamada a `/api/tts`.
- O **Whisper** (`faster-whisper`, modelo `small` por padrão) baixa os pesos na primeira chamada
  a `/api/stt`.
- Ambos ficam em cache local depois disso; as próximas sessões carregam na hora.

## Variáveis de ambiente (`backend/.env`)

| Variável | Padrão | Descrição |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Chave da API da Anthropic. Sem ela, perguntas e relatório usam heurística local. |
| `ANTHROPIC_MODEL` | `claude-haiku-4-5` | Modelo Claude usado para gerar perguntas e relatório. |
| `WHISPER_MODEL` | `small` | Tamanho do modelo Whisper (`tiny`, `base`, `small`, `medium`, `large-v3`). Modelos maiores são mais precisos e mais lentos. |
| `KOKORO_VOICE` | `pm_alex` | Voz do Kokoro para o recrutador. Vozes pt-BR: `pf_dora` (feminina), `pm_alex`, `pm_santa` (masculinas). |

## Arquitetura

```
backend/
  main.py            FastAPI: /api/extract-resume (PDF/DOCX/TXT -> texto),
                      /api/tts (Kokoro), /api/stt (Whisper),
                      /api/questions e /api/report (Claude API com fallback local),
                      e serve o frontend estático.
  requirements.txt
  .env.example
frontend/
  index.html         Interface única (HTML/CSS/JS puro, sem build step).
```

O frontend grava a resposta com `MediaRecorder`, envia o áudio para `/api/stt` e recebe o texto
transcrito (não é streaming ao vivo — grava, para, transcreve). A pergunta é sintetizada uma vez
por `/api/tts` e o áudio fica em cache no navegador durante a sessão, então "ouvir de novo" não
gera uma nova chamada ao Kokoro.

## Problemas comuns

- **Erro ao gerar voz (Kokoro)**: confirme que `espeak-ng` está instalado (`espeak-ng --version`).
- **Erro ao transcrever (Whisper)**: confirme que `ffmpeg` está instalado (`ffmpeg -version`).
- **Microfone não pede permissão**: acesse via `http://localhost:8000` (não `127.0.0.1` misturado
  com outro host) — navegadores exigem contexto seguro (localhost conta como seguro) para o
  `getUserMedia`.
