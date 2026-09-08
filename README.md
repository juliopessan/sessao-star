# Sessão STAR

Simulador de entrevista por voz: envie o currículo do candidato (PDF, DOCX ou texto colado) e a
descrição da vaga (JD), o Claude cruza os dois para gerar perguntas comportamentais no método STAR
(Situação, Tarefa, Ação, Resultado) — inclusive perguntas que sondam requisitos da vaga que não
aparecem no currículo — e um recrutador simulado conduz a entrevista falando com você — voz gerada
pelo **Kokoro TTS** e transcrita pelo **Whisper**, ambos rodando localmente no seu computador.

Toda sessão fica salva num banco **SQLite** local, e um modelo preditivo (**scikit-learn**) usa
esse histórico para identificar em quais temas suas respostas costumam cobrir menos o método STAR
— e passa a priorizar perguntas nesses temas nas próximas sessões. O projeto literalmente aprende
com o seu uso.

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
                      /api/tutor-feedback (rodada de treino),
                      /api/session/save e /api/insights (histórico + modelo preditivo),
                      e serve o frontend estático.
  db.py              Persistência em SQLite (backend/data/sessao_star.db, fora do git).
  ml.py              Modelo preditivo (scikit-learn) de cobertura STAR.
  requirements.txt
  .env.example
frontend/
  index.html         Interface única (HTML/CSS/JS puro, sem build step).
```

O frontend grava a resposta com `MediaRecorder`, envia o áudio para `/api/stt` e recebe o texto
transcrito (não é streaming ao vivo — grava, para, transcreve). A pergunta é sintetizada uma vez
por `/api/tts` e o áudio fica em cache no navegador durante a sessão, então "ouvir de novo" não
gera uma nova chamada ao Kokoro. As chamadas ao Kokoro e ao Whisper são serializadas com um lock
no backend — os dois modelos não são seguros para chamadas concorrentes (ex.: dois cliques rápidos
em "ouvir de novo").

### Histórico local e modelo preditivo (`db.py` + `ml.py`)

Cada sessão finalizada é salva em SQLite: currículo/JD (resumidos), tema e texto de cada pergunta
e resposta, e o relatório final. A partir daí:

1. **Rótulo heurístico** — uma lista de palavras-chave (`STAR_HINTS` em `main.py`) decide, resposta
   por resposta, quais dos quatro elementos do método (Situação/Tarefa/Ação/Resultado) parecem
   cobertos. Esse é o "professor" que ensina o modelo.
2. **Início frio** — com menos de 20 respostas reais acumuladas, o app usa só essa heurística.
3. **Modelo treinado** — a partir de 20 respostas, `ml.train()` treina um classificador de texto
   (TF-IDF + regressão logística, um por elemento do STAR) sobre todo o histórico, e passa a usar
   esse modelo em vez da lista fixa de palavras-chave para avaliar respostas novas — capturando
   padrões de escrita que a heurística não prevê. Reaprende a cada sessão salva.
4. **Perguntas mais direcionadas** — `ml.weak_theme_profile()` identifica os temas onde sua
   cobertura histórica é mais baixa, e esse sinal entra tanto no prompt do Claude quanto na ordem
   das perguntas de fallback local, priorizando exatamente onde você mais precisa treinar.

O painel "Seu histórico & modelo preditivo", na tela de relatório, mostra esse estado em tempo
real — quantas sessões/respostas já foram registradas, se o modelo já foi treinado, e quais temas
estão mais fracos.

**Privacidade:** `backend/data/` (banco SQLite + modelo treinado) fica fora do git — contém
trechos de currículo e as respostas faladas nas suas entrevistas de treino.

## Problemas comuns

- **Erro ao gerar voz (Kokoro)**: confirme que `espeak-ng` está instalado (`espeak-ng --version`).
- **Erro ao transcrever (Whisper)**: confirme que `ffmpeg` está instalado (`ffmpeg -version`).
- **Microfone não pede permissão**: acesse via `http://localhost:8000` (não `127.0.0.1` misturado
  com outro host) — navegadores exigem contexto seguro (localhost conta como seguro) para o
  `getUserMedia`.
