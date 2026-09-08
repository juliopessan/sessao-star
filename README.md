# Sessão STAR

![Sessão STAR — tela inicial](docs/screenshot.png)

## Por que isso existe

Todo mundo que já passou por um processo seletivo ouviu o conselho: "treine suas respostas pelo método STAR". O problema nunca foi saber o que é STAR — Situação, Tarefa, Ação, Resultado. O problema é treinar isso *em voz alta*, contra uma pergunta que realmente tem a ver com o seu currículo e com a vaga específica que você está disputando, sem precisar convencer um amigo a bancar o recrutador de novo, ou pagar por um coach só pra ouvir você falar por vinte minutos.

As alternativas de sempre também esbarram em outro problema: bancos de perguntas genéricas não sabem que você liderou um time de quatro pessoas na Norteluz Logística, e ferramentas de voz por IA de verdade normalmente significam mandar seu currículo e sua voz pra alguma API paga na nuvem, sessão após sessão.

A Sessão STAR nasceu pra resolver as duas coisas ao mesmo tempo: perguntas que cruzam o *seu* currículo com a JD *dessa* vaga, e uma entrevista falada de verdade — Kokoro TTS fazendo a voz do recrutador, Whisper transcrevendo sua resposta — rodando inteiramente no seu computador, sem custo por minuto e sem sua voz saindo da máquina.

## O que ela faz

1. Você envia o currículo (PDF, DOCX, TXT ou colado direto) e cola a descrição da vaga (JD).
2. O Claude cruza as duas fontes e monta 6 perguntas comportamentais no método STAR — inclusive perguntas que sondam de propósito um requisito da vaga que seu currículo ainda não comprova.
3. Antes de valer alguma coisa, uma rodada de treino com um tutor explica o método rapidinho e dá feedback sobre uma resposta de aquecimento, sem contar pra nota final.
4. A entrevista de verdade começa: um recrutador simulado fala cada pergunta em voz alta, você responde pelo microfone, e a transcrição aparece na tela — editável, caso o Whisper entenda alguma palavra errada.
5. No fim, um relatório aponta, pergunta por pergunta, onde sua resposta cobriu Situação, Tarefa, Ação e Resultado — e onde faltou.
6. A sessão fica salva no seu histórico local. Depois de acumular respostas suficientes, um modelo treinado nos seus próprios padrões de resposta passa a mirar as próximas perguntas exatamente nos temas onde você costuma deixar a peteca cair.

## Como funciona por baixo

```
navegador  <-- HTML/CSS/JS puro, sem build step
    |
    | fetch()
    v
FastAPI (backend/main.py)
    |-- /api/extract-resume  -> pypdf / python-docx
    |-- /api/questions       -> Claude API (fallback: heurística local)
    |-- /api/tts             -> Kokoro TTS   (lock: 1 chamada por vez)
    |-- /api/stt             -> Whisper      (lock: 1 chamada por vez)
    |-- /api/tutor-feedback  -> Claude API (fallback: heurística local)
    |-- /api/report          -> Claude API (fallback: heurística local)
    |-- /api/session/save    -> SQLite (backend/db.py)
    |-- /api/insights        -> modelo preditivo (backend/ml.py)
```

Tudo passa pelo backend, inclusive a geração de voz e a transcrição — não porque o navegador não saiba gravar áudio, mas porque o Kokoro e o Whisper são modelos Python de verdade, com pesos que passam de 100 MB, e não existe versão deles rodando puramente em JavaScript no seu navegador. A chave da Claude API segue a mesma lógica de sempre: nunca sai do servidor.

**Histórico e modelo preditivo.** Cada sessão finalizada — currículo, JD, perguntas, respostas, relatório — é salva em SQLite (`backend/db.py`). Uma lista de palavras-chave decide, resposta por resposta, quais dos quatro elementos do STAR parecem cobertos: é o "professor" que rotula os dados. Com menos de 20 respostas reais no histórico, é só essa heurística que fala. A partir da vigésima, `ml.py` treina um classificador de texto (TF-IDF + regressão logística, um por elemento do STAR) sobre tudo que já foi respondido, e passa a usar esse modelo em vez da lista fixa — capturando um pouco do jeito como *você* escreve, não só palavras isoladas. Ele reaprende a cada sessão salva, e o tema onde sua cobertura histórica é mais baixa entra direto no prompt da próxima geração de perguntas.

**Limitação honesta:** é um classificador simples, treinado só com o que você mesmo pratica nessa máquina — ele não avalia se sua resposta é *boa*, só se ela parece cobrir estruturalmente os quatro elementos do método, e precisa de dezenas de respostas antes de dizer algo que a heurística de palavras-chave já não diria. E como o Kokoro e o Whisper não são seguros para chamadas concorrentes, o backend serializa as duas com um lock — ótimo para um candidato treinando sozinho, um gargalo se um dia isso precisasse atender várias pessoas ao mesmo tempo.

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

## Estrutura do projeto

```
backend/
  main.py            FastAPI: todos os endpoints e o serviço do frontend estático.
  db.py              Persistência em SQLite (backend/data/sessao_star.db, fora do git).
  ml.py              Modelo preditivo (scikit-learn) de cobertura STAR.
  requirements.txt
  .env.example
frontend/
  index.html         Interface única (HTML/CSS/JS puro, sem build step).
docs/
  screenshot.png     Captura usada neste README.
```

O frontend grava a resposta com `MediaRecorder`, envia o áudio para `/api/stt` e recebe o texto
transcrito (não é streaming ao vivo — grava, para, transcreve). A pergunta é sintetizada uma vez
por `/api/tts` e o áudio fica em cache no navegador durante a sessão, então "ouvir de novo" não
gera uma nova chamada ao Kokoro.

**Privacidade:** `backend/data/` (banco SQLite + modelo treinado) e `backend/.env` (sua chave de
API) ficam fora do git — o primeiro contém trechos de currículo e as respostas faladas nas suas
entrevistas de treino.

## Problemas comuns

- **Erro ao gerar voz (Kokoro)**: confirme que `espeak-ng` está instalado (`espeak-ng --version`).
- **Erro ao transcrever (Whisper)**: confirme que `ffmpeg` está instalado (`ffmpeg -version`).
- **Microfone não pede permissão**: acesse via `http://localhost:8000` (não `127.0.0.1` misturado
  com outro host) — navegadores exigem contexto seguro (localhost conta como seguro) para o
  `getUserMedia`.
