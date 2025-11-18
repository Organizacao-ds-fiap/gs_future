# LLM Matchmaker — Recomendador Inteligente de Modelos de Linguagem

## 📌 1. Problema

Nos últimos anos o ecossistema de modelos de linguagem explodiu:
modelos open-source, proprietários, especializados, com diferentes custos, latências, requisitos de hardware e níveis de precisão.

Usuários e equipes técnicas enfrentam um desafio central:

“Qual LLM escolher para minha tarefa específica?”

A escolha equivocada pode gerar:
- custos elevados
- resultados imprecisos
- latência inaceitável
- riscos de privacidade
- modelos incapazes de lidar com o domínio desejado

Na prática, escolher um LLM virou um problema de decisão com múltiplas dimensões.

## 📌 2. Proposta

O LLM Matchmaker é uma PoC que demonstra um sistema inteligente capaz de:

Given:

- Tarefa (ex.: resumo jurídico, classificação de texto, chatbot offline)

- Contexto (domínio, idioma, estilo de saída)

- Restrições do usuário (privacidade, hardware, precisão vs velocidade)

Predict:

Qual LLM tem maior probabilidade de entregar a melhor performance neste cenário.

Modelos considerados nesta PoC:
{ Gemini, Deepseek, Llama-3-70B, Claude-2, GPT-4o }

## 📌 3. Objetivo da PoC

Demonstração compacta, mas robusta, de:

- 📡 API moderna (FastAPI) servindo a inferência
- 🧠 Parsing semântico que transforma texto livre → features do modelo
- 🔌 MCP Server atuando como “ferramenta plugável”
- 🤝 Integração com agentes modernos (ChatGPT, Claude, Cursor, IDEs)
- 🚀 Visão arquitetural de futuro para recomendação automatizada de LLMs

Não é apenas um classificador — é um protótipo funcional de como ferramentas inteligentes podem auxiliar usuários e agentes a navegar o ecossistema de LLMs.

## 📌 4. O que o sistema faz
Entrada

Exemplos reais:

“Preciso gerar resumos jurídicos em português com baixa taxa de alucinação.”

“Quero um modelo rápido e leve para rodar localmente em um chatbot.”

“Preciso classificar reviews em inglês com alta determinismo.”

O sistema interpreta:

- tipo de tarefa
- domínio
- idioma
- tolerância a alucinação
- necessidade de determinismo
- restrições de hardware
- estilo de saída desejado
- requisitos de privacidade

Essas características alimentam o modelo treinado.

Saída

best_model (classe predita)


## Dependências

- **Docker**
- **uv** (gerenciador de ambiente e execução Python)
- **Cliente MCP** (ex.: Claude Desktop)

---

## Como executar

1. **Subir containers**
   ```bash
        docker-compose up -d
    ``` 
2. **Verificar container ativo**
   ```bash
        docker ps
    ```


3. **Entrar no diretorio do server**
   ```bash
        cd src\mcp_server
    ``` 

4. **Atualizar dependencias uv**
   ```bash
        uv sync
    ``` 

4. **Iniciar MCP Server**
   ```bash
        uv run matchmaker_server.py
    ``` 

## Configurar MCP CLient: Claude

1. **Localizar arquivo de configuração (Windows)**
   ```makefile
        C:\Users\<USERNAME>\AppData\Roaming\Claude\claude_desktop_config.json
    ``` 


2. **Inserir configuração de MCP Server**
    ``` json
        {
            "mcpServers": {
                "matchmaker": {
                    # path to uv
                    "command": "C:/Users/<USERNAME>/.local/bin/uv.exe",
                    "args": [
                        "run",
                        "--with",
                        "mcp[cli]",
                        "--with",
                        "requests",
                        %# Absolute path to file 
                        "C:/Users/USENAME/Desktop/gs_future/src/mcp_server/matchmaker_server.py"
                    ]
                }
            }
        }
    ``` 
