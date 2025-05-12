# API Previsão de Ações do Banco do Brasil

## Sobre

Este projeto tem como objetivo prever os preços de fechamento das ações do Banco do Brasil (BBAS3.SA), utilizando o algoritmo Prophet. O modelo é integrado a uma API REST, permitindo a realização de previsões em tempo real e MLflow para monitorar o desempenho do modelo, tanto no treinamento quanto em produção.

## Funcionalidades

✅ Coleta de dados históricos da bolsa de valores via API Yahoo Finance

✅ Treinamento e teste do modelo Prophet

✅ API para previsão com FastAPI

✅ Monitoramento do modelo com MLflow

✅ Ambiente Docker

✅ Scripts automatizados via Makefile


## Configuração

Ciar e ativar um ambiente virtual:

`python3 -m venv .venv`
`source .venv/bin/activate   # Linux/macOS`
`.venv\Scripts\activate      # Windows`     


Treinar o modelo:

`python3 train.py`

Executar a aplicação:

`make build`

`make up`

Acessar: <a href="http://localhost:8000/docs">

Exemplo requisição:

```
{
  "future_dates": ["2025-05-13", "2025-05-14", "2025-05-15"]
}
```

Exemplo de resposta:

```
[
  {
    "Date": "2025-05-13T00:00:00",
    "Price": 27.131381117483162
  },
  ...
]
```

Parar a aplicação:

`make down`