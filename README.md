# GLiNER API — serviço compartilhado de extração estruturada

API HTTP local (CPU) com GLiNER2.5 multilíngue. Uso: Input, Agente, Radar, Inseg, n8n, etc.

## URL (EasyPanel VPS dedicada)

- Host: `https://nlp-gliner.z23axp.easypanel.host`
- Auth: `Authorization: Bearer <GLINER_API_TOKEN>`
- Health (sem token): `GET /health`

## Endpoints

| Método | Path | Função |
|--------|------|--------|
| GET | `/health` | readiness |
| GET | `/v1/info` | modelo e rotas |
| POST | `/v1/extract/entities` | NER por labels |
| POST | `/v1/classify` | classificação |
| POST | `/v1/extract` | entidades + classify |

## Exemplo

```bash
curl -sS https://nlp-gliner.z23axp.easypanel.host/v1/extract/entities \
  -H "Authorization: Bearer $GLINER_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Clécio Luís é candidato a governador no Amapá em 2026.",
    "labels": ["pessoa", "cargo", "uf", "ano"]
  }'
```

## Env

| Variável | Default | Notas |
|----------|---------|-------|
| `GLINER_MODEL` | `fastino/gliner2.5-multi-v1` | checkpoint HF |
| `GLINER_API_TOKEN` | (obrigatório em prod) | Bearer |
| `GLINER_DEVICE` | `cpu` | `cpu` ou `cuda` |
| `GLINER_WARMUP` | `1` | pré-carrega no boot |
| `HF_HOME` | `/models` | volume de cache do modelo |

## Consumo nos projetos

Defina `GLINER_URL` + `GLINER_API_TOKEN` no serviço cliente. Não embutir o modelo em cada app.
