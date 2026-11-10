# Sobe o tracking server local: metadados em SQLite, artefatos servidos pelo próprio servidor.
# Usamos --artifacts-destination (proxy de artefatos) em vez de --default-artifact-root ./mlruns:
# com caminho relativo no default-artifact-root, cada CLIENTE grava artefatos relativo ao SEU
# diretório atual — scripts rodados de pastas diferentes espalhariam artefatos pelo disco.
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
& "$root\.venv\Scripts\mlflow.exe" server `
    --backend-store-uri "sqlite:///mlflow.db" `
    --artifacts-destination "./mlartifacts" `
    --host 127.0.0.1 `
    --port 5000 `
    --workers 1
# --workers 1: o padrão (4 processos) disputa o lock de escrita do SQLite e, no Windows, alguns
# requests ficavam pendurados até o timeout de 120 s do cliente. Com Postgres/MySQL pode subir.
