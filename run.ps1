# Start DocsRAG API Server
Write-Host "Starting DocsRAG API Server on http://127.0.0.1:8000 ..." -ForegroundColor Cyan
Write-Host "Swagger Documentation: http://127.0.0.1:8000/docs" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray

if (Test-Path ".\.venv\Scripts\uvicorn.exe") {
    & ".\.venv\Scripts\uvicorn.exe" src.api.main:app --reload
} elseif (Test-Path ".\venv\Scripts\uvicorn.exe") {
    & ".\venv\Scripts\uvicorn.exe" src.api.main:app --reload
} else {
    python -m uvicorn src.api.main:app --reload
}
