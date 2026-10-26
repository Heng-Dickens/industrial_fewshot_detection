# Each public setup script exposes its own model options.
function Invoke-ModelSetup {
    param([string]$BasePython, [string[]]$SetupArguments)
    $projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
    if (-not $BasePython) {
        $candidate = Join-Path $projectRoot '.venv-yoloe\Scripts\python.exe'
        if (Test-Path -LiteralPath $candidate) { $BasePython = $candidate }
        else { $BasePython = 'python' }
    }
    & $BasePython (Join-Path $projectRoot 'scripts\install_models.py') --python $BasePython @SetupArguments
    if ($LASTEXITCODE -ne 0) { throw "Model setup failed (exit $LASTEXITCODE)." }
}
