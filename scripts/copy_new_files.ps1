# Script para copiar archivos nuevos de prev_data a test_set
# Copia archivos que existen en X:\bsicos01\__comun\Physionet\prev_data\training_set
# pero que NO existen en C:\BSICoS\CincChallenge2026\python-example-2026\data\training_set
# Los archivos nuevos se copian a data\test_set

# Definir rutas
$sourceDir = "X:\bsicos01\__comun\Physionet\prev_data\training_set"
$trainingSetDir = "C:\BSICoS\CincChallenge2026\python-example-2026\data\training_set"
$testSetDir = "C:\BSICoS\CincChallenge2026\python-example-2026\data\test_set"

# Validar que existan los directorios de origen y destino
if (-not (Test-Path $sourceDir)) {
    Write-Host "ERROR: Directorio fuente no encontrado: $sourceDir" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $trainingSetDir)) {
    Write-Host "ERROR: Directorio training_set no encontrado: $trainingSetDir" -ForegroundColor Red
    exit 1
}

# Crear directorio test_set si no existe
if (-not (Test-Path $testSetDir)) {
    Write-Host "Creando directorio: $testSetDir" -ForegroundColor Cyan
    New-Item -ItemType Directory -Path $testSetDir -Force | Out-Null
}

# Obtener archivos recursivamente de ambos directorios
Write-Host "Leyendo archivos..." -ForegroundColor Cyan

# Obtener nombres relativos de archivos en training_set actual
$trainingFiles = @{}
Get-ChildItem -Path $trainingSetDir -Recurse -File | ForEach-Object {
    $relativePath = $_.FullName.Substring($trainingSetDir.Length + 1)
    $trainingFiles[$relativePath] = $true
}

Write-Host "Se encontraron $($trainingFiles.Count) archivos en training_set actual" -ForegroundColor Yellow

# Procesar archivos de la fuente
$newFilesCount = 0
$sourceFiles = Get-ChildItem -Path $sourceDir -Recurse -File

Write-Host "Procesando $($sourceFiles.Count) archivos de la fuente..." -ForegroundColor Cyan

foreach ($file in $sourceFiles) {
    $relativePath = $file.FullName.Substring($sourceDir.Length + 1)
    
    # Si el archivo NO existe en training_set, copiar a test_set
    if (-not $trainingFiles.ContainsKey($relativePath)) {
        $destFile = Join-Path $testSetDir $relativePath
        $destDir = Split-Path -Parent $destFile
        
        # Crear directorio de destino si no existe
        if (-not (Test-Path $destDir)) {
            New-Item -ItemType Directory -Path $destDir -Force | Out-Null
        }
        
        Write-Host "Copiando: $relativePath" -ForegroundColor Green
        Copy-Item -Path $file.FullName -Destination $destFile -Force
        $newFilesCount++
    }
}

Write-Host ""
Write-Host "RESUMEN:" -ForegroundColor Yellow
Write-Host "Archivos nuevos copiados: $newFilesCount" -ForegroundColor Green
Write-Host "Ubicación: $testSetDir" -ForegroundColor Cyan
Write-Host "Script completado exitosamente" -ForegroundColor Green
