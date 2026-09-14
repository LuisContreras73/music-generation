# Cola de entrenamiento desatendida, con reintento automatico.
#
#     powershell -ExecutionPolicy Bypass -File "scripts\19_cola_entrenamiento.ps1"
#
# Por que una cola y no los dos a la vez: la GPU son 12.3 GB y el run de contexto
# 2048 se come ~8.3 GB. En paralelo se arriesga un OOM que mataria al que lleva
# horas, y compartir la tarjeta no acelera: el cuello es la propia GPU.
#
# Por que reintento: este entorno mata el proceso en silencio cada pocas horas
# (sin traza en stderr; ya paso en varios entrenamientos). Como cada run guarda
# last.pt periodicamente y --resume recupera tambien el estado del optimizador,
# relanzar es barato: se pierde como mucho lo andado desde el ultimo checkpoint.
# El bucle relanza hasta que aparece logs/summary.json, que train.py solo escribe
# cuando el entrenamiento TERMINA de verdad.
#
# Orden, y el motivo del orden:
#   1. lstm_96ep -- el lstm es el MEJOR generador medido (78.9 +-3.0 de gen_score
#      pareado frente a 49.3 del mejor en bits/paso) y el MENOS entrenado (3.7
#      epocas equivalentes). Es el que mas puede ganar y el que hay que tener
#      terminado si el dia se corta, asi que va primero.
#   2. estilo_llama_ctx2048_96ep -- mejor bits/paso del laboratorio. Se entrena
#      para ver DONDE satura, no porque se espere que mejore la generacion.
#
# Los dos llevan la generacion durante el entrenamiento DESACTIVADA
# (gen_every enorme): era justo ahi donde el proceso moria. Las generaciones se
# hacen despues con scripts/18_comparacion_pareada.py, que ademas es el protocolo
# comparable.

# Rutas derivadas, no fijadas a esta maquina: el script debe funcionar en el
# equipo de cualquiera que clone el repositorio.
$root  = Split-Path -Parent $PSScriptRoot
$py    = (Get-Command python).Source
$traza = Join-Path $root "experiments\cola.log"
$MAX_INTENTOS = 10

function Nota($texto) {
    "[$((Get-Date).ToString('HH:mm:ss'))] $texto" | Out-File -FilePath $traza -Append -Encoding utf8
}

$cola = @("lstm_96ep", "estilo_llama_ctx2048_96ep")

foreach ($nombre in $cola) {
    $dir     = Join-Path $root "experiments\$nombre\logs"
    $resumen = Join-Path $dir "summary.json"
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }

    if (Test-Path $resumen) { Nota "$nombre ya estaba terminado, se salta"; continue }

    $intento = 0
    $ini = Get-Date
    while ((-not (Test-Path $resumen)) -and ($intento -lt $MAX_INTENTOS)) {
        $intento++
        Nota "$nombre intento $intento"
        try {
            Start-Process -FilePath $py `
                -ArgumentList "-u","src/train.py","--config","experiments/$nombre/config.json","--resume" `
                -WorkingDirectory $root `
                -RedirectStandardOutput (Join-Path $dir "entrenamiento_$intento.log") `
                -RedirectStandardError  (Join-Path $dir "entrenamiento_$intento.err") `
                -NoNewWindow -Wait -ErrorAction Stop
        } catch {
            Nota "   ERROR al lanzar $nombre : $_"
        }
        if (-not (Test-Path $resumen)) { Start-Sleep -Seconds 20 }   # deja que la GPU se libere
    }

    $mins = [math]::Round(((Get-Date) - $ini).TotalMinutes, 1)
    if (Test-Path $resumen) { Nota "$nombre TERMINADO tras $mins min y $intento intento(s)" }
    else { Nota "$nombre ABANDONADO tras $MAX_INTENTOS intentos y $mins min" }
}

Nota "cola terminada"
