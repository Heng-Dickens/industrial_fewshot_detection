param(
    [string]$SourceDir = 'D:\Work\JSTongYong\result\hanging_object_det1_0_0_0_1'
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..')).TrimEnd('\')
$sourceRoot = (Resolve-Path -LiteralPath $SourceDir).Path.TrimEnd('\')
$targetRoot = [IO.Path]::GetFullPath((Join-Path $workspaceRoot 'weights\jsty_hanging_object'))
if (-not $targetRoot.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Destination must remain inside the project workspace.'
}
if ((Get-Item -LiteralPath $sourceRoot).Attributes -band [IO.FileAttributes]::ReparsePoint) {
    throw 'Source directory must not be a reparse point.'
}
if (Test-Path -LiteralPath $targetRoot) {
    throw "Destination already exists; no existing model will be overwritten: $targetRoot"
}
$checkParent = Split-Path -Parent $targetRoot
while ($checkParent -and $checkParent.Length -ge $workspaceRoot.Length) {
    if ((Test-Path -LiteralPath $checkParent) -and
        ((Get-Item -LiteralPath $checkParent).Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Destination ancestor is a reparse point: $checkParent"
    }
    $checkParent = Split-Path -Parent $checkParent
}

$prefix = 'jsty_hanging_object_det1_0_0_0_1'
$names = @("$prefix.pt", "${prefix}_fp32.onnx", "${prefix}_int8.onnx", "${prefix}_int8.tflite",
           "${prefix}_config.py", "${prefix}_result.png", 'train_result.png')
$entries = @()
foreach ($name in $names) {
    $sourcePath = [IO.Path]::GetFullPath((Join-Path $sourceRoot $name))
    $destinationPath = [IO.Path]::GetFullPath((Join-Path $targetRoot $name))
    if (-not $sourcePath.StartsWith($sourceRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
        -not $destinationPath.StartsWith($targetRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'An asset escaped the explicitly selected source or destination.'
    }
    $asset = Get-Item -LiteralPath $sourcePath
    if ($asset.PSIsContainer -or ($asset.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw "Expected an ordinary model asset file: $sourcePath"
    }
    $entries += [pscustomobject]@{
        name = $name; source = $sourcePath; destination = $destinationPath
        size_bytes = $asset.Length; sha256 = (Get-FileHash -LiteralPath $sourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

New-Item -ItemType Directory -Path $targetRoot | Out-Null
foreach ($entry in $entries) {
    Move-Item -LiteralPath $entry.source -Destination $entry.destination
    $actual = (Get-FileHash -LiteralPath $entry.destination -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $entry.sha256) {
        throw "Post-move SHA256 verification failed: $($entry.destination)"
    }
}
$manifest = [ordered]@{
    schema = 'industrial_fewshot.model_import.v1'
    imported_at_utc = [DateTime]::UtcNow.ToString('o')
    source_directory = $sourceRoot
    operation = 'move_named_model_assets'
    files = $entries
    default_runtime = 'ONNX Runtime CPU; FP32 raw DFL heads'
    model_class_names = @('hanging object')
    original_deployment_num_classes = 2
    class_note = 'Checkpoint and FP32 ONNX metadata show one actual class; the adapted decoder uses one class.'
    architecture = 'Custom Ultralytics YOLOv26_v8; PT references C2PSA_DA'
    model_license = 'AGPL-3.0, as recorded in the model metadata'
    retained_at_source = @("${prefix}_val.tar.gz")
}
$json = $manifest | ConvertTo-Json -Depth 8
[IO.File]::WriteAllText((Join-Path $targetRoot 'IMPORT.manifest.json'), $json, [Text.UTF8Encoding]::new($false))
[pscustomobject]@{
    moved_files = $entries.Count
    total_bytes = ($entries | Measure-Object -Property size_bytes -Sum).Sum
    destination = $targetRoot
    sha256_verified = $true
    validation_archive_retained = Test-Path -LiteralPath (Join-Path $sourceRoot "${prefix}_val.tar.gz")
} | ConvertTo-Json -Compress
