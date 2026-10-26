# Compatible download-only entry; both Tiny and Base are supported.
param(
    [string]$Python = '',
    [ValidateSet('tiny','base')][string]$Variant = 'tiny',
    [string]$WeightsEndpoint = 'https://huggingface.co',
    [switch]$Plan
)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'setup_groundingdino.ps1') -Python $Python -Variant $Variant -WeightsEndpoint $WeightsEndpoint -DownloadWeights -Plan:$Plan
