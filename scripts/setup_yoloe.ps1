<#
.SYNOPSIS
Prepare pinned YOLOE visual/text/fusion. Legacy installation/download flags remain supported.
#>
param(
    [string]$Python = '',
    [ValidateSet('visual','text','fusion')][string]$Mode = 'visual',
    [ValidateSet('yoloe-v8s-seg.pt','yoloe-v8m-seg.pt','yoloe-v8l-seg.pt','yoloe-11s-seg.pt','yoloe-11m-seg.pt','yoloe-11l-seg.pt')][string]$Model = '',
    [ValidateSet('cpu','cu118','cu121','cu124')][string]$TorchBuild = 'cpu',
    [string]$WeightsEndpoint = 'https://huggingface.co', [string]$ExpectedWeightsSha256 = '',
    [switch]$ImportSource, [switch]$InstallDependencies, [switch]$DownloadWeights,
    [switch]$Check, [switch]$LoadModel, [switch]$All, [switch]$Plan
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'setup_model_common.ps1')
$setupArguments = @('--backend','yoloe','--mode',$Mode,'--torch-build',$TorchBuild,'--weights-endpoint',$WeightsEndpoint)
if ($Model) { $setupArguments += @('--model',$Model) }
if ($ExpectedWeightsSha256) { $setupArguments += @('--expected-weights-sha256',$ExpectedWeightsSha256) }
foreach ($flag in @('ImportSource','InstallDependencies','DownloadWeights','Check','LoadModel','All','Plan')) {
    if (Get-Variable -Name $flag -ValueOnly) {
        $setupArguments += @{ImportSource='--import-source';InstallDependencies='--install-dependencies';DownloadWeights='--download-weights';Check='--check';LoadModel='--load-model';All='--all';Plan='--plan'}[$flag]
    }
}
Invoke-ModelSetup -BasePython $Python -SetupArguments $setupArguments
