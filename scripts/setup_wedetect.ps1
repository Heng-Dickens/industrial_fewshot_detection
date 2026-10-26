<#
.SYNOPSIS
Prepare standard WeDetect Tiny/Base/Large using the official standalone PyTorch implementation.
#>
param(
    [string]$Python = '',
    [ValidateSet('tiny','base','large')][string]$Variant = 'base',
    [ValidateSet('cpu','cu118','cu121','cu124')][string]$TorchBuild = 'cpu',
    [string]$WeightsEndpoint = 'https://huggingface.co',
    [switch]$ImportSource, [switch]$InstallDependencies, [switch]$DownloadWeights,
    [switch]$Check, [switch]$LoadModel, [switch]$All, [switch]$Plan
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'setup_model_common.ps1')
$setupArguments = @('--backend','wedetect','--variant',$Variant,'--torch-build',$TorchBuild,'--weights-endpoint',$WeightsEndpoint)
foreach ($flag in @('ImportSource','InstallDependencies','DownloadWeights','Check','LoadModel','All','Plan')) {
    if (Get-Variable -Name $flag -ValueOnly) {
        $setupArguments += @{ImportSource='--import-source';InstallDependencies='--install-dependencies';DownloadWeights='--download-weights';Check='--check';LoadModel='--load-model';All='--all';Plan='--plan'}[$flag]
    }
}
Invoke-ModelSetup -BasePython $Python -SetupArguments $setupArguments
