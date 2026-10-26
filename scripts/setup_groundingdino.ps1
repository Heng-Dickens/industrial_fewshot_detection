<#
.SYNOPSIS
Prepare GroundingDINO Tiny/Base. No flags prints a plan; -All installs/downloads/checks.
#>
param(
    [string]$Python = '',
    [ValidateSet('tiny','base')][string]$Variant = 'tiny',
    [ValidateSet('cpu','cu118','cu121','cu124')][string]$TorchBuild = 'cpu',
    [string]$WeightsEndpoint = 'https://huggingface.co',
    [switch]$InstallDependencies, [switch]$DownloadWeights, [switch]$Check,
    [switch]$LoadModel, [switch]$All, [switch]$Plan
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'setup_model_common.ps1')
$setupArguments = @('--backend','groundingdino','--variant',$Variant,'--torch-build',$TorchBuild,'--weights-endpoint',$WeightsEndpoint)
foreach ($flag in @('InstallDependencies','DownloadWeights','Check','LoadModel','All','Plan')) {
    if (Get-Variable -Name $flag -ValueOnly) {
        $setupArguments += @{InstallDependencies='--install-dependencies';DownloadWeights='--download-weights';Check='--check';LoadModel='--load-model';All='--all';Plan='--plan'}[$flag]
    }
}
Invoke-ModelSetup -BasePython $Python -SetupArguments $setupArguments
