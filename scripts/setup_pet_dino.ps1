<#
.SYNOPSIS
Prepare PET-DINO Swin-T Visual-G/Text, including compiled MMCV.
#>
param(
    [string]$Python = '',
    [ValidateSet('visual','text')][string]$Mode = 'visual',
    [ValidateSet('cpu','cu118','cu121')][string]$TorchBuild = 'cpu',
    [string]$MMCVWheel = '', [string]$WeightsEndpoint = 'https://huggingface.co',
    [switch]$ImportSource, [switch]$InstallDependencies, [switch]$DownloadWeights,
    [switch]$Check, [switch]$LoadModel, [switch]$All, [switch]$Plan
)
$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'setup_model_common.ps1')
$setupArguments = @('--backend','pet_dino','--mode',$Mode,'--torch-build',$TorchBuild,'--weights-endpoint',$WeightsEndpoint)
if ($MMCVWheel) { $setupArguments += @('--mmcv-wheel',$MMCVWheel) }
foreach ($flag in @('ImportSource','InstallDependencies','DownloadWeights','Check','LoadModel','All','Plan')) {
    if (Get-Variable -Name $flag -ValueOnly) {
        $setupArguments += @{ImportSource='--import-source';InstallDependencies='--install-dependencies';DownloadWeights='--download-weights';Check='--check';LoadModel='--load-model';All='--all';Plan='--plan'}[$flag]
    }
}
Invoke-ModelSetup -BasePython $Python -SetupArguments $setupArguments
