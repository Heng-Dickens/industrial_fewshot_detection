"""Render standalone experiment plots from archived CSV/JSON; no training or inference."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from summarize_yoloe_optimization import ROOT,read,optimization_report_dir


def rows(path):
    return list(csv.DictReader(path.open(encoding='utf-8-sig')))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stamp',default='yoloe_search_20261029_r2')
    parser.add_argument('--dev-only',action='store_true')
    args=parser.parse_args()
    report=optimization_report_dir(args.stamp, project_root=ROOT)
    from summarize_finetuning_best import is_compact
    if is_compact(report):
        print('This report uses the compact layout; historical plots are in the archive recorded in best_evidence.json.')
        return
    scene=read(report/'plan.json').get('scene','chache')
    trials=rows(report/'all_trials.csv')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(17,12),gridspec_kw={'wspace':.72})
    palette={'head':'#8c9cb8','neck':'#3e83a8','full':'#285269','savpe':'#b9a58f'}
    for ax,family,title in zip(axes,['yoloe-head','yoloe-vp'],['Fixed class head (22 trials)','Visual prompting (26 trials)']):
        selected=[r for r in trials if r['method']==family and '_seed' not in r['id'] and r.get('role')!='baseline' and r.get('dev_mAP50_95')]
        selected.sort(key=lambda r:float(r['dev_mAP50_95']))
        y=np.arange(len(selected));scores=[100*float(r['dev_mAP50_95']) for r in selected]
        colors=['#cb6b40' if r['protocol']=='cross_image' else palette[r['scope']] for r in selected]
        ax.barh(y,scores,color=colors,height=.72)
        ax.set_yticks(y,[r['id'] for r in selected],fontsize=8)
        for i,v in enumerate(scores):ax.text(v+.5,i,f'{v:.2f}',va='center',fontsize=8)
        ax.set_xlim(0,max(73,max(scores,default=0)+8));ax.set_xlabel('Dev AP50:95 (%)');ax.set_title(title)
        ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle(f'YOLOE {scene} search on fixed {50 if scene=="chache" else 10} train / 100 dev images\nSeed 0; orange = cross-image support training; different scopes and budgets',fontsize=14)
    fig.savefig(report/'dev_ranking.png',dpi=180,bbox_inches='tight');plt.close(fig)
    statistics_path=report/'seed_statistics.json'
    statistics=read(statistics_path)['rows'] if statistics_path.exists() else []
    if args.dev_only or not statistics:
        print('Saved dev ranking; test seed statistics pending.');return
    baseline=rows(report/'baselines.csv')
    names=['Raw YOLOE VP 5-shot','Raw YOLOE VP 10-shot','Old fixed-head v1',
           'Old VP v1 5-shot','Old VP v1 10-shot','YOLOv8-L v1']
    if scene!='chache':names=[n.replace('Old','Base').replace(' v1','') for n in names]
    entries=[dict(name=n,ap50=float(r['mAP50']),ap95=float(r['mAP50_95']),s50=0,s95=0,color='#9aa3af') for n,r in zip(names,baseline)]
    for r in statistics:
        n='New fixed head' if r['method']=='yoloe-head' else f'New VP {r["prompt_shots"]}-shot'
        entries.append(dict(name=n,ap50=r['mAP50_mean'],ap95=r['mAP50_95_mean'],
            s50=r['mAP50_std'] or 0,s95=r['mAP50_95_std'] or 0,color='#285269' if r['method']=='yoloe-head' else '#cb6b40'))
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
    for ax,key,sd,title in zip(axes,['ap50','ap95'],['s50','s95'],['Test AP50','Test AP50:95']):
        values=[100*r[key] for r in entries];errors=[100*r[sd] for r in entries]
        ax.barh(range(len(entries)),values,xerr=errors,color=[r['color'] for r in entries],capsize=3)
        ax.set_yticks(range(len(entries)),[r['name'] for r in entries])
        for i,(v,e) in enumerate(zip(values,errors)):ax.text(v+e+1,i,f'{v:.2f}',va='center',fontsize=9)
        ax.set_xlim(0,100);ax.set_xlabel('AP (%)');ax.set_title(title);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle('Fixed 300-image test: dev-selected configurations\nNew: training-seed mean +/- sample SD (n=3); historical baselines: one run')
    fig.tight_layout();fig.savefig(report/'test_comparison.png',dpi=180,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,family in zip(axes,['yoloe-head','yoloe-vp']):
        group=next(r for r in statistics if r['method']==family)
        winner=group['dev_selected_config']
        for seed,color in zip([0,1,2],['#285269','#cb6b40','#668c65']):
            name=winner+(f'_seed{seed}' if seed else '')
            values=rows(report/'trials'/name/'results.csv')
            key=next(k for k in values[0] if k.endswith('/mAP50-95(B)'))
            ax.plot([int(v['epoch']) for v in values],[100*float(v[key]) for v in values],label=f'seed {seed}',color=color)
        ax.set_title(winner);ax.set_xlabel('Epoch');ax.set_ylabel('Native dev AP50:95 (%)');ax.legend();ax.grid(alpha=.15)
    fig.suptitle('Native training validation curves (selection fitness differs from unified dev inference)')
    fig.tight_layout();fig.savefig(report/'training_curves.png',dpi=180,bbox_inches='tight');plt.close(fig)
    print('Saved dev_ranking.png, test_comparison.png, training_curves.png')


if __name__=='__main__':main()
