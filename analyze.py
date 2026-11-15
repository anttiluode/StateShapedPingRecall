"""Summarize the fixed pilot; bootstrap paired schedules, not individual history bodies."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'results.json').read_text());rows=r['results']
assert len(rows)==96
MODES=['impulse','fixed_early','fixed_late','state']
labels=['Original ping','Fixed early shape','Fixed late shape','State-shaped ping']
summary={}
for m in MODES:
    a=[x for x in rows if x['mode']==m]
    assert len(a)==24 and len({x['seed'] for x in a})==12
    summary[m]={'median_score':float(np.median([x['metrics']['trained']['score'] for x in a])),
                'own_preference':sum(x['metrics']['trained']['own_preferred'] for x in a),
                'training_spikes':sum(x['training_events'] for x in a),
                'median_control_scores':{v:float(np.median([x['metrics'][v]['score'] for x in a])) for v in a[0]['metrics']}}
assert all(x['oracle_score']>1-1e-12 and x['silent_peak']==0 for x in rows)
assert sum(x['training_clipped_frames']+x['probe_clipped_frames'] for x in rows)==0
def paired_delta(m):
    vals=[]
    for seed in r['seeds']:
        state=np.mean([x['metrics']['trained']['score'] for x in rows if x['seed']==seed and x['mode']=='state'])
        other=np.mean([x['metrics']['trained']['score'] for x in rows if x['seed']==seed and x['mode']==m])
        vals.append(state-other)
    rng=np.random.default_rng(890);vals=np.asarray(vals)
    boot=vals[rng.integers(0,len(vals),(20000,len(vals)))].mean(axis=1)
    return {'mean_difference':float(vals.mean()),'paired_seed_bootstrap_95':np.quantile(boot,[.025,.975]).tolist()}
contrasts={m:paired_delta(m) for m in MODES[:-1]}
inv=json.loads((ROOT/'inverse_control.json').read_text())
state=[x for x in rows if x['mode']=='state']
out={'summary':summary,'state_minus_other':contrasts,
     'payload_shuffled_changed_bodies':sum(x['payload_changed']>0 for x in state),
     'payload_shuffled_total_changed_events':sum(x['payload_changed'] for x in state),
     'inverse_median_score':float(np.median([x['heldout_wave_score'] for x in inv['results']])),
     'inverse_median_relative_source_error':float(np.median([x['source_relative_error'] for x in inv['results']]))}
(ROOT/'summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(12,7.6),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1,1.2])
ax=fig.add_subplot(gs[0,:]);errs=[100*(1-summary[m]['median_score']) for m in MODES]
bars=ax.barh(labels,errs,color=['#728394','#728394','#728394','#d56841'],height=.6)
ax.invert_yaxis();ax.axvline(100,color='#183e51',ls='--',lw=1.5,label='A-only passive baseline: 100%')
ax.bar_label(bars,fmt='%.0f%%',padding=5);ax.set_xlim(0,215);ax.set_xlabel('Median missing-wave error: 100% = passive A-only baseline; lower is better')
ax.set_title('This state-shaped ping did not produce recall',loc='left',fontweight='bold',fontsize=17)
samples=[]
for o in ['ABC','ACB']:
    d=np.load(ROOT/'traces'/f'120001_state_{o}.npz');samples.append(d)
v=max(float(np.max(np.abs(d['target'][23:180]))) for d in samples)
for j,(o,d) in enumerate(zip(['ABC','ACB'],samples)):
    ax=fig.add_subplot(gs[1,j]);t=np.arange(23,180)
    # Two target-site traces, signed and on a shared y scale: expected vs actual.
    for site,color in [(7,'#256d9b'),(4,'#a04c2a')]:
        ax.plot(t,d['target'][23:180,site],color=color,lw=2,label=f'{"B" if site==7 else "C"} target')
        ax.plot(t,(d['trained']-d['base'])[23:180,site],color=color,ls='--',lw=1.3,label=f'{"B" if site==7 else "C"} produced')
    ax.set_title(f'Learned {o}; given only A',loc='left',fontsize=12)
    ax.set_xlabel('Simulation frame');ax.set_ylabel('Wave beyond passive A response');ax.set_ylim(-v*.3,v*1.2)
    ax.legend(fontsize=8,ncol=2,frameon=False)
fig.savefig(ROOT/'recall_results.png',dpi=160,bbox_inches='tight');plt.close(fig)
