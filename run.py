"""Reproducible exploratory recall assay; see PROTOCOL.md. Requires numpy, opencv-python-headless."""
from pathlib import Path
import sys, json, copy, time, argparse, hashlib
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import importlib.util

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('core009', ROOT/'vendor/experiments/009_morphology_as_memory/core.py')
core=importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
UP=ROOT/'vendor/up'
MODES=('impulse','fixed_early','fixed_late','state')
SITES={'A':0,'B':7,'C':4}
GROW={'budget':1.,'learning_rate':.2,'trace_decay':.97,'prune_threshold':.02}

def simulate(st, th, sequence, mode, learn=False, steps=400, replay=None):
    graph=core.build_graph(st,th,learn=int(learn),params=GROW)
    obj=core.instantiate(graph,UP)
    # Sources bypassed by explicitly timed experimental drive, with the same one-edge latency.
    edges=[e for e in graph['edges'] if e['from_node_id'] not in (0,1)]
    history=[deque(maxlen=51) for _ in core.UNIT_IDS]
    pending=[np.zeros(8) for _ in core.UNIT_IDS]
    records=[]; read=[]; clipped=0
    replay_map={} if replay is None else {(e['tick'],e['unit']):e for e in replay}
    for tick in range(steps):
        for n in obj.values(): n.pre_step()
        for e in edges:
            a=obj[e['from_node_id']];v=a.get_output(e['from_port'])
            if v is not None: obj[e['to_node_id']].set_input(e['to_port'],v,a.outputs[e['from_port']],e.get('learned_weight',1.))
        for site,onset,amp in sequence:
            if onset+1 <= tick < onset+3: obj[2].set_input('write_%d'%site,.12*amp)
        # Input here is exactly the one about to be consumed by the unchanged fade node.
        for k,u in enumerate(core.UNIT_IDS): history[k].append(float(obj[u].get_blended_input('signal_in','sum') or 0.))
        for n in obj.values(): n.step()
        for k,g in enumerate(core.GROW_IDS):
            # Grower sees previous-frame pulse under the actual host schedule.
            pre=float(obj[g].get_blended_input('pre_spike','sum') or 0.)
            out=pending[k].copy();pending[k][:]=0
            event=replay_map.get((tick,k)) if replay is not None else None
            fire=bool(event) if replay is not None else bool(pre)
            if fire:
                h=np.asarray(list(history[k])[:-1],dtype=float)  # excludes the sample after this spike's emission
                q=float(np.var(h)/(np.mean(h*h)+1e-30)) if h.size else 0.
                late=bool(event['late']) if event else (mode=='fixed_late' or (mode=='state' and q>.5))
                coeff=np.array([1.,0.]) if mode=='impulse' else np.array([.2,.8] if late else [.8,.2])
                weights=np.asarray(obj[g].weights)
                out += .24*coeff[0]*weights
                pending[k] += .24*coeff[1]*weights
                records.append({'tick':tick,'unit':k,'late':late,'q':q})
            obj[g].out=out.tolist()
        read.append(obj[2].read_values.copy())
        clipped += int(np.max(np.abs(obj[2].field))>=3.9998 or np.max(np.abs(obj[2].vel))>=3.9998)
    st2={'weights':[list(map(float,obj[g].weights)) for g in core.GROW_IDS],
         'alive':[list(map(int,obj[g].alive)) for g in core.GROW_IDS]}
    return {'structure':st2,'wave':np.asarray(read),'events':records,'clipped':clipped}

def sequence(order,onset,gap,amp=1.):
    return [(SITES[c],onset+i*gap,amp) for i,c in enumerate(order)]

def score(y,base,target,other,window):
    a,b=window;p=(y-base)[a:b];t=target[a:b];o=other[a:b]
    denom=float(np.sum(t*t));err=float(np.sum((p-t)**2));wrong=float(np.sum((p-o)**2))
    return {'score':1-err/denom,'own_preferred':bool(err<wrong-1e-12),'own_sse':err,'other_sse':wrong,
            'target_norm2':denom,'response_norm2':float(np.sum(p*p))}

def job(arg):
    seed,mode,th,episodes=arg
    rng=np.random.default_rng(seed);gap=int(rng.integers(28,43));amp=float(rng.uniform(.9,1.12))
    schedule=[(20+int(rng.integers(-5,6)),gap+int(rng.integers(-3,4))) for _ in range(episodes)]
    probe_onset=23;window=(probe_onset+gap+2,probe_onset+2*gap+100)
    zero={'weights':[[0.]*8 for _ in range(3)],'alive':[[1]*8 for _ in range(3)]}
    base=simulate(zero,th,sequence('A',probe_onset,gap),'impulse')['wave']
    targets={o:simulate(zero,th,sequence(o,probe_onset,gap),'impulse')['wave']-base for o in ('ABC','ACB')}
    untrained=simulate(core.uniform_structure(),th,sequence('A',probe_onset,gap),mode)
    results=[]
    for order in ('ABC','ACB'):
        st=core.uniform_structure();train_clip=0;train_late=train_events=0
        for onset,g in schedule:
            r=simulate(st,th,sequence(order,onset,g,amp),mode,learn=True)
            st=r['structure'];train_clip+=r['clipped'];train_late+=sum(e['late'] for e in r['events']);train_events+=len(r['events'])
        cue=sequence('A',probe_onset,gap);trained=simulate(st,th,cue,mode)
        perm=np.random.default_rng(seed+4000);scr=copy.deepcopy(st)
        for k in range(3):
            ix=perm.permutation(8);scr['weights'][k]=np.asarray(st['weights'][k])[ix].tolist();scr['alive'][k]=np.asarray(st['alive'][k])[ix].tolist()
        scrambled=simulate(scr,th,cue,mode);silent=simulate(st,th,[],mode)
        other='ACB' if order=='ABC' else 'ABC'
        variants={'trained':trained,'no_growth':untrained,'scrambled':scrambled}
        if mode=='state':
            ev=copy.deepcopy(trained['events'])
            for k in range(3):
                ids=[i for i,e in enumerate(ev) if e['unit']==k];bits=[ev[i]['late'] for i in ids];perm.shuffle(bits)
                for i,bit in zip(ids,bits): ev[i]['late']=bit
            variants['payload_shuffled']=simulate(st,th,cue,mode,replay=ev)
            replay_same=simulate(st,th,cue,mode,replay=trained['events'])
            assert np.array_equal(replay_same['wave'],trained['wave']), 'replay timing mismatch'
            payload_changed=sum(a['late']!=b['late'] for a,b in zip(ev,trained['events']))
        else: payload_changed=None
        rec={'seed':seed,'mode':mode,'order':order,'gap':gap,'training_amplitude':amp,'window':window,'structure':st,
             'training_clipped_frames':train_clip,'training_events':train_events,'training_late':train_late,
             'probe_events':trained['events'],'probe_clipped_frames':trained['clipped'],
             'silent_peak':float(np.max(np.abs(silent['wave']))),'payload_changed':payload_changed,
             'metrics':{name:score(r['wave'],base,targets[order],targets[other],window) for name,r in variants.items()}}
        rec['oracle_score']=score(base+targets[order],base,targets[order],targets[other],window)['score']
        assert rec['oracle_score']>1-1e-12 and rec['silent_peak']==0
        results.append(rec)
        if seed in (120001,991):
            path=ROOT/'traces';path.mkdir(exist_ok=True)
            np.savez_compressed(path/f'{seed}_{mode}_{order}.npz',base=base,target=targets[order],other=targets[other],
                                trained=trained['wave'],untrained=untrained['wave'],scrambled=scrambled['wave'])
    return results

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--workers',type=int,default=4);args=ap.parse_args()
    th=core.calibrate(UP)['thresholds'];start=time.time()
    seeds=[991] if args.smoke else list(range(120001,120013));episodes=2 if args.smoke else 20
    receipt={'label':'smoke plumbing only' if args.smoke else 'exploratory 12-seed paired pilot',
             'thresholds':th,'episodes':episodes,'seeds':seeds,'results':[],
             'source_shas':{'PerceptionLabAI':'54b20b90b9356d7d622695ad9072b07ce569c5f7','PerceptionLab':'0291f579123ce4261aade4e6c5a7b040f8177f1e'},
             'protocol_sha256':hashlib.sha256((ROOT/'PROTOCOL.md').read_bytes()).hexdigest(),
             'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out=ROOT/('smoke.json' if args.smoke else 'results.json')
    jobs=[(seed,mode,th,episodes) for seed in seeds for mode in MODES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n,rows in enumerate(pool.map(job,jobs),1):
            receipt['results']+=rows;receipt['elapsed_seconds']=time.time()-start
            out.write_text(json.dumps(receipt,indent=2));print(f'{n}/{len(jobs)} paired conditions complete; {time.time()-start:.1f}s',flush=True)
    for mode in MODES:
        rows=[r for r in receipt['results'] if r['mode']==mode]
        print(mode,'median score',np.median([r['metrics']['trained']['score'] for r in rows]),
              'own preference',sum(r['metrics']['trained']['own_preferred'] for r in rows),
              'shape late / total',sum(r['training_late'] for r in rows),sum(r['training_events'] for r in rows),flush=True)

if __name__=='__main__':main()
