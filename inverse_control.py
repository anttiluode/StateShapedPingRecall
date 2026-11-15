"""Separate engineered sanity control, NOT the grown circuit and NOT a comparable task.

Fixed before this diagnostic's runs: sources at sites 0,4,7, known onset20;
three unknown amplitudes uniform .5..1.5; 32 seed 150112 held-out mixtures.
Observe sites1,6 at frames24..49 with Gaussian noise SD=1% observed RMS.
Recover amplitudes by least squares using three known unit-source responses.
Predict other six sensors at frames50..139, never used for inference.
The forward dictionary is supplied by us, not learned by neurons.
"""
import json
import numpy as np
import run as p

def main():
    th=p.core.calibrate(p.UP)['thresholds']
    zero={'weights':[[0.]*8 for _ in range(3)],'alive':[[1]*8 for _ in range(3)]}
    sites=[0,4,7];basis=np.stack([p.simulate(zero,th,[(s,20,1.)],'impulse',steps=140)['wave'] for s in sites],axis=-1)
    obs=[1,6];held=[0,2,3,4,5,7]
    X=basis[24:50,obs,:].reshape(-1,3);Z=basis[50:140,held,:].reshape(-1,3)
    assert np.linalg.matrix_rank(X)==3
    rng=np.random.default_rng(150112);rows=[]
    for j in range(32):
        truth=rng.uniform(.5,1.5,3)
        actual=p.simulate(zero,th,[(s,20,float(a)) for s,a in zip(sites,truth)],'impulse',steps=140)
        observation=actual['wave'][24:50,obs].ravel();noise=.01*np.sqrt(np.mean(observation**2))
        observation=observation+rng.normal(0,noise,observation.shape)
        inferred=np.linalg.lstsq(X,observation,rcond=None)[0]
        target=actual['wave'][50:140,held].ravel();pred=Z@inferred
        score=1-float(np.sum((pred-target)**2)/np.sum(target**2))
        rows.append({'id':j,'source_amplitudes':truth.tolist(),'inferred':inferred.tolist(),
                     'heldout_wave_score':score,'source_relative_error':float(np.linalg.norm(inferred-truth)/np.linalg.norm(truth)),
                     'clipped':actual['clipped']})
    out={'label':'engineered known-forward-model inverse control; separate task, no learned neurons',
         'observation_sensors':obs,'observation_frames':[24,50],'heldout_sensors':held,'heldout_frames':[50,140],
         'noise_rms_fraction':.01,'matrix_rank':int(np.linalg.matrix_rank(X)),
         'condition_number':float(np.linalg.cond(X)),'results':rows}
    (p.ROOT/'inverse_control.json').write_text(json.dumps(out,indent=2))
    print('Known-model inverse diagnostic:',len(rows),'mixtures; median withheld-wave score',np.median([r['heldout_wave_score'] for r in rows]),
          'median relative amplitude error',np.median([r['source_relative_error'] for r in rows]))

if __name__=='__main__':main()
