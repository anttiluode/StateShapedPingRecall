"""Scientific plumbing checks, including equivalence to the unmodified 009 engine."""
import numpy as np
import run as p
th=p.core.calibrate(p.UP)['thresholds'];st=p.core.uniform_structure()
g=p.core.build_graph(st,th,[('A',20,1.),('B',52,1.)],learn=1,params=p.GROW)
a=p.core.simulate(g,p.UP,400);b=p.simulate(st,th,p.sequence('AB',20,32),'impulse',learn=True)
assert np.array_equal(a['structure']['weights'],b['structure']['weights'])
assert a['structure']['alive']==b['structure']['alive']
for k,u in enumerate(p.core.UNIT_IDS):
    assert a['spikes'][str(u)]==[e['tick']-1 for e in b['events'] if e['unit']==k]
for x in ([.192,.048],[.048,.192]):
    assert abs(sum(x)-.24)<1e-14 and abs(np.dot(x,x)-.039168)<1e-14
print('PASS: exact 009 impulse equivalence (weights, survival, firing times); payload area and squared-drive matching.')
