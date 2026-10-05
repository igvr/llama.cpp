#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('baseline',type=Path);p.add_argument('arena',type=Path);p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
h1=np.fromfile(a.baseline,dtype='<u4',count=4);h2=np.fromfile(a.arena,dtype='<u4',count=4)
assert np.array_equal(h1,h2) and h1[0]==0x5141524c
_,cases,steps,nv=map(int,h1);count=cases*steps
assert a.baseline.stat().st_size==a.arena.stat().st_size==16+count*nv*4
x=np.memmap(a.baseline,dtype='<f4',mode='r',offset=16,shape=(count,nv))
y=np.memmap(a.arena,dtype='<f4',mode='r',offset=16,shape=(count,nv))
maxerr=0.;squared=0.;nwrong=0;bitsdifferent=0;kl=[];maxprob=[]
for i in range(count):
 b=np.asarray(x[i],dtype=np.float64);c=np.asarray(y[i],dtype=np.float64)
 assert np.isfinite(b).all() and np.isfinite(c).all()
 error=c-b;maxerr=max(maxerr,float(np.abs(error).max()));squared+=float(error@error)
 bitsdifferent+=int(np.count_nonzero(x[i].view('u4')!=y[i].view('u4')))
 nwrong+=int(b.argmax()!=c.argmax())
 lb=b-b.max();lc=c-c.max();lb-=np.log(np.exp(lb).sum());lc-=np.log(np.exp(lc).sum())
 pb=np.exp(lb);pc=np.exp(lc);kl.append(float(np.sum(pb*(lb-lc))));maxprob.append(float(np.abs(pc-pb).max()))
r=dict(vectors=count,vocab=nv,compared_logits=count*nv,finite=True,bit_different_logits=bitsdifferent,
       max_absolute_error=maxerr,root_mean_squared_error=float(np.sqrt(squared/(count*nv))),
       greedy_disagreements=nwrong,mean_kl_divergence=float(np.mean(kl)),max_kl_divergence=max(kl),max_probability_error=max(maxprob))
a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if bitsdifferent:raise SystemExit('bit-identical full-logit parity gate failed')
