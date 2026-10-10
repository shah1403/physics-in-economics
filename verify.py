"""Independent arithmetic and reproducibility checks; does not certify all paper claims."""
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit,brentq
from cda import run_cda
from environments import ENVIRONMENTS

out=Path('verification');out.mkdir(exist_ok=True)
y=np.array([1.7,1.8,1.75,1.85,1.9,1.88,1.8,1.86,1.86,1.9,1.94,1.94,2.2,2,2,2,2.06,1.9,2,2,2.225,2,2.1,2.1,2.095,2,2,1.95])
t=np.arange(len(y))
tau=float(curve_fit(lambda x,z:2-.3*np.exp(-x/z),t,y,p0=[5],bounds=(1e-6,1000))[0][0])
pred=2-.3*np.exp(-t/tau)
pd.DataFrame({'transaction':t+1,'t':t,'price':y,'fit':pred,'relative_error_percent':100*abs(pred-y)/y,'paper_error_metric':100*abs(pred-y)}).to_csv(out/'smith_test1.csv',index=False)
result={'smith_source':'Table 2, p.14 of supplied paper; manually transcribed chart-derived values, not original experimental raw tape',
 'smith':dict(n=28,tau=tau,rate=1/tau,r2=float(1-sum((y-pred)**2)/sum((y-y.mean())**2)),rmse=float(np.sqrt(np.mean((y-pred)**2))),within_5pct=int(sum(abs(pred-y)/y<=.05)),paper_metric_below_5=int(sum(abs(pred-y)*100<=5))), 'equilibria':[]}
for name,e in ENVIRONMENTS.items():
 q=brentq(lambda q:e.P_D(q)-e.P_S(q),0,5000)
 assert abs(q-e.Q_star)<1e-3 and abs(e.P_S(q)-e.P_star)<1e-5
 result['equilibria'].append(dict(env=name,q=q,p=float(e.P_S(q))))
v,c=ENVIRONMENTS['symmetric'].values_costs(N=100)
a=run_cda(v,c,2.1,4750,seed=7,max_events=2000)
b=run_cda(v,c,12345,4750,seed=7,max_events=2000)
assert np.array_equal(a['contracts'],b['contracts'])
assert (a['spreads']>=-1e-10).all()
result['seed_repeat_and_unused_pstar_check']='passed; q_star still affects stopping rule'
code="import hashlib;print(int.from_bytes(hashlib.sha256(b'10000:symmetric:shortage:0').digest()[:8],'big'))"
assert subprocess.check_output([sys.executable,'-c',code])==subprocess.check_output([sys.executable,'-c',code])
result['cross_process_seed_check']='passed'
(out/'independent_checks.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
