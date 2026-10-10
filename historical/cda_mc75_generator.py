"""
Generate TeX charts and tables for 75 Monte Carlo CDA simulations.

Run:
    python cda_mc75_generator.py

Output:
    cda_mc75_generated_results.tex
    cda_mc75_results.json

Design:
    - 3 S&D environments: original, flat supply, flat demand
    - 3 IC regimes: shortage, middle, surplus
    - 75 total simulations: 25 per S&D environment, split 9/8/8 across ICs
    - CDA mechanics follow the corrected dashboard logic: sequential trader activation,
      standing quote execution, no P* leakage, behavioral s_i aggressiveness update.
"""

import json, math, heapq, time
from pathlib import Path
import numpy as np
from scipy.optimize import curve_fit

MARKETS = {
    "Original": {"a": 0.20, "b": 0.00032, "c": 4.00, "d": 0.00048, "N": 5000},
    "Flat supply": {"a": 1.50, "b": 0.0, "c": 5.00, "d": 0.00100, "N": 5000},
    "Flat demand": {"a": 0.50, "b": 0.00050, "c": 2.80, "d": 0.0, "N": 5000},
}
ICS = {
    "Shortage": {"alpha_b": 0.50, "alpha_s": 1.05, "R": 9},
    "Middle": {"alpha_b": 0.75, "alpha_s": 1.25, "R": 8},
    "Surplus": {"alpha_b": 0.95, "alpha_s": 1.50, "R": 8},
}
MA_WINDOW = 100
MAX_EVENTS = 70000
DELTA0 = 0.01
SPREAD_THRESHOLD = 0.08
BURN_IN = 250

LATEX_MARKET_NAMES = {
    "Original": r"Original sloped S\&D",
    "Flat supply": r"Flat supply",
    "Flat demand": r"Flat demand",
}
COLORS = {"Shortage": "blue!65!black", "Middle": "orange!85!black", "Surplus": "green!55!black"}

def market_arrays(a,b,c,d,N):
    q = np.arange(N) + 0.5
    values = c - d*q
    costs = a + b*q
    pstar = (b*c + a*d)/(b+d)
    qstar = (c-a)/(b+d)
    return values, costs, float(pstar), float(qstar)

def moving_average(x,w=MA_WINDOW):
    x=np.asarray(x,float)
    if len(x)<w: return np.array([]),np.array([])
    return np.arange(w-1,len(x)), np.convolve(x,np.ones(w)/w,mode="valid")

def run_cda(values,costs,pstar,qstar,seed,alpha_b,alpha_s,N):
    rng=np.random.default_rng(seed)
    active_b=np.ones(N,dtype=bool); active_s=np.ones(N,dtype=bool)
    b_ids=list(range(N)); s_ids=list(range(N))
    b_pos=np.arange(N); s_pos=np.arange(N)
    bids=np.full(N,-np.inf); asks=np.full(N,np.inf)
    s_b=np.ones(N); s_s=np.ones(N)
    bid_heap=[]; ask_heap=[]
    contracts=[]; book_events=[]; spreads=[]
    crossed=0; neg=0; no_trade=0
    step_mult=1.5**(1/5)
    stop_contracts=int(min(N,max(100,math.ceil(1.03*qstar))))

    def remove_buyer(i):
        if not active_b[i]: return
        active_b[i]=False
        pos=int(b_pos[i]); last=b_ids[-1]
        b_ids[pos]=last; b_pos[last]=pos; b_ids.pop()
        bids[i]=-np.inf

    def remove_seller(j):
        if not active_s[j]: return
        active_s[j]=False
        pos=int(s_pos[j]); last=s_ids[-1]
        s_ids[pos]=last; s_pos[last]=pos; s_ids.pop()
        asks[j]=np.inf

    def best_bid():
        while bid_heap:
            negp,i=bid_heap[0]; p=-negp
            if active_b[i] and np.isfinite(bids[i]) and abs(bids[i]-p)<1e-12:
                return p,i
            heapq.heappop(bid_heap)
        return None,None

    def best_ask():
        while ask_heap:
            p,j=ask_heap[0]
            if active_s[j] and np.isfinite(asks[j]) and abs(asks[j]-p)<1e-12:
                return p,j
            heapq.heappop(ask_heap)
        return None,None

    def post_bid(i,p):
        bids[i]=float(p); heapq.heappush(bid_heap,(-float(p),int(i)))
    def post_ask(j,p):
        asks[j]=float(p); heapq.heappush(ask_heap,(float(p),int(j)))

    def sample_book(event):
        nonlocal crossed
        if event%20!=0 or event>20000: return
        bb,_=best_bid(); aa,_=best_ask()
        if bb is not None and aa is not None:
            if bb>=aa+1e-10: crossed+=1
            book_events.append(event); spreads.append(aa-bb)

    def execute_at_ask(i):
        nonlocal neg
        aa,j=best_ask()
        if aa is None: return False
        if aa>values[i]+1e-10: neg+=1
        if aa<costs[j]-1e-10: neg+=1
        contracts.append(float(aa)); remove_buyer(i); remove_seller(j); return True

    def execute_at_bid(j):
        nonlocal neg
        bb,i=best_bid()
        if bb is None: return False
        if bb>values[i]+1e-10: neg+=1
        if bb<costs[j]-1e-10: neg+=1
        contracts.append(float(bb)); remove_seller(j); remove_buyer(i); return True

    for event in range(MAX_EVENTS):
        if not b_ids or not s_ids: break
        if len(contracts)>=stop_contracts: break
        if len(contracts)>int(0.90*qstar) and no_trade>12000: break
        traded=False
        if rng.random()<0.5:
            i=b_ids[int(rng.integers(len(b_ids)))]
            aa,_=best_ask()
            if aa is not None and aa<=values[i]:
                p_acc=min(1.0,s_b[i]*max(0.0,1-aa/max(values[i],1e-12)))
                if rng.random()<p_acc: traded=execute_at_ask(i)
            if not traded:
                bb,_=best_bid()
                new_bid=alpha_b*values[i] if bb is None else bb+DELTA0*s_b[i]
                new_bid=min(new_bid,values[i])
                aa,_=best_ask()
                if aa is not None and new_bid>=aa and aa<=values[i]: traded=execute_at_ask(i)
                else:
                    bb,_=best_bid()
                    if bb is None or new_bid>bb: post_bid(i,new_bid)
                    else: s_b[i]=min(4.0,s_b[i]*step_mult)
        else:
            j=s_ids[int(rng.integers(len(s_ids)))]
            bb,_=best_bid()
            if bb is not None and bb>=costs[j]:
                p_acc=min(1.0,s_s[j]*max(0.0,1-costs[j]/max(bb,1e-12)))
                if rng.random()<p_acc: traded=execute_at_bid(j)
            if not traded:
                aa,_=best_ask()
                new_ask=alpha_s*costs[j] if aa is None else aa-DELTA0*s_s[j]
                new_ask=max(new_ask,costs[j])
                bb,_=best_bid()
                if bb is not None and new_ask<=bb and bb>=costs[j]: traded=execute_at_bid(j)
                else:
                    aa,_=best_ask()
                    if aa is None or new_ask<aa: post_ask(j,new_ask)
                    else: s_s[j]=min(4.0,s_s[j]*step_mult)
        no_trade=0 if traded else no_trade+1
        sample_book(event)
    return np.array(contracts), np.array(book_events), np.array(spreads), crossed, neg

def detect_formation(book_events,spreads,n_contracts):
    if len(spreads)>=40:
        for i in range(len(spreads)-29):
            if np.median(spreads[i:i+30])<SPREAD_THRESHOLD:
                fe=int(book_events[i+29])
                idx=max(BURN_IN,min(1200,int(fe/3.5)))
                idx=min(idx,max(0,n_contracts-MA_WINDOW-10))
                return idx,fe
    return min(500,max(0,n_contracts-MA_WINDOW-10)),None

def fit_stats(contracts,book_events,spreads,crossed,neg,pstar,qstar):
    x,y=moving_average(contracts)
    idx,fe=detect_formation(book_events,spreads,len(contracts))
    mask=x>=idx; xx,yy=x[mask],y[mask]
    result={}
    if len(yy)>=20:
        x0=xx[0]; p0=yy[0]
        def model(t,beta): return pstar+(p0-pstar)*np.exp(-(t-x0)/beta)
        try:
            popt,_=curve_fit(model,xx,yy,p0=[300],bounds=([1e-6],[30000]),maxfev=20000)
            pred=model(xx,popt[0]); res=yy-pred
            mse=float(np.mean(res**2)); rmse=float(np.sqrt(mse)); mae=float(np.mean(np.abs(res)))
            ssr=float(np.sum(res**2)); sst=float(np.sum((yy-np.mean(yy))**2))
            signed=float(np.trapz(res,xx)); absI=float(np.trapz(np.abs(res),xx))
            result.update(beta=float(popt[0]),mse=mse,rmse=rmse,mae=mae,r2=float(1-ssr/sst) if sst>0 else np.nan,signedI=signed,absI=absI,signed_abs_ratio=abs(signed)/absI if absI>0 else np.nan)
        except Exception:
            result.update(beta=np.nan,mse=np.nan,rmse=np.nan,mae=np.nan,r2=np.nan,signedI=np.nan,absI=np.nan,signed_abs_ratio=np.nan)
    else:
        result.update(beta=np.nan,mse=np.nan,rmse=np.nan,mae=np.nan,r2=np.nan,signedI=np.nan,absI=np.nan,signed_abs_ratio=np.nan)
    final=float(np.mean(contracts[-500:])) if len(contracts)>=500 else np.nan
    result.update(n_contracts=len(contracts), final500=final, pstar_dev=final-pstar, formation_idx=idx, formation_event=fe, crossed=crossed, neg=neg)
    if len(x)>0:
        ids=np.linspace(0,len(x)-1,min(42,len(x))).astype(int)
        result["traj"]=[(int(x[i]),float(y[i])) for i in ids]
    else: result["traj"]=[]
    return result

def mean_sd(vals):
    arr=np.array([v for v in vals if np.isfinite(v)],float)
    if len(arr)==0: return np.nan,np.nan
    return float(np.mean(arr)), float(np.std(arr,ddof=1)) if len(arr)>1 else 0.0

def coords(points):
    return " ".join(f"({x},{y:.5f})" for x,y in points)

def build_tex(all_results, rows):
    parts=[]
    parts.append(r'''
% ============================================================
% CDA Monte Carlo Results Addendum
% Requires in preamble:
% \usepackage{pgfplots}
% \pgfplotsset{compat=1.18}
% \usepackage{float}
% \usepackage{booktabs}
% \usepackage{array}
% ============================================================

\subsubsection{Monte Carlo CDA simulations across three supply--demand environments}

To evaluate robustness, the CDA was repeated under three supply--demand environments and three
initial-condition regimes. In total, $75$ Monte Carlo simulations were performed. For each
supply--demand environment, $25$ simulations were run, divided across shortage, middle, and surplus
initial-condition regimes. The random seeds are fixed, so the experiment is exactly reproducible.

The residual-balance statistic follows the convention
\[
\rho_{\mathrm{bal}}=\frac{|I_{\mathrm{signed}}|}{I_{\mathrm{abs}}},
\qquad
I_{\mathrm{signed}}=\int e(n)\,dn,
\qquad
I_{\mathrm{abs}}=\int |e(n)|\,dn.
\]
A smaller value of $\rho_{\mathrm{bal}}$ means that positive and negative residual regions cancel more strongly.

\paragraph{Space for interpretation.}
\vspace{2em}
\noindent\textit{[Add discussion here: compare convergence speed, final $P^*$ deviation, and residual balance across the three supply--demand geometries.]}
\vspace{2em}
''')
    parts.append(r'''
\begin{figure}[H]\centering
\begin{tikzpicture}
\begin{axis}[width=0.95\textwidth,height=8cm,xlabel={$Q$},ylabel={$P$},grid=both,legend style={font=\scriptsize, at={(0.98,0.98)}, anchor=north east}]
''')
    for mn,m in MARKETS.items():
        sc={"Original":"blue","Flat supply":"green!60!black","Flat demand":"purple"}[mn]
        dc={"Original":"red","Flat supply":"orange!90!black","Flat demand":"cyan!60!black"}[mn]
        parts.append(f"\\addplot[{sc}, thick, domain=0:{m['N']}, samples=100] {{{m['a']} + {m['b']}*x}};\\addlegendentry{{{LATEX_MARKET_NAMES[mn]} supply}}\n")
        parts.append(f"\\addplot[{dc}, dashed, thick, domain=0:{m['N']}, samples=100] {{{m['c']} - {m['d']}*x}};\\addlegendentry{{{LATEX_MARKET_NAMES[mn]} demand}}\n")
    parts.append(r'''\end{axis}\end{tikzpicture}
\caption{The three supply--demand environments used in the Monte Carlo CDA simulations.}
\label{fig:cda_mc_three_sd_curves}\end{figure}
''')
    for mn,md in all_results.items():
        pstar=md['meta']['pstar']
        parts.append(f'''\begin{{figure}}[H]\centering
\begin{{tikzpicture}}
\begin{{axis}}[width=0.95\textwidth,height=8cm,xlabel={{Transaction index $n$}},ylabel={{Moving-average price $\bar{{P}}_n$}},grid=both,legend style={{font=\scriptsize, at={{(0.98,0.98)}}, anchor=north east}},title={{{LATEX_MARKET_NAMES[mn]}: 25 Monte Carlo CDA paths}}]
''')
        maxn=1
        for icn in ICS.keys():
            for run in md['ics'][icn]['runs']:
                if run['traj']:
                    maxn=max(maxn,max(x for x,y in run['traj']))
                    parts.append(f"\\addplot[{COLORS[icn]}, opacity=0.16, line width=0.3pt] coordinates {{{coords(run['traj'])}}};\n")
        for icn in ICS.keys():
            runs=md['ics'][icn]['runs']
            max_common=int(np.nanmin([max([x for x,y in r['traj']]) if r['traj'] else 0 for r in runs]))
            grid=np.linspace(MA_WINDOW-1,max_common,50).astype(int) if max_common>MA_WINDOW else []
            mean_curve=[]
            for gx in grid:
                vals=[]
                for r in runs:
                    tr=r['traj']
                    if len(tr)>=2:
                        xs=np.array([x for x,y in tr]); ys=np.array([y for x,y in tr])
                        vals.append(float(np.interp(gx,xs,ys)))
                if vals: mean_curve.append((int(gx),float(np.mean(vals))))
            if mean_curve:
                parts.append(f"\\addplot[{COLORS[icn]}, very thick] coordinates {{{coords(mean_curve)}}};\\addlegendentry{{{icn} mean}}\n")
        parts.append(f"\\addplot[black, dashed, thick, domain=0:{maxn}] {{{pstar:.6f}}};\\addlegendentry{{$P^*={pstar:.3f}$}}\n")
        parts.append(f'''\end{{axis}}\end{{tikzpicture}}
\caption{{Monte Carlo CDA moving-average price paths for the {LATEX_MARKET_NAMES[mn]} environment. Thin curves show individual simulations; thick curves show the mean path for each initial condition.}}
\label{{fig:cda_mc_{mn.lower().replace(' ','_')}_paths}}\end{{figure}}

\paragraph{{Space for analysis.}}
\vspace{{1.5em}}
\noindent\textit{{[Discuss convergence to the equilibrium line, IC-specific speed, and whether stochastic spread is economically reasonable.]}}
\vspace{{1em}}
''')
    parts.append(r'''
\begin{table}[H]\centering\scriptsize
\begin{tabular}{llrrrrrrrr}\toprule
Market & IC & $R$ & $\bar{\beta}$ & $\overline{\Delta P^*}$ & $\overline{\mathrm{RMSE}}$ & $\overline{\mathrm{MSE}}$ & $\overline{R^2}$ & $\overline{\rho_{\mathrm{bal}}}$ & Viol. \\
\midrule
''')
    for row in rows:
        parts.append(f"{LATEX_MARKET_NAMES[row['market']]} & {row['ic']} & {row['n_runs']} & {row['beta_mean']:.2f} & {row['pstar_dev_mean']:.4f} & {row['rmse_mean']:.4f} & {row['mse_mean']:.5f} & {row['r2_mean']:.3f} & {row['signed_abs_ratio_mean']:.3f} & {row['violations_total']} \\\\\n")
    parts.append(r'''\bottomrule\end{tabular}
\caption{Monte Carlo summary statistics. $\Delta P^*$ is final-window deviation from equilibrium. $\rho_{\mathrm{bal}}=|I_{\mathrm{signed}}|/I_{\mathrm{abs}}$ follows the paper's residual-balance convention.}
\label{tab:cda_mc_summary_stats}\end{table}

\paragraph{Space for table interpretation.}
\vspace{2em}
\noindent\textit{[Focus on final $P^*$ deviation, RMSE/MSE, and $\rho_{\mathrm{bal}}$. Do not overread $R^2$, since the CDA path is stochastic and the fitted ODE has only one free parameter.]}
\vspace{2em}
''')
    parts.append(r'''
\begin{table}[H]\centering\scriptsize
\begin{tabular}{llrrrrrr}\toprule
Market & IC & $\beta$ sd & $\Delta P^*$ sd & RMSE sd & $\rho_{\mathrm{bal}}$ sd & Mean contracts & Contract sd \\
\midrule
''')
    for row in rows:
        parts.append(f"{LATEX_MARKET_NAMES[row['market']]} & {row['ic']} & {row['beta_sd']:.2f} & {row['pstar_dev_sd']:.4f} & {row['rmse_sd']:.4f} & {row['signed_abs_ratio_sd']:.3f} & {row['n_contracts_mean']:.0f} & {row['n_contracts_sd']:.1f} \\\\\n")
    parts.append(r'''\bottomrule\end{tabular}
\caption{Monte Carlo dispersion statistics across random seeds.}\label{tab:cda_mc_dispersion_stats}\end{table}
''')
    return "\n".join(parts)

def main():
    all_results={}; rows=[]; start=time.time(); count=0
    for mi,(mn,m) in enumerate(MARKETS.items()):
        values,costs,pstar,qstar=market_arrays(m['a'],m['b'],m['c'],m['d'],m['N'])
        all_results[mn]={"meta":{**m,"pstar":pstar,"qstar":qstar},"ics":{}}
        for ii,(icn,ic) in enumerate(ICS.items()):
            runs=[]
            for r in range(ic['R']):
                seed=2026000+mi*10000+ii*1000+r
                con,be,sp,cross,neg=run_cda(values,costs,pstar,qstar,seed,ic['alpha_b'],ic['alpha_s'],m['N'])
                st=fit_stats(con,be,sp,cross,neg,pstar,qstar)
                st['seed']=seed; runs.append(st); count+=1
                print(f"completed {count}/75")
            all_results[mn]['ics'][icn]={"params":ic,"runs":runs}
            row={"market":mn,"ic":icn,"n_runs":len(runs)}
            for key in ['beta','r2','mse','rmse','mae','signedI','absI','signed_abs_ratio','pstar_dev','final500','formation_idx','n_contracts']:
                mu,sd=mean_sd([x[key] for x in runs]); row[key+'_mean']=mu; row[key+'_sd']=sd
            row['violations_total']=int(sum(x['crossed']+x['neg'] for x in runs))
            rows.append(row)
    tex=build_tex(all_results,rows)
    Path('cda_mc75_generated_results.tex').write_text(tex,encoding='utf-8')
    Path('cda_mc75_results.json').write_text(json.dumps({'runtime_seconds':time.time()-start,'summary_rows':rows},indent=2),encoding='utf-8')
    print('Saved cda_mc75_generated_results.tex and cda_mc75_results.json')

if __name__ == '__main__':
    main()
