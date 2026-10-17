#!/usr/bin/env python3
"""Offline spatial analysis of saved local correspondences; no model execution.

Fit orientation-preserving similarity transforms on checkerboard training cells,
score held-out cells, and compare with shuffled reference coordinates. This is
selected-clip exploratory measurement, never a tracking/identity acceptance rule.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(values, weights):
    order = np.argsort(values, kind='stable')
    return float(values[order[np.searchsorted(np.cumsum(weights[order]), weights.sum()/2)]])


def predict(q, model):
    # XY coordinates; a+ib is rotation times isotropic scale, never reflection.
    a, b, tx, ty = model
    return np.column_stack((a*q[:, 0]-b*q[:, 1]+tx, b*q[:, 0]+a*q[:, 1]+ty))


def least_squares(q, r, w):
    if len(q) < 3:
        return None
    qbar = np.average(q, axis=0, weights=w)
    rbar = np.average(r, axis=0, weights=w)
    x, y = q-qbar, r-rbar
    denominator = (w*(x*x).sum(axis=1)).sum()
    if denominator <= 1e-12:
        return None
    a = float((w*(x*y).sum(axis=1)).sum()/denominator)
    b = float((w*(x[:, 0]*y[:, 1]-x[:, 1]*y[:, 0])).sum()/denominator)
    if not .125 <= np.hypot(a, b) <= 8:
        return None
    t = rbar-predict(qbar[None], [a, b, 0, 0])[0]
    return [a, b, float(t[0]), float(t[1])]


def fit(q, r, w, seed):
    """Fixed candidate budget and least-median selection; no label-driven tuning."""
    if len(q) < 6 or np.linalg.matrix_rank(q-q.mean(0)) < 2:
        return None
    candidates = []
    initial = least_squares(q, r, w)
    if initial is not None:
        candidates.append(initial)
    rng = np.random.default_rng(seed)
    for _ in range(96):
        i, j = rng.choice(len(q), size=2, replace=False, p=w/w.sum())
        x, y = q[j]-q[i], r[j]-r[i]
        denominator = float(x@x)
        if denominator < 4:
            continue
        a = float(x@y/denominator)
        b = float((x[0]*y[1]-x[1]*y[0])/denominator)
        if not .125 <= np.hypot(a, b) <= 8:
            continue
        t = r[i]-predict(q[i:i+1], [a,b,0,0])[0]
        candidates.append([a,b,float(t[0]),float(t[1])])
    if not candidates:
        return None
    scores = [median(np.linalg.norm(predict(q,m)-r,axis=1),w) for m in candidates]
    best = candidates[int(np.argmin(scores))]
    # One fixed refit to the lower-error half, accepted only if training median improves.
    residual = np.linalg.norm(predict(q,best)-r,axis=1)
    use = residual <= median(residual,w)
    refined = least_squares(q[use],r[use],w[use])
    if refined is not None and median(np.linalg.norm(predict(q,refined)-r,axis=1),w) < min(scores):
        best = refined
    return best


def crossfit(q, r, w, seed):
    folds = (q[:,0].astype(int)+q[:,1].astype(int)) % 2
    residuals = np.full(len(q),np.nan)
    models = []
    for parity in [0,1]:
        train = folds==parity
        if train.sum()<6 or (~train).sum()<6:
            return {'available':False,'reason':'Fewer than six cells in a checkerboard fold.'}
        model=fit(q[train],r[train],w[train],seed+parity)
        if model is None:
            return {'available':False,'reason':'No nondegenerate bounded similarity fit.'}
        residuals[~train]=np.linalg.norm(predict(q[~train],model)-r[~train],axis=1)
        models.append({'trainingParity':parity,'modelABTxTy':model,
                       'scale':float(np.hypot(*model[:2]))})
    return {'available':True,'models':models,'heldOutMedianReferenceCells':median(residuals,w),
            'heldOutMeanReferenceCells':float(np.average(residuals,weights=w)),
            'heldOutCoverageWithinCells':{str(radius):float(np.average(residuals<=radius,weights=w)) for radius in [1,2,4]},
            'heldOutResiduals':residuals.tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Choose a fresh output file.')
    report=json.loads(args.report.read_text())
    if report.get('contract')!='bytewave.saved-mask-local-parts.diagnostic.v1' or report.get('completed') is not True:
        raise ValueError('Expected completed local-parts diagnostic.')
    # Verify saved descriptors before interpreting stored correspondence coordinates.
    for row in list(report['anchors'].values())+report['measurements']:
        f=row['localFeatures']
        if f is not None and sha(args.report.parent/f['path'])!=f['sha256']:
            raise ValueError('Feature archive changed.')
    result={'scope':__doc__,'contract':'bytewave.saved-mask-geometry.offline.v1','completed':False,
            'sourceReportSHA256':sha(args.report),'scriptSHA256':sha(Path(__file__)),
            'method':{'fit':'Orientation-preserving similarity, scale [1/8,8], weighted least-median of 96 deterministic sampled pairs plus least-squares candidate; one lower-half refit.',
                      'validation':'Checkerboard two-fold crossfit, six cells minimum per fold, no acceptance decisions.',
                      'null':'16 deterministic independent reference-coordinate permutations per frame/anchor/mode; repeat entire fitting procedure. Descriptive baseline, not a calibrated p value.',
                      'weights':'Saved mask coverage only, no similarity or human-label threshold.',
                      'units':'64x64 feature grid reference cells; 1 cell corresponds to 16 input pixels. Radius 1/2/4 are reported scales, not identity gates.',
                      'limits':'Adjacent train/test cells share encoder context; not independent validation. Deformation, rotation, scale, partial visibility and repeated texture affect fit. Coherent wrong regions can also fit. Selected development frames only.'},
            'measurements':[]}
    for row in report['measurements']:
        out={'index':row['index'],'againstAnchors':{}}
        for anchor,modes in row['localAgainstAnchors'].items():
            if modes is None:
                out['againstAnchors'][anchor]=None
                continue
            out['againstAnchors'][anchor]={}
            for mode in ['majorityReference','interiorReference']:
                original=modes[mode]
                if not original['available']:
                    out['againstAnchors'][anchor][mode]={'available':False,'reason':'Reference cells unavailable.'}
                    continue
                cells=original['cells'];q=np.asarray(cells['queryYX'],dtype=float)[:,::-1];r=np.asarray(cells['matchedReferenceYX'],dtype=float)[:,::-1];w=np.asarray(cells['coverage'],dtype=float)
                seed=20261016+row['index']*100+int(anchor)
                fit_result=crossfit(q,r,w,seed)
                if fit_result['available']:
                    rng=np.random.default_rng(seed+10000)
                    nulls=[crossfit(q,r[rng.permutation(len(r))],w,seed+i*2+1000) for i in range(16)]
                    valid=[n for n in nulls if n['available']]
                    fit_result['nullValidCount']=len(valid)
                    fit_result['nullMediansReferenceCells']=[n['heldOutMedianReferenceCells'] for n in valid]
                    fit_result['nullMedianOfMedians']=float(np.median(fit_result['nullMediansReferenceCells'])) if valid else None
                    fit_result['queryCells']=len(q)
                out['againstAnchors'][anchor][mode]=fit_result
        result['measurements'].append(out)
        print('Offline geometry:',row['index'],flush=True)
    result['completed']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
