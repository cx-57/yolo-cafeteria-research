"""Retention and source gates; source-only gains cannot authorize a benchmark probe."""
import math
def valid(values):
 if len(values)!=6 or not all(isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) for v in values) or not all(0<=values[i+1]<=values[i]<=1 for i in [0,2,4]):raise ValueError('Invalid three-cohort AP pairs')
def eligible(candidate,baseline):
 valid(candidate);valid(baseline);delta=[a-b for a,b in zip(candidate,baseline,strict=True)]
 return min(delta[:2])>=-.005-1e-12 and min(delta[2:])>=0 and sum(delta[2:])/4>=.01-1e-12
def rank(candidate,baseline):
 delta=[a-b for a,b in zip(candidate,baseline,strict=True)];return min(delta[:2]),sum(delta[2:])/4
def choose(candidates,baseline):
 initial=next(c for c in candidates if c['role']=='initial');accepted=[c for c in candidates if c['role']!='initial' and eligible(c['values'],baseline)]
 return max(accepted,key=lambda c:(*rank(c['values'],baseline),-c['epoch'])) if accepted else initial
def receipt(report,expected):
 assert all(report.get(k)==v for k,v in expected.items()),'Saved score belongs to another model/cohort/protocol'
 a,b=report['AP50'],report['AP50_95']
 if not all(isinstance(v,(float,int)) and math.isfinite(v) for v in [a,b]) or not 0<=b<=a<=1:raise ValueError('Invalid AP pair')
 return report
