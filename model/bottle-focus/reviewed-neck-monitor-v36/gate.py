"""Predeclared minimum monitor gain before reusing the fixed benchmark."""
import math
MIN_AP50=0.8058415166393892
MIN_AP95=0.6373210856519369
def eligible(report):
 a,b=report['AP50'],report['AP50_95']
 if not all(math.isfinite(v) for v in (a,b)) or not 0<=b<=a<=1:
  raise ValueError('Invalid monitor pair')
 return a>=MIN_AP50 and b>=MIN_AP95
