"""Source-balanced checkpoint selection and explicit full-training pause guard."""
import math
def validate_authorization(receipt,declaration_sha):
 if not isinstance(receipt,dict) or receipt.get('declaration_sha')!=declaration_sha or receipt.get('full_training_authorized') is not True or not isinstance(receipt.get('user_instruction'),str) or not receipt['user_instruction'].strip():
  raise ValueError('User previously requested pause before full training; explicit authorization for this declaration is required')
def valid(values):
 if len(values)!=4 or not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values) or not (0<=values[1]<=values[0]<=1 and 0<=values[3]<=values[2]<=1):raise ValueError('Invalid source monitor metric pairs')
def rank(candidate,baseline):
 valid(candidate);valid(baseline);delta=[a-b for a,b in zip(candidate,baseline,strict=True)]
 return min(delta),sum(delta)/4
def eligible(candidate,baseline):
 worst,mean=rank(candidate,baseline);return worst>=0 and mean>=.01-1e-12
