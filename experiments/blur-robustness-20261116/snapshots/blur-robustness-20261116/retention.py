
def class_operating_retention(results,reference):
 from specialists import NAMES
 checks={}
 for condition in ['clean','gaussian','downsample']:
  checks[condition]={}
  for name in NAMES:
   before=results[reference][condition]['operating']['per_class'][name];after=results['candidate'][condition]['operating']['per_class'][name]
   recall_ok=before['recall'] is None or (after['recall'] is not None and after['recall']>=before['recall']-.02)
   precision_ok=before['precision'] is None or (after['precision'] is not None and after['precision']>=before['precision']-.02)
   checks[condition][name]={'acceptable':recall_ok and precision_ok,'before':before,'after':after}
 return checks
