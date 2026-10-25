"""Reject byte/pixel/capture aliases of permanently reserved monitoring data."""
def check(records,monitors):
 for key in ['id','image_sha','pixels','source_group']:
  values=[r[key] for r in records]
  assert all(values) and len(set(values))==len(values),'Repeated training '+key
  assert not set(values)&{r[key] for r in monitors},'Reserved monitor '+key
