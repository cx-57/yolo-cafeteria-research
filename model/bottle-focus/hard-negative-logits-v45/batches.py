"""Bound actual source-list length: Ultralytics in-memory lists ignore batch size."""
def predict_chunks(model,rows):
 for start in range(0,len(rows),4):
  chunk=rows[start:start+4]
  results=model.predict(source=[r['image'] for r in chunk],imgsz=640,batch=4,device='mps',conf=.1,iou=.5,save=False,verbose=False,stream=True)
  yield from zip(chunk,results,strict=True)
