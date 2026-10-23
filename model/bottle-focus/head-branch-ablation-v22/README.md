# Classification versus regression head ablation

Predeclare two saved-model comparisons using frozen-feature compatible
sources: e222 (83.7287% ordinary) and its quarter quality-head blend 88877
(84.0711% ordinary). Start from e222 and replace either its cv2 regression
branch or its cv3 classification branch with the corresponding 88877 branch.
All backbone tensors and BatchNorm running statistics must match exactly.

Before scoring, real-model tests must show the hybrid's decoded boxes and
class scores equal the intended source outputs exactly on the same input.
Evaluate ordinary 640 inference, batch 4, NMS 0.5 on CPU with two threads;
record device explicitly. Validation remains 246 photos / 67 boxes, with no
training or label changes and no independent test claim. No score claimed.
