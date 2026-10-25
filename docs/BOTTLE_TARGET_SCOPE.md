# Bottle target scope

User clarification: **only disposable plastic drink bottles**.

Include identifiable single-use plastic water, soft drink, juice and milk bottles, including empty or crushed examples and opaque bottles when their type is clear. A recognizable drink-bottle container can still count when repurposed; RGB images cannot establish its current contents or certify polymer composition.

Exclude glass bottles, reusable sports/insulated bottles, condiment containers, detergent/toiletry bottles, cups, cans and other packaging. Hold uncertain training photos rather than force them into either class. Mixed scenes must label every eligible visible bottle; otherwise hold the whole scene/group until independently reviewed annotations exist.

The preserved246-photo/67-object benchmark is unchanged for continuity. Its inherited labels remain provisional and may include scope inconsistencies. Any later correction must be separately versioned and evaluated against a fresh baseline; it must not be presented as a model accuracy gain caused by training.

Latest implementation and exact source/review/weight identities are in `model/bottle-focus/hard-negative-logits-v45/`. All monitoring is development validation; an independent cafeteria-session test remains outstanding.
