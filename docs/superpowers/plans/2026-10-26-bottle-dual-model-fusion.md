# Bounded dual-model bottle fusion plan

> Execute inline; user explicitly authorized20minutes of experiments toward86–87%AP50 androughly65%AP95, accepting slightlyless. Deadline03:24:46UTC.

**Hypothesis:** V42 improved separate source recall/AP but regressed main precision. Balanced88877 protects main ranking/localization. A fixed equal-weight detection ensemble may preserve complementary detections and discount unsupported alarms. Unlike previous flip/rotation fusion, these are different learned feature models.

**Architecture:** Two frozen checkpoint snapshots:88877balanced and95370V42epoch17. Same640batch4MPS inputs, each model ordinary single-view inference; standard upstream WBF after per-modelNMS.5, WBFIoU.55/skip.001/conf_typeavg/weights[1,1]/nooverflow. Reuse reviewed upstream fusion implementation. No training or label edits. Explicitly label all results as two-model ensemble, not a single checkpoint. No sweep.

**Validation:** Bind code/data/ancestor/cohort/model hashes before inference. Four development cohorts100photos (24retention,36OI,32litter,8fresh). Reuse exact640matched baseline source receipts fromV50. Same gate: retentioneach>=baseline-.005; other6APmetrics nonregressing,mean gain>=.01. Max1original246/67candidate ifeligible. Originallabels/hashestayfixed. Promotionrequiresboth84.071068%/62.060566%floors; usernewdesired86–87/65areaspirations,notscore guarantees. Fresh8ispublicpositive-onlydevelopmentdata,notindependentNHS.

- [ ] Failing real-model test: native dual forward concatenates each intended model's outputs, shapes/batches/coordinate frames unchanged, secondmodelregisteredfordevice/precision; originalcheckpointSHAsunchanged. Preserve upstream WBF behavior tests.
- [ ] Implement frozen two-model wrapper and exact tracing; standard validator maps fused predictions back to native image frame. Test actualAutoBackend/validator on synthetic images and saved source shapes.
- [ ] Freeze campaign; independent review; evaluate sourcegates then max1mainprobe. Record modelpair, settings, speed, traces and hashes. Respect20minute deadline and preserve incumbent onfailure.
