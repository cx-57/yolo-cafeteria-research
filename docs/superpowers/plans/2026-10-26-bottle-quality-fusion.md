# Localization-aware ensemble score test

> Execute inline withinuser's20minwindow ending03:24:46UTC.

**Evidence:** FixedV51ensemblefreshmain85.3558%AP50/62.13294%AP95 improvedbothmetrics butmissed86–87/~65aspiration. PreserveV51recipe/checkpoints. Testwhetherlesscertainboxpredictionsshouldhavelessranking/fusioninfluence.

**Method:** Onefixedheuristic,bothmodels:softmaxeachside's16-binDFLdistribution;sumtop2probabilitiesforeachside;average4sides;scoremultiplier=sqrt(meanmass). Multiplyonlyclassprobabilitiesbeforeper-modelNMS/WBF. Rawdecodedboxesunchanged. Samefrozen88877+95370modelpairandWBFsettingsasV51. No training,sweep,labelchange. InspiredbydistributionqualityestimationinGFLV2(https://arxiv.org/abs/2011.12885),butthisisNOTareproductionofthepaper'slearnedDGQP andnoautomaticgainisassumed.

**Selection:** Exactsame100developmentphotos,sourcefloorsnowagainstV51ensemble(all6nonretentionmetricsnonregressing,mean gain>=.01;retentioneachwithin.005). Max1maincandidateifgatespass. MainbothmetricsmustnonregressfromV51actual85.3558/62.13294forpromotion. Same246/67unchangedlabels,explicitquality-awaretwo-modelensembleprotocol. Aspiration86/65,notpromise. Stop bydeadline;preserveallnondominatedresults.

- [ ] Failingqualitytests:peakeddistributionsweightaboveuniform,uniformtop2mass=2/16;finite/range/shapevalidation;actualheaddecodedboxesequalwhileclassscoreschange;fullsyntheticvalidatorstillfusesallframes.
- [ ] Bindnewcode/qualityconfigurationandV51receipts;independentreviewbeforemeasurement. Evaluateonefixedsourcecandidate,thenmax1gatedmainprobe. Verifycode/weights/frozenimagesandarchivebestrecipe+metricsbeforedeadline.
