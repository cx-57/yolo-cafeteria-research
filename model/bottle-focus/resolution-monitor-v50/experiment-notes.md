# One fixed higher-resolution inference test

No training or weight blending. The balanced88877incumbent is compared atordinary640control andone896candidate, batch4/MPS/NMS.5/conf.001/noTTA. Earlierresolutionchecksusedoldercheckpoint/TTAsettings;thissinglecurrent-checkpointtesthasmatched640controlsandafreshmonitor,notaresolutionsweep. Report896explicitlyasnewinferenceprotocol,neveras640improvement. Original246/67imagesandlabelsremainunchanged.

Fourcohorts:24previouslyseenretention/16boxes,36OpenImages/17boxes,32litter/18boxes,andnew8previouslyuntrainedlitter/9boxes. FreshsourceQAandall1672protectedphotobyte/pixel/rotation-aware-dHash/SIFTscreensplusgroupguardsverified;all8nowpermanenttrainingexclusions. Thefreshmonitorispositive-onlyandpublicsource/pretrainingoverlapremainspossible.Allcohortsaredevelopmentvalidation,notindependentNHSsessions.

Gatesbeforeonepossibleoriginal896probe:retentioneachwithin640-.005;all6otherAPmetricsnonregressing;meanothergain>=.01. Exactscorecachesandattemptlocksbindweights,settings,resolution,cohortandfrozendeclaration. Promotionrequiresboth84.071068%AP50/62.060566%AP95floors;move-onrequiresAP50>=90%sameweights/settingsonoriginalbenchmark. Sixfocusedtestsverifygatefailure,invalidmetrics,stalesettingsreceipt,settings-boundattempt,realfinalizationbranchskipandmax1originalprobe. IndependentreviewfoundnoconcreteP1/P2. Source/data/code/ancestorhashesfrozenbeforeinference. Noaccuracygainclaimedyet.

Completedmatchedsourceevaluations. PercentAP50/AP95control640 -> candidate896:

retention: 99.500000/90.776984 -> 93.500000/88.347273.

openimages: 61.798470/51.383497 -> 75.347256/59.138726.

litter: 47.560575/28.203331 -> 50.208772/28.748327.

fresh-litter: 49.714286/32.264286 -> 36.295238/19.102619.

Retentionandfreshlitterregressed,so896failedunchangedgates. Originalbenchmarkskipped;original640incumbent84.071068%/62.060566%unchanged. Newfresh8photo/9boxbaseline49.714286%/32.264286%illustratesremainingpubliclittergeneralizationweakness,notacorrectionoftheoriginalscore. Allweightsunchanged;no90%success.
