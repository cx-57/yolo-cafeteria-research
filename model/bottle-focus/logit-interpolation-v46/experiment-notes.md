# Completed bounded logit interpolation experiment

Two fixed interpolation strengths (.25,.5) of V45 epoch-one score-only update; no training. Initial model and all original benchmark image/label hashes verified before and after. Saved checkpoints roundtrip-exact; every nonlogit tensor remains bitwise identical to incumbent. Four focused tests and237project tests pass; independent review found no concrete P1/P2.

initial: retention/OI/litter AP50/AP95 percent 99.50000, 90.77698, 61.79847, 51.38350, 47.56057, 28.20333

alpha-0.25: retention/OI/litter AP50/AP95 percent 99.50000, 90.89658, 66.54529, 54.09795, 47.54747, 28.49221

alpha-0.5: retention/OI/litter AP50/AP95 percent 99.50000, 90.66619, 72.73527, 58.09571, 48.25465, 28.21730

Alpha.25 failed litter AP50 nonregression; alpha.5 passed all unchanged source gates and alone reached the original246photo67box benchmark. Original result83.338353%AP50/61.072597%AP95; both regress from84.071068%/62.060566%. No promotion. Original benchmark and both protected incumbents unchanged. Source gains did not transfer. Public monitors are repeatedly used development data; retention was previously seen by starting model. No independent NHS test or90% success claimed.
