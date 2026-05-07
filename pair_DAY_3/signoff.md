# Asker Sign-Off — Day 3

**Asker:** Yosef Zewdu
**Gap status:** Closed

---

Amare's explainer closed the central part of my question: I now have a clear mechanical account of why a steeply falling DPO training loss can coexist with score collapse on held-out evaluation. The key move was naming the exact gap between what DPO optimizes (preference separation — can the model rank chosen above rejected?) and what my prompt asked for (calibrated scoring — can the model express nuanced confidence on a 1–5 scale?). Before reading the explainer, I understood that something went wrong with my judge; I did not have a model for *why* the optimizer had no incentive to preserve score entropy once it had reliably separated the binary. The sentence "the optimizer succeeded at the objective it was given — the problem is that the objective was narrower than the behavior you actually wanted" is the clearest statement of the gap I have seen.

The beta section also moved my understanding. I had assumed beta=0.1 was some kind of general regularizer that would prevent degenerate outputs. I now understand it is specifically a KL constraint on policy drift from the reference model — it preserves distributional proximity, not score geometry. That distinction explains why the judge outputs remained fluent and grammatical (the reference model's language distribution was mostly preserved) while the scoring behavior collapsed (the KL term had nothing to say about ordinal structure).


What I now understand that I did not before: a low DPO training loss is evidence that the model learned to rank, not that it learned to score. Those are related objectives but they are not the same, and conflating them is a production risk in any system that uses a preference-trained model as a calibrated evaluator.
