# Meta Reinforcement Learning with Backpropamine
**Jin Yeom**  

This project reimplements [Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity](https://openreview.net/forum?id=r1lrAiA5Ym) (Miconi et al., 2018), which was presented at ICLR 2019. This implementation adds to the original experiment:
- layer normalization for better performance
- non-recurrent Backpropamine agent
- fully observable state (to verify that recurrency is not necessary)

Currently, one of the major limitations that keeps this project from moving forward is that meta reinforcement learning is computationally expensive, as a recurrent network (or, in our case, a plastic network) must differentiate through the entire lifetime with multiple trials. From a different perspective, however, this limitation also suggests a future direction for this project.

## Todo
- [ ] Retroactive neuromodulation with eligibility trace
- [ ] Plastic LSTM
