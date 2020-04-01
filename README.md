# Meta Reinforcement Learning with Backpropamine
**Jin Yeom**  

This project reimplements [Backpropamine: training self-modifying neural networks with differentiable neuromodulated plasticity](https://openreview.net/forum?id=r1lrAiA5Ym) (Miconi et al., 2018), which was presented at ICLR 2019. This implementation adds to the original experiment:
- layer normalization for better performance
- non-recurrent Backpropamine agent
- fully observable state (to verify that recurrency is not necessary)
