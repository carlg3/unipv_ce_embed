Lectures & Suggested Readings:
------------------------------

*   _Reports of errors in the resources below are always welcome_

1.  2023.03.10 (theory)
    
    **Introduction** [\[pdf\]](01-Introduction.pdf)  
    AI spring? Artificial Intelligence, Machine Learning, Deep Learning: facts, myths and a few reflections.
    
2.  2023.03.10 (theory)
    
    **Fundamentals: Artificial Neural Networks** [\[pdf\]](02-ArtificialNeuralNetworks.pdf)  
    Foundations of machine learning: dataset, representation, evaluation, optimization. Feed-forward neural networks as universal approximators.
    
3.  2023.03.17 (theory)
    
    **Flow Graphs and Automatic Differentiation** [\[pdf\]](03-FlowGraphsAutomaticDifferentiation.pdf)  
    Tensorial representation, flow graphs. Automatic differentiation: primal graph, adjoint graph.
    
4.  2023.03.24 (theory)
    
    **Deep Networks** [\[pdf\]](04-DeepNeuralNetworks.pdf)  
    Deeper networks: potential advantages and new challenges. Tensorial layerwise representation. Softmax and cross-entropy.
    
    [Shannon Entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)) (Wikipedia)
    
    [Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy) (Wikipedia)
    
5.  2023.03.24 (theory)
    
    **Learning as Optimization** [\[pdf\]](05-LearningAsOptimization.pdf)  
    Vanishing and exploding gradients. First and second order optimization, approximations, optimizers. Further tricks.
    
    **Aside 1: Exponential Moving Average** [\[pdf\]](Aside1-ExponentialMovingAverage.pdf)
    
    **Aside 2: Predictors** [\[pdf\]](Aside2-Predictors.pdf)  
    From in-sample optimization to out-of-sample generalization.
    
6.  2023.03.31 (theory)
    
    **Convolutional Networks** [\[pdf\]](06-DeepConvolutionalNeuralNetworks.pdf)  
    Convolutional filter, filter banks, feature maps, pooling, layerwise gradients.
    
7.  2023.04.14 (theory)
    
    **Deep Convolutional Neural Networks and Beyond** [\[pdf\]](07-DCNNsAndBeyond.pdf)  
    Some insight into what happens in convolution layers. DCNN architectures. Transfer learning. Working in reverse: image generation. Generative adversarial networks. Autoencoders and segmentation. Object detection.
    
    J Yosinski, J Clune, Y Bengio, H Lipson, "How transferable are features in deep neural networks?" in Advances in Neural Information Processing Systems (NIPS 2014) [\[link\]](https://proceedings.neurips.cc/paper/2014/file/375c71349b295fbe2dcdca9206f20a06-Paper.pdf)
    
8.  2023.04.21 (theory)
    
    **Aside 3: Tensor Broadcasting** [\[pdf\]](Aside3-TensorBroadcasting.pdf)
    
    **Aside 4: Differentiating Algorithms?** [\[pdf\]](Aside4-DifferentiatingAlgorithms.pdf)  
    Graph-based vs. tape-based automatic differentiation. The engineering solutions in TensorFlow and PyTorch.
    
    A Paszke et al. "Automatic differentiation in PyTorch" in Advances in Neural Information Processing Systems (NIPS 2017) [\[link\]](https://openreview.net/pdf?id=BJJsrmfCZ)
    
9.  2023.05.05 (theory)
    
    **Aside 6: Word Embedding** [\[pdf\]](Aside6-WordEmbedding.pdf)  
    Skip-grams, probability distributions of context and center words, training and results, continuous bag of words (CBOW) model.
    
    **Attention and Transformers** [\[pdf\]](09-Attention and Transformers.pdf)  
    Attention as a kernel, attention maps, queries, key and values, attention-based encoder and decoder, transformer architecture, translator.
    
    A Vaswani, N Shazeer, N Parmar, J Uszkoreit, L Jones, A N Gomez, L Kaiser, I Polosukhin, "Attention Is All You Need" in Advances in Neural Information Processing Systems (NIPS 2017) [\[link\]](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
    
10.  2023.05.14 (theory)
    
    **Reinforcement Learning** [\[pdf\]](10-ReinforcementLearning.pdf)  
    A short recap about RL foundations, Markov decision process, state value function, policy, optimality, action value function, Q-learning.
    
11.  2023.05.26 (theory)
    
    **Deep Reinforcement Learning** [\[pdf\]](11-DeepReinforcementLearning.pdf)  
    Integrating DNNs into the RL paradigm, DQN algorithm, policy gradient, Actor-Critic methods, NAF algorithm.
    
12.  2023.06.09 (theory)
    
    **Monte Carlo Tree Search** [\[pdf\]](12-MonteCarloTreeSearch.pdf)  
    Game trees, Monte Carlo strategy, Monte Carlo Tree Search (MCTS), Upper Confidence Bounds applied to Trees (UCT).
    
13.  2023.06.16 (theory)
    
    **Alpha Zero** [\[pdf\]](13-AlphaZero.pdf)  
    MCTS + DNN, network architecture, replacing MCTS rollout with estimation, network training, AlphaZero in continuous spaces (hints).
    
    D J Mankowitz et al., "Faster sorting algorithms discovered using deep reinforcement learning", Nature 618, 257:263 (2023) [\[link\]](https://www.nature.com/articles/s41586-023-06004-9)