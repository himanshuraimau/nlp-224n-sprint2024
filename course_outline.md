# CS224N Course Outline (Spring 2024)

## Week 1: Word Vectors and Python Review

### Lecture 1: Word Vectors (Apr 2)
- **Video**: [Lecture 1: Word Vectors](https://www.youtube.com/watch?v=8rXD5-xhemo)
- **Slides**: [Word Vectors Slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2024-lecture01-wordvecs1.pdf)
- **Required Readings**:
  - [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
  - [Distributed Representations of Words and Phrases](https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf)
- **Optional Readings**:
  - [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
  - [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://aclanthology.org/Q15-1016.pdf)

### Lecture 2: Word Vectors and Language Models (Apr 4)
- **Video**: [Lecture 2: Word Vectors and Language Models](https://www.youtube.com/watch?v=kEMJRjEdNzM)
- **Slides**: [Word Vectors and Language Models Slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2024-lecture02-wordvecs2.pdf)
- **Required Readings**:
  - [Evaluation methods for unsupervised word embeddings](https://aclanthology.org/D15-1036.pdf)
- **Optional Readings**:
  - [A Latent Variable Model Approach to PMI-based Word Embeddings](https://aclanthology.org/Q16-1028.pdf)
  - [Linear Algebraic Structure of Word Senses](https://aclanthology.org/N18-2034.pdf)

### Python Review Session (Apr 5)
- **Time**: 3:30pm - 4:20pm
- **Location**: Gates B01
- **Materials**: [Python Review Slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2024-lecture00-python-review.pdf)
- **Colab**: [Python Review Notebook](https://colab.research.google.com/drive/1oHy4VwrsXdhtLW9jxwtI0mGsUoWnz-D8)

### Assignment 1: Word Vectors
- **Released**: Apr 2
- **Due**: Apr 9
- **Materials**:
  - [Assignment Handout](https://web.stanford.edu/class/cs224n/assignments/a1.pdf)
  - [Starter Code](https://github.com/cs224n/cs224n.github.io/tree/master/assignments/a1)
  - [Latex Template](https://web.stanford.edu/class/cs224n/assignments/a1.tex)

## Week 2: Neural Networks and Dependency Parsing

### Lecture 3: Backpropagation and Neural Network Basics (Apr 9)
- **Video**: [Lecture 3: Neural Networks](https://www.youtube.com/watch?v=8CWyBXsG3-Y)
- **Slides**: [Neural Networks Slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2024-lecture03-neuralnets.pdf)
- **Required Readings**:
  - [Matrix Calculus Notes](http://cs231n.stanford.edu/vecDerivs.pdf)
  - [CS231n Neural Network Notes](http://cs231n.github.io/neural-networks-1/)
- **Optional Readings**:
  - [Learning Representations by Backpropagating Errors](https://www.nature.com/articles/323533a0)
  - [Yes you should understand backprop](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)

### Lecture 4: Dependency Parsing (Apr 11)
- **Video**: [Lecture 4: Dependency Parsing](https://www.youtube.com/watch?v=nC9_RfjYwqA)
- **Slides**: [Dependency Parsing Slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2024-lecture04-dependencyparsing.pdf)
- **Required Readings**:
  - [A Fast and Accurate Dependency Parser using Neural Networks](https://aclanthology.org/D14-1082.pdf)
  - [Universal Stanford Dependencies](https://nlp.stanford.edu/pubs/USD_LREC14_paper_camera_ready.pdf)
- **Optional Readings**:
  - [Incrementality in Deterministic Dependency Parsing](https://aclanthology.org/W04-0308.pdf)
  - [Dependency Parsing](https://www.morganclaypool.com/doi/abs/10.2200/S00169ED1V01Y200901HLT002)

### PyTorch Tutorial Session (Apr 12)
- **Time**: 3:30pm - 4:20pm
- **Location**: Gates B01
- **Materials**: [PyTorch Tutorial Colab](https://colab.research.google.com/drive/1oHy4VwrsXdhtLW9jxwtI0mGsUoWnz-D8)

### Assignment 2: Neural Networks
- **Released**: Apr 9
- **Due**: Apr 18
- **Materials**:
  - [Assignment Handout](https://web.stanford.edu/class/cs224n/assignments/a2.pdf)
  - [Starter Code](https://github.com/cs224n/cs224n.github.io/tree/master/assignments/a2)
  - [Latex Template](https://web.stanford.edu/class/cs224n/assignments/a2.tex)

## Week 3: RNNs and Sequence Models

### Lecture 5: Recurrent Neural Networks (Apr 16)
- **Video**: [Lecture 5: RNNs](https://www.youtube.com/watch?v=6niqTuYFZLQ)
- **Slides**: [RNNs Slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2024-lecture05-rnnlm.pdf)
- **Required Readings**:
  - [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
  - [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- **Optional Readings**:
  - [On the difficulty of training RNNs](http://proceedings.mlr.press/v28/pascanu13.pdf)
  - [Learning long-term dependencies with gradient descent is difficult](https://ieeexplore.ieee.org/document/279181)

### Lecture 6: Sequence to Sequence Models (Apr 18)
- **Video**: [Lecture 6: Seq2Seq](https://www.youtube.com/watch?v=XXtpJxZBa2c)
- **Slides**: [Seq2Seq Slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2024-lecture06-seq2seq.pdf)
- **Required Readings**:
  - [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf)
  - [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/pdf/1409.0473.pdf)
- **Optional Readings**:
  - [Attention and Augmented RNNs](https://distill.pub/2016/augmented-rnns/)
  - [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/pdf/1703.03906.pdf)

### Assignment 3: Dependency Parsing
- **Released**: Apr 18
- **Due**: Apr 30
- **Materials**:
  - [Assignment Handout](https://web.stanford.edu/class/cs224n/assignments/a3.pdf)
  - [Starter Code](https://github.com/cs224n/cs224n.github.io/tree/master/assignments/a3)
  - [Latex Template](https://web.stanford.edu/class/cs224n/assignments/a3.tex)

## Week 4: Transformers and Project Planning

### Lecture 7: Final Projects and LLM intro (Apr 23)
- **Video**: [Lecture 7: Projects and LLMs](https://www.youtube.com/watch?v=8CWyBXsG3-Y)
- **Slides**: [Projects and LLMs Slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2024-lecture07-projects.pdf)
- **Project Materials**:
  - [Project Proposal Handout](https://web.stanford.edu/class/cs224n/assignments/project-proposal.pdf)
  - [Default Final Project](https://web.stanford.edu/class/cs224n/assignments/default-final-project.pdf)

### Lecture 8: Transformers (Apr 25)
- **Video**: [Lecture 8: Transformers](https://www.youtube.com/watch?v=8CWyBXsG3-Y)
- **Slides**: [Transformers Slides](https://web.stanford.edu/class/cs224n/slides/cs224n-2024-lecture08-transformers.pdf)
- **Required Readings**:
  - [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
  - [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- **Optional Readings**:
  - [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
  - [Image Transformer](https://arxiv.org/pdf/1802.05751.pdf)

## Week 5-10: Advanced Topics and Project Work

### Key Topics
1. Pretraining and Fine-tuning
2. Post-training (RLHF, SFT, DPO)
3. Benchmarking and Evaluation
4. Efficient Neural Network Training
5. Speech Brain-Computer Interface
6. Reasoning and Agents
7. Life after DPO
8. ConvNets and Tree Recursive Neural Networks
9. Responsible NLP
10. NLP, Linguistics, and Philosophy

### Important Deadlines
- Project Proposal Due: May 3
- Assignment 4 Due: May 9
- Project Milestone Due: May 22
- Final Project Due: June 6
- Poster Session: June 10

## Additional Resources

### Tutorial Sessions
1. Python Review (Apr 5)
2. PyTorch Tutorial (Apr 12)
3. Hugging Face Transformers Tutorial (May 3)

### Office Hours
- Regular office hours schedule
- Project office hours
- Emergency assistance sessions

### Computing Resources
- Google Colab
- Stanford Computing Clusters
- Cloud Computing Credits (if available)

### Discussion Forums
- Ed Discussion
- Slack Channel
- Piazza (if applicable)

## Study Tips

1. **Before Each Lecture**
   - Review required readings
   - Check lecture slides
   - Prepare questions

2. **After Each Lecture**
   - Review notes
   - Implement concepts
   - Complete practice problems

3. **For Assignments**
   - Start early
   - Use version control
   - Test thoroughly
   - Document well

4. **For Projects**
   - Regular progress updates
   - Clear documentation
   - Reproducible results
   - Meaningful visualizations 