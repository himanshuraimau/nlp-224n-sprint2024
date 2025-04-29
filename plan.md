the plan is to setup a repo that will will contain all my notes and assingment and projects related to the course cs224n 
2024 spring nlp course

- so the resources and all the papers will be in the resources.md and the readme will contain my todo list and the plan for the course


- then my-notes folder will contain all my notes not the course notes
- assignments folder will contain all my assignments
- projects folder will contain all my projects

here is the full info of the event:
Schedule
Updated lecture slides will be posted here shortly before each lecture. Other links contain last year's slides, which are mostly similar.

Lecture notes will be uploaded a few days after most lectures. The notes (which cover approximately the first half of the course content) give supplementary detail beyond the lectures.

Disclaimer: Assignments change; please do not do old assignments. We will give no points for doing last year's assignments.

Date	Description	Course Materials	Events	Deadlines
Week 1

Tue Apr 2	Word Vectors
[slides] [notes]	Suggested Readings:
Efficient Estimation of Word Representations in Vector Space (original word2vec paper)
Distributed Representations of Words and Phrases and their Compositionality (negative sampling paper)
Assignment 1 out
[code]
[preview]	
Thu Apr 4	Word Vectors and Language Models
[slides] [notes] [code]	Suggested Readings:
GloVe: Global Vectors for Word Representation (original GloVe paper)
Improving Distributional Similarity with Lessons Learned from Word Embeddings
Evaluation methods for unsupervised word embeddings
Additional Readings:
A Latent Variable Model Approach to PMI-based Word Embeddings
Linear Algebraic Structure of Word Senses, with Applications to Polysemy
On the Dimensionality of Word Embedding
Fri Apr 5	Python Review Session
[slides] [colab]	 3:30pm - 4:20pm
Gates B01		
Week 2

Tue Apr 9	Backpropagation and Neural Network Basics
[slides] [notes]	Suggested Readings:
matrix calculus notes
Review of differential calculus
CS231n notes on network architectures
CS231n notes on backprop
Derivatives, Backpropagation, and Vectorization
Learning Representations by Backpropagating Errors (seminal Rumelhart et al. backpropagation paper)
Additional Readings:
Yes you should understand backprop
Natural Language Processing (Almost) from Scratch
Assignment 2 out
[code]
[handout]
[latex template]	Assignment 1 due
Thu Apr 11	Dependency Parsing
[slides] [notes]	Suggested Readings:
Incrementality in Deterministic Dependency Parsing
A Fast and Accurate Dependency Parser using Neural Networks
Dependency Parsing
Globally Normalized Transition-Based Neural Networks
Universal Stanford Dependencies: A cross-linguistic typology
Universal Dependencies website
Jurafsky & Martin Chapter 18
Fri Apr 12	PyTorch Tutorial Session
[colab]	 3:30pm - 4:20pm
Gates B01		
Week 3

Tue Apr 16	Recurrent Neural Networks
[slides] [notes (lectures 5 and 6)]	Suggested Readings:
N-gram Language Models (textbook chapter)
The Unreasonable Effectiveness of Recurrent Neural Networks (blog post overview)
Sequence Modeling: Recurrent and Recursive Neural Nets (Sections 10.1 and 10.2)
On Chomsky and the Two Cultures of Statistical Learning
Sequence Modeling: Recurrent and Recursive Neural Nets (Sections 10.3, 10.5, 10.7-10.12)
Learning long-term dependencies with gradient descent is difficult (one of the original vanishing gradient papers)
On the difficulty of training Recurrent Neural Networks (proof of vanishing gradient problem)
Vanishing Gradients Jupyter Notebook (demo for feedforward networks)
Understanding LSTM Networks (blog post overview)
Thu Apr 18	Sequence to Sequence Models and Machine Translation
[slides] [notes (lectures 5 and 6)]	Suggested Readings:
Statistical Machine Translation slides, CS224N 2015 (lectures 2/3/4)
Statistical Machine Translation (book by Philipp Koehn)
BLEU (original paper)
Sequence to Sequence Learning with Neural Networks (original seq2seq NMT paper)
Sequence Transduction with Recurrent Neural Networks (early seq2seq speech recognition paper)
Neural Machine Translation by Jointly Learning to Align and Translate (original seq2seq+attention paper)
Attention and Augmented Recurrent Neural Networks (blog post overview)
Massive Exploration of Neural Machine Translation Architectures (practical advice for hyperparameter choices)
Achieving Open Vocabulary Neural Machine Translation with Hybrid Word-Character Models
Revisiting Character-Based Neural Machine Translation with Capacity and Compression
Assignment 3 out
[code]
[handout]
[latex template]
[overleaf link]	Assignment 2 due
Week 4

Tue Apr 23	Final Projects and LLM intro
[slides]	Suggested Readings:
Practical Methodology (Deep Learning book chapter)
Project Proposal out
[handout]

Default Final Project out
[handout]	
Thu Apr 25	Transformers
(by Anna Goldie)
[slides] [notes]	Suggested Readings:
Attention Is All You Need
The Illustrated Transformer
Transformer (Google AI blog post)
Layer Normalization
Image Transformer
Music Transformer: Generating music with long-term structure
Jurafsky and Martin Chapter 10 (Transformers and Large Language Models)
Week 5

Tue Apr 30	Pretraining
[slides]	Suggested Readings:
BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Contextual Word Representations: A Contextual Introduction
The Illustrated BERT, ELMo, and co.
Jurafsky and Martin Chapter 11 (Fine-Tuning and Masked Language Models)
Assignment 4 out
[code]
[handout]
[overleaf]
[colab run script]	Assignment 3 due
Thu May 2	Post-training (RLHF, SFT, DPO)
(by Archit Sharma)
[slides]	Suggested Readings:
Aligning language models to follow instructions
Scaling Instruction-Finetuned Language Models
AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback
How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources
Direct Preference Optimization: Your Language Model is Secretly a Reward Model
Fri May 3	Hugging Face Transformers Tutorial Session
[colab]	 3:30pm - 4:20pm
Gates B03		Project Proposal due
Week 6

Tue May 7	Benchmarking and Evaluation
(by Yann Dubois)
[slides]	Suggested Readings:
Challenges and Opportunities in NLP Benchmarking
Measuring Massive Multitask Language Understanding
Holistic Evaluation of Language Models
AlpacaEval
Thu May 9	Efficient Neural Network Training
(by Shikhar Murty)
[slides]	Suggested readings:
Mixed Precision Training
ZeRO: Memory Optimizations Toward Training Trillion Parameter Models
PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel
LoRA: Low-Rank Adaptation of Large Language Models
Final Project Proposals Returned

Project Milestone out
[handout]	Assignment 4 due
Week 7

Tue May 14	Speech Brain-Computer Interface
(by Chaofei Fan)
[slides]	Suggested readings:
A high-performance speech neuroprosthesis
An accurate and rapidly calibrating speech neuroprosthesis
A high-performance neuroprosthesis for speech decoding and avatar control
Brain-Machine Interfaces (Principles of Neural Science chapter)
Thu May 16	Reasoning and Agents
(by Shikhar Murty)
[slides]	Suggested readings:
Orca: Progressive Learning from Complex Explanation Traces of GPT-4
Least-to-Most Prompting Enables Complex Reasoning in Large Language Models
ReAct: Synergizing Reasoning and Acting in Language Models
BAGEL: Bootstrapping Agents by Guiding Exploration with Language
WebArena: A Realistic Web Environment for Building Autonomous Agents
Additional Readings:
Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks
Response: Emergent analogical reasoning in large language models
WebLINX: Real-World Website Navigation with Multi-Turn Dialogue
Week 8

Tue May 21	Life after DPO
(by Nathan Lambert)
[slides]
Suggested readings:
RewardBench: Evaluating Reward Models for Language Modeling
D2PO: Discriminator-Guided DPO with Response Evaluation Models
Social Choice for AI Alignment: Dealing with Diverse Human Feedback
Wed May 22				Final Project Milestone due
Thu May 23	ConvNets, Tree Recursive Neural Networks and Constituency Parsing
[slides]	Suggested readings (tentative):
Convolutional Neural Networks for Sentence Classification
Improving neural networks by preventing co-adaptation of feature detectors
A Convolutional Neural Network for Modelling Sentences
Parsing with Compositional Vector Grammars.
Constituency Parsing with a Self-Attentive Encoder
Final Project Report Instructions out
[Instructions]	
Fri May 24	
Course Withdrawal Deadline
Week 9

Tue May 28	An Introduction to Responsible NLP
(by Adina Williams)	Suggested readings:
Preface + Introduction chapter of the FairML book by Solon Barocas, Moritz Hardt, Arvind Narayanan
Introducing v0.5 of the AI Safety Benchmark from MLCommons
Final Project Milestones Returned	
Thu May 30	NLP, linguistics, and philosophy
[slides]	Suggested readings:		
Week 10

Tue June 4	Final Project Emergency Assistance (no lecture)	Extra project office hours available during usual lecture time, see Ed.		
Thu June 6	No class			Final project due
Mon June 10	Final Project Poster Session	 11 am - 3 pm [More details]
Location: McCaw Hall and Ford Gardens
On-campus students must attend in person!		[Printing guide]