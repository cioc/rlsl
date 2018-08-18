# rlsl

Reinforcement Learning for Skip Lists

## Abstract

![uniform play](https://raw.githubusercontent.com/cioc/rlsl/master/content/uniform.gif)

The advent of deep reinforcement learning - with strong results in Atari, Go, and other Games -  enables us to reimagine how we build systems. Instead of carefully constructing of policies, systems programmers can focus on creating small, but correct, submodules and training agents to combine these submodules into higher program behavior. These systems outperform the classical programs based on fixed policies by continuously adapting to user access patterns in real time. To prove out this idea, we trained a deep q-learning model to construct skip list internal structures to minimize key lookup time. Our agent performs well in the presence of uniform as well as peaked access.           

## Skip Lists

![wikipedia](https://raw.githubusercontent.com/cioc/rlsl/master/content/skip_list.PNG)

First detailed by William Pugh in 1989, a skip list is a probabilistic data structure alternative to balanced trees that provides for fast lookups for key ranges.  A skip list is a collection of sorted linked lists, the base list with a hierarchy of descending linked lists. Pugh showed that given a uniform key access pattern, the expected lookup time for a element is logarithmic in the length of the skip list when node height is randomly chosen. The probabilistic nature of the skip list creates a tolerance to a broad collection of optimal solutions for internal structure making it a natural data structure for learning problems. The mechanism of key storage and lookup is simple - it can be implemented correctly in a few lines of code. We devised a clean interface between structure policy and implementation by decoupling the policy from the mechanism of setting heights.

## The Agent

![uniform performance](https://raw.githubusercontent.com/cioc/rlsl/master/content/max_score_uniform.PNG)

Our agent trains on state frames that include current skip list internal structure, access patterns heatmaps, access distributions, and key lookup distribution.  Our model uses 2d convolutions to process the internal structure and access pattern heatmaps, 1d convolutions to process the access and key lookup distributions, and fully connected layers to resolve the final output.  The reward signal is the reciprocal of the log of the average length of lookup path. On each turn, the agent is given the ability to choose one of six actions: set height to 0, 1, 2, or 3, move left or move right.  We trained a deep q-learning network with experience replay based on the implementation from the pytorch tutorial.  In order for our network to train, we introduced batch normalization and a learning rate schedule.  We used the same network and training system on both uniform and peak distributions.     

## Results

![peak play](https://raw.githubusercontent.com/cioc/rlsl/master/content/peaks.gif)

When play begins, the agent is given a skip list with an empty internal structure - every lookup must begin at the beginning of the list and traverse sequentially to find the corresponding entry.  The agent successfully learns to construct skip list structure, setting the internal heights of the elements, to rapidly drive up reward i.e. minimize lookup path length.  We trained the agent for ~100 games, each game running for 2000 turns.  Performance is generally poor for the first 30 (60,000 turns) games before play improves.  While the agent does learn to play well in the presence of peaked rather than uniform data access, further training and tuning would improve peaked play.    

![peak performance](https://raw.githubusercontent.com/cioc/rlsl/master/content/max_score_peaks.PNG)

## Future 

In this piece, we showed the ability of current day reinforcement learning algorithms to perform well in historical systems tasks such as data structure production.  This current implementation is slow - it is nowhere near the performance needed for production workloads.  However, the principles embodied here could be applied more broadly.  Databases could be constructed where the internal structure automatically adapts to user access pattern, enabling performant keyless databases.  Garbage collection could be based on program performance and automatically tuned.  Systems may self assemble e.g. automatically determining the number of caching layers and the type of backing storage.  Overall, the nature of the work in systems programming will change - the systems programmer will need to incorporate agents to learn policies from experience rather than construct solely closed solutions in order to remain on the cutting edge.  

# About the Authors

Charles Cary is the cofounder of Ruby Plants, a startup improving photosynthesis by bringing machine learning to plant science.  He is ex AWS, where he built out the AWS Demand Forecasting & DynamoDB teams, and formerly ran software at the Parker Institute for Cancer Immunotherapy.  Contact: cclego@gmail.com


Boya Fang develops backend cloud infrastructure for Airware, a leader in the drone technology space.  She attended HackBright Academy, where she developed a novel map-based visualization for disaster survival.  Before software engineering, she focused on biological sciences, developing next-gen sequencing workflows.  
