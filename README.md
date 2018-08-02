# ALICE deep learning bandits

Deep neural networks (DNNs) are commonly trained in batch, which requires knowledge of the entire training set a priori. In an era of Big Data and IoT, this paradigm is unsustainable as data sizes exceed memory capacities and many real-world problems involve data streams and online time-series data. 

While deep learning currently dominates the field of machine learning, the successful training of DNNs is dependent upon the sensitive tuning of hyperparameters, including number of layers, type of layer, activation function, and the learning rate schedule are just some of the decisions essential for optimising performance. We present Deep learning Bandit (DLB). DLB is a novel setting of the contextual multi-armed bandit problem for online time-series forecasting and online DNN hyperparameter optimisation. 

Future versions of DLB will perform anomaly detection on the forecasted online time-series signals, with the aim of providing a proof-of-concept for online anomaly prediciton of detector control systems on ALICE (A Large Ion Collider Experiment) at the Large Hadron Collider, CERN.
