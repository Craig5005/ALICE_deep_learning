![Figure_0](AliceBandit_Logo.png)
# Adversarial thresholding semi-bandits for online monitoring of ALICE detector controls

The classical multi-armed bandit is one of the most common examples of sequential decision-making, either by trading-off between exploiting and exploring arms to maximise some payoff or purely exploring arms until the optimal arm is identified. In particular, a bandit player wanting to only pull arms with stochastic feedback exceeding a given threshold, has been studied extensively in a pure exploration context. However, numerous applications fail to be expressed, where a player wishes to balance the need to observe regions of an uncertain environment that are currently interesting (exploit) and checking if neglected regions have become interesting since last observed (explore).

We introduce the adversarial thresholding semi-bandit problem: a non-stochastic bandit model, where a player wants to only pull (potentially multiple) arms with feedback meeting some threshold condition. Our main objective is to design algorithms that meet the requirements of the adversarial thresholding semi-bandit problem theoretically, empirically and computationally, for a given application. In other words, we want to develop a machine that learns to pull arms according to some threshold condition and adapts quickly if the feedback from pulling an arm unexpectedly changes. This work has many real-world applications and is motivated by online detector control monitoring in high-energy physics experiments, on the Large Hadron Collider.

One of the approaches proposed in our work decomposes the action of pulling an arm. The player decides which arms to select for observation (pulling) and then decides which selected arms to observe. This decomposition enables the adversarial thresholding semi-bandit player to apply multiple policies for observing feedback from arms. Significant reduction in computational complexity are now achievable compared with exponentially-weighted adversarial thresholding semi-bandits.

![Figure_1](AliceBandit_diagram_1.png)
Due to the extreme conditions experienced in heavy-ion particle colliders, the power supply to any device exceeding safe operating parameters is automatically shut down (tripped) to preserve integrity and functionality of the device. Prior to recent upgrades, a device (channel) trip would halt data-taking for the entire experiment. Post-trip recovery requires a costly procedure both in terms of expertise and data-taking time. After the completion of the current upgrading phyase (scheduled for 2021), the detector will collect data continuously. In this new regime, a channel trip will result in only the affected components of the experiment being shut down. However, since the new upgraded experiment will enable data-taking to increase by a factor of 100, each trip will have a significant impact on the experiments ability to provide physicists withe reliable data to analyse. We demonstrate that adversarial thresholding semi-bandits efficiently identify device channels either exceeding a fixed threshold or deviating by more than a prescribed range priot to a trip, extending the state-of-the-art in high-energy physics detector control.

# Implementation

The ATSBP folder contains two notebooks, ATSBP.ipynb and ATSBP_plots.ipynb. The former provides the code required for implementing adversarial thresholding semi-bandit algorithms in both fixed threshold and threshold interval settings. The latter contains the code for generating the necessary plots in our thesis, "adversarial thresholding semi-bandit problems".

The actual datasets are not provided in this Github repo, but are available on request.
