1. Visualise the output of the Conv1D layer to see if it is low pass/ higpass filtering
2. If in #1 there is no sufficient filtering, then we should implement a low and highpass filtering of the input data.
    2.1. How do we implement the filtering to ensure it is deployable.
        2.1.1. Online filtering during data collection-> But this enforces how the data must be collected
        2.1.2. Encoporate freqency filtering in the training and inference pipelines:
                While this is easy during training, it is not straightforward during
                inference since the signal frame length can be too small small if inference is 
                done frame by frame. A potential solution is to pad (both training 
                and ??) inference data with 1second zeros to allow filtering. The 
                padded data can then be zerophase filtered.
                
2. Mix the training data b4 cutting out the evaluation set

# Possible benefits of using a classifier for EP detection
1. Possibility to automatically detect and group different EP mophology/shapes. 
    The can also be used to detect abnormal response morphology. 
2. Artefacts can automatically be rejected. For example stimulation artefacts can be rejected. Our study will focus on this.

# Other possible uses of classifier for EP detection
1. Detection of other potentials such as readiness motor potential, SSEP responses, P300 etc.(CHECK: Isn't there a classifier in the literature for detecting potentials such as readiness potential and p300 already? e.g: in paired associative stimulation with readiness potential.)