classdef DefaultClassifier < mepclassifier.ClassifierContract
    %Implements a classifier
    
    properties(Access=protected)
        weights % Model weight
        mean % Model mean
        variance % Model variance
        inputShape=[50,1];
    end
    

    methods
        function this = DefaultClassifier()
            %DEFAULTCLASSIFIER Construct an instance of this class
            % [INPUTS]
            % identifier string: Unique identifier of the classifier model
            
            % load the model
            c=load('default_classifier.mat');
            this.weights=c.weights;
            this.mean=c.mean;
            this.variance=c.variance;
            this.sampleFreq=c.sample_freq;
            this.sanityTestInputs=c.sanity_test_inputs;
            this.sanityTestOutputs=c.sanity_test_outputs;
            this.name='Default MEP classifier';
        end
        
        function output = predict(this,dataset,sampleFreq)   
            % dataset: Dataset of shape= [batch x 50 x 1]
            % [OUTPUTS}
            % output: The prediction.
            
            if this.sampleFreq ~=sampleFreq
                error(' The correct sampling frequency is %g, but %g was provided %g',this.sampleFreq,sampleFreq);
            end
            
            %% Normalisation layer
            dataset=(dataset-this.mean)./sqrt(this.variance);
            
            %% Conv1D layer
            dataset=mepclassifier.Conv1D('same',1,'relu').setWeights(this.weights{1},this.weights{2}).call(dataset);
            
            %% LSTM layer
            output=mepclassifier.LSTM().setWeights(this.weights{3},this.weights{4},this.weights{5}).call(dataset);
            
            %% Dense layer
            output=mepclassifier.Dense().setWeights(this.weights{6},this.weights{7}).call(output);
        end
    end
    
end

