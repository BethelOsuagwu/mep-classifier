classdef UClassifier < mepclassifier.ClassifierContract
    %Implements a classifier
    
    properties(Access=protected)

        
        inputShape=[50,1];
    end
    

    methods
        function this = UClassifier()
            %Construct an instance of this class
            % [INPUTS]
            % identifier string: Unique identifier of the classifier model
            
            % load the model
            %addpath './uclassifier'
            %c=load('./uclassifier/uclassifier.mat');
            c=load('uclassifier/uclassifier.mat');
            this.layers=c.layers;
            this.sampleFreq=c.sample_freq;
            this.sanityTestInputs=c.sanity_test_inputs;
            this.sanityTestOutputs=c.sanity_test_outputs;
            this.name='U MEP classifier';
        end
        
        function output = predict(this,dataset,sampleFreq)   
            % dataset: Dataset of shape= [batch x time x features]
            % [OUTPUTS}
            % output: The prediction.
            
            if this.sampleFreq ~=sampleFreq
                error(' The correct sampling frequency is %g, but %g was provided %g',this.sampleFreq,sampleFreq);
            end
            
            output=this.predictionLoop(dataset);
        end
    end
    
end

