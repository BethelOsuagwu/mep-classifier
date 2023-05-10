classdef (Abstract) ClassifierContract < handle
    %CLASSIFIERCONTRACT is the interface for classifier
    
    properties(Access=protected,Abstract)
        inputShape; % 1D vector of input dimension without the batch size.
    end

    properties(Access=protected)
        name='untitled'; % string: Set the name of the classifier
        weights % Model weight
        sampleFreq % double: Sample frequency(Hz) of model. This is the sample frequency of the training data. 
        sanityTestInputs % Model sanity test input
        sanityTestOutputs % Model sanity inputs output
    end
    
    
    methods(Abstract)
        preds=predict(this,dataset,sampleFreq)
            % Make predictions for the given dataset.
            % [INPUTS]
            % dataset array: Dataset of shape= [batches x steps x features]
            % sampleFreq double: Sampling frequency in Hz, of the given
            %   inference data. 
            % [OUTPUTS]
            % preds array: Predictions. Shape=[batches,output_dim]
    end
    
    methods
        function set.name(this,name)
            this.name=name;
        end
    end
    
    
    methods
        function this = ClassifierContract()
            %CLASSIFIERINTERFACE Construct an instance of this class  
        end
        
        function result= sanityTest(this)
            % Run a sanity test for the classifier.
            % [OUTPUTS]
            % result logical: True if the test passes
            tol=10^-2; % Error tolerance
            
            result=false;
            expected =this.sanityTestOutputs;
            actual=this.predict(this.sanityTestInputs,this.sampleFreq);
            if all(abs(actual-expected)<tol)
                result=true;
            end
            
            if nargout <1
                if result
                    fprintf('_/: %s passed sanity test \n',this.name);
                else
                    error('x: %s failed sanity test: \n expected output=%s, \n but got= %s',this.name,mat2str(expected),mat2str(actual));
                end
                clear result
                return;
            end
            
        end
        
        function [start,stop,preds]=classify(this,data,dataSampleFreq)
            % Classify every point in the data.
            % [INPUTS]
            % data array<double>: Samples of shape=[num_samples,features]
            % dataSampleFreq double: The sample frequency in Hz, of the
            %   given data, i.e. inference data. 
            % [OUTPUTS]
            % start int: Start sample of MEP in data
            % stop int: Stop sample of MEP in data
            % preds array<double>: The probability of each sample in data
            %   being MEP.

            dataset=this.sequence(data);
            preds=this.predict(dataset,dataSampleFreq);
            [start,stop]=this.trace(preds);
        end
        
    end
    
    methods(Access=protected)
        function dataset=sequence(this,data)
            % Generate sequences for the given data
            % [INPUTS]
            % data array<double>: Samples of shape=[num_samples,features]
            % [OUTPUTS]
            % dataset array<double>: Input sequence for classifier.
            %   Shape=[batch_size=num_samples,sequence_length,features]

            feature_dim=this.inputShape(2); % Feature dimension

            % Throw error if feature dimension of data is not equal to the
            % feature dimension of the model
            if size(data,2)~=feature_dim
                error('x: Feature dimension of data is not equal to the feature dimension of the model');
            end
            
            num_samples=size(data,1); % Number of samples

            L=this.inputShape(1); % Sequence length
            
            % Pad data with the last sample to make it divisible by L
            data_padded=[data;repmat(data(end,:),L-mod(num_samples,L),1)];

            % Again pad data with the last sample to increase it by L-1.
            % This is done to make sure that the first sample of the data
            % is the first sample of the first sequence.
            data_padded=[data_padded;repmat(data_padded(end,:),L-1,1)];

            % Initialize dataset
            dataset=zeros(num_samples,L,feature_dim);

            % Generate sequences
            for i=1:num_samples
                dataset(i,:,:)=data_padded(i:i+L-1,:);
            end

            % Assert that the dataset is of the correct shape
            if feature_dim==1
                % MATLAB does not allow trailing singleton dimensions
                assert(all(size(dataset)==[num_samples,L]),'x: Dataset is not of the correct shape');
            else
                assert(all(size(dataset)==[num_samples,L,feature_dim]),'x: Dataset is not of the correct shape');
            end

        end
        
        function [start,stop]=trace(this,preds)
            % Trace traces prediction probabilities to determin MEP onset
            % and offset.
            % [INPUTS]
            % preds array<double>: The probability of MEP being MEP.
            %   It uis assumed that preds has shape=[num_samples,units=3]
            %   where units=3 is the number of output units of the model.
            %   The first unit is the probability of the sample being
            %   background, the second unit is the probability of the 
            %   sample being stimulation artifact and the third unit is the 
            %   probability of the sample being MEP. This method should be
            %   overriden by the concrete class if these do not apply.
            % [OUTPUTS]
            % start double|NaN: Start sample of MEP in preds.
            % stop double|NaN: Stop sample of MEP in preds.
                     

            % Find row predicted to be MEP
            threshold=0.5;
            mep_preds=preds(:,3)>threshold;

            % Find the streaks of consecutive MEP predictions.
            streaks=this.findStreaks(mep_preds);

            % Find the longest streak
            [~,idx]=max(streaks(:,2)-streaks(:,1));

            % Get the start and stop indices of the longest streak
            start=streaks(idx,1);
            stop=streaks(idx,2);

            % If the start and stop indices are the same, then there is no
            % MEP
            if start==stop
                start=NaN;
                stop=NaN;
            end
            
        end
    end
    
    methods(Access=protected,Static)
        function streaks = findStreaks(vec, tolerance)
            % find indices of all streaks of consecutive 1s with a
            % tolerance for broken streaks 
            % [INPUTS]
            % vec array<logical|double>: column vector containing 0 or 1
            % tolerance int: maximum number of broken streaks allowed
            %[OUTPUTS]
            % streaks array<double>: Shape=[len(vec),2], each row represents a
            %   streak and the two columns contain the start and end
            %   indices of the streak, respectively.  
            if nargin < 2
                tolerance=1;
            end

            % initialize variables
            streaks = zeros(0, 2);
            i = 1;
            n = length(vec);



            % search for streaks
            while i <= n
                % find start of streak
                if vec(i) == 1
                    start = i;
                    % find end of streak
                    while i < n && hasMore1s(vec, i, tolerance)
                        i = i + 1;
                    end
                    streaks(end+1, :) = [start i];
                end
                i = i + 1;
            end

            function more = hasMore1s(vec, idx, tolerance)
                % check if there are more 1s in vec after index idx.
                % [INPUTS]
                % vec array: 1D array within which to search for re 1s.
                % idx int: The index of vec above which more 1s is
                %   searched.
                % tolerance int: How much more indexs above idx is
                %   searched. 
                % [OUTPUTS]
                % more logical: True if a one is found.
                more = false;
                for j = 1:tolerance
                    k = idx + j;
                    if k <= length(vec) && vec(k) == 1
                        more = true;
                        break;
                    end
                end
            end
        end

    end
        
    

    
end

