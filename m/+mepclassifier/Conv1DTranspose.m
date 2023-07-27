classdef Conv1DTranspose
    % Conv 1D transpose
    
    properties(SetAccess=protected)
        kernels
        biases
        padding='valid';
        strides=1;
        
        activation=@mepclassifier.Utils.relu;
    end
    
    methods
        function this = Conv1DTranspose(padding, strides,activation)
            %CONV1DTranspose Construct an instance of this class
            %  [INPUTS]
            %   padding string: {valid,same}
            %   strides int: Strides 
            %   activation string|function: Activation function to use. Default is relu. Set to [] to disable activation.
            %   
            %   [OUTPUTS]
            %   this: The instance of the Conv1DTranspose class.

            if nargin<1
                this.padding='valid';
            end
            if nargin<2
                this.strides=1;
            end
            if nargin >=3 % set only if given
                if strcmp(activation,'relu')
                    activation=@mepclassifier.Utils.relu;
                end
                this.activation=activation;
            end
            
        
            if strides ~= 1
                error('Non unitary stride is not supported');
            end
            
            this.padding=padding;
            this.strides=strides;
        end
        
        function this=setWeights(this,kernels,biases)
            % Set the weights of the Conv1DTranspose layer.
            %
            % [INPUTS]
            % kernels: The kernel weights. Shape=(filter_len,1,num_filters).
            % biases: The biases. Shape=(filter_len,).
            %
            % [OUTPUTS]
            % this: The instance of the Conv1DTranspose class.

            this.kernels=kernels;
            this.biases=biases;
        end
        
        function output = call(this,inputs)
            % Conv1DTranspose forward pass.
            % 
            % [INPUTS]
            % inputs: The input data of shape [batch x steps x features]
            %
            % [OUTPUTS]
            % output: The output data of shape [batch x steps x features]
            
            
            padding=this.padding;
            
            
            dataset=inputs;
            kernel_size=size(this.kernels);
            num_filters=kernel_size(2);
            batch_size=size(dataset,1);
            features=size(dataset,3);
            
            data_filtered = [];
        
            for batch_n = 1:batch_size
                for filter_n = 1:num_filters
                   for feature=1:features
                        kern=this.kernels(:,filter_n,feature);
                        result = mepclassifier.Utils.conv1d_transpose(dataset(batch_n,:,feature), kern, padding);
                        if isempty(data_filtered)
                            data_filtered = zeros(batch_size, numel(result), num_filters);
                        end
                        data_filtered(batch_n, :, filter_n) = data_filtered(batch_n, :, filter_n) + result;
                   end
                    data_filtered(batch_n, :, filter_n) = data_filtered(batch_n, :, filter_n) + this.biases(filter_n);
                end
            end
            output=data_filtered;
            if ~isempty(this.activation)
                output=this.activation(output);
            end
            
        end
    end
    
    methods(Static)
        function test_call()
            disp('TEST: Conv1DTranspose call method')
            
            disp('_________________')
            disp('Test that it works')
            % The shape of the data in python is (1, 4, 1) i.e (batch,steps,chan)
            data=[[[0.5488135 ],[0.71518937],[0.60276338],[0.54488318]]];
            filters=[[[0.4236548 ]],
                       [[0.64589411]],
                       [[0.43758721]]];
            biases=0;
            
            actual=mepclassifier.Conv1DTranspose('valid',1).setWeights(filters,biases).call(data);
            expected=[[[0.23250748]
                      [0.6574688 ]
                      [0.95745397]
                      [0.9331214 ]
                      [0.6156984 ]
                      [0.23843391]]]';%this should shape=(1,6,1)
                  
            if(all((size(actual)==size(expected))) && all(abs(actual(:)-expected(:)) <0.00001) )
                disp('Pass');
            else
                error('Fail\n Expected:%s;\n Actual:%s\n',mat2str(expected),mat2str(actual));
            end
        end
    end
end

