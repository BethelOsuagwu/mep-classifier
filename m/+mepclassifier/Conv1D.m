classdef Conv1D
    % Conv 1D
    
    properties(SetAccess=protected)
        kernels
        biases
        activation=@mepclassifier.Utils.relu;
    end
    
    methods
        function this = Conv1D(activation)
            %CONV1D Construct an instance of this class
            %  [INPUTS]
            %   activation: Activation function to use. Default is relu. Set to [] to disable activation.
            %
            %   [OUTPUTS]
            %   this: The instance of the Conv1D class.


            if nargin >=1
                this.activation=activation;
            end
        end
        
        function this=setWeights(this,kernels,biases)
            % Set the weights of the Conv1D layer.
            %
            % [INPUTS]
            % kernels: The kernel weights.
            % biases: The biases.
            %
            % [OUTPUTS]
            % this: The instance of the Conv1D class.

            this.kernels=kernels;
            this.biases=biases;
        end
        
        function output = call(this,inputs)
            % Conv1D forward pass.
            % 
            % [INPUTS]
            % inputs: The input data of shape [batch x steps x features]
            %
            % [OUTPUTS]
            % output: The output data of shape [batch x steps x features]

            dataset=inputs;
            kernel_size=size(this.kernels);
            num_filters=kernel_size(end);
            batch_size=size(dataset,1);
            steps=size(dataset,2);
            
            data_filtered=zeros(batch_size,steps,num_filters);
            for batch_n=1:batch_size
                for i= 1:num_filters
                    kern=this.kernels(:,1,i);
                    data_filtered(batch_n,:,i)=mepclassifier.Utils.crossCorrelation1DSame(dataset(batch_n,:,1),kern)+this.biases(i);
                end
            end
            output=data_filtered;
            if ~isempty(this.activation)
                output=this.activation(output);
            end
        end
    end
end

