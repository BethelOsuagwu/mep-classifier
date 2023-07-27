classdef Conv1D
    % Conv 1D
    
    properties(SetAccess=protected)
        kernels
        biases
        padding='valid';
        strides=1;
        activation=[];%@mepclassifier.Utils.relu;
    end
    
    methods
        function this = Conv1D(padding, strides,activation)
            %CONV1D Construct an instance of this class
            %  [INPUTS]
            %   padding string: {valid,same}
            %   strides int: Strides 
            %   activation string|function: Activation function to use. Default is relu. Set to [] to disable activation.
            %
            %   [OUTPUTS]
            %   this: The instance of the Conv1D class.


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
                warning('Non unitary stride is not tested');
            end
            
            this.padding=padding;
            this.strides=strides;
        end
        
        function this=setWeights(this,kernels,biases)
            % Set the weights of the Conv1D layer.
            %
            % [INPUTS]
            % kernels: The kernel weights. Shape=(filter_len,1,num_filters).
            % biases: The biases. Shape=(filter_len,).
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

            features=size(dataset,3);
            
            %data_filtered=zeros(batch_size,steps,num_filters);
            data_filtered=[];
            for batch_n=1:batch_size
                %for feature=1:features
                    for kernel_n= 1:num_filters
                        for feature=1:features
                            kern=this.kernels(:,feature,kernel_n);
                            result=mepclassifier.Utils.convolve1d(dataset(batch_n,:,feature),kern,this.padding,this.strides);
                            if isempty(data_filtered)
                                data_filtered = zeros(batch_size, numel(result), num_filters);
                            end
                            data_filtered(batch_n,:,kernel_n) = data_filtered(batch_n,:,kernel_n) + result;
                            %method in Utils
                            %data_filtered(batch_n,:,kernel_n) = data_filtered(batch_n,:,kernel_n) ...
                            %                                  + mepclassifier.Utils.convolve1d(dataset(batch_n,:,feature),kern,this.padding,this.strides);

                        end
                        data_filtered(batch_n,:,kernel_n)=data_filtered(batch_n,:,kernel_n)+this.biases(kernel_n);
                    end
                %end
            end
            output=data_filtered;
            if ~isempty(this.activation)
                output=this.activation(output);
            end
        end
    end
end

