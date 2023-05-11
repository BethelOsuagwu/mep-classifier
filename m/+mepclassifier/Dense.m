classdef Dense
    % Dense
    
    properties(SetAccess=protected)
        kernels % Kernel weight
        biases % Biases weight
        activation=@mepclassifier.Utils.sigmoid;
    end
    
    methods
        function this = Dense(activation)
            %Dense Construct an instance of this class
            %   [INPUTS]
            %   activation: Activation function to use. Default is sigmoid. Set to [] to disable activation.
            %
            %   [OUTPUTS]
            %   this: The instance of the Dense class.
            
            if nargin >=1
                this.activation=activation;
            end
        end
        
        function this=setWeights(this,kernels,biases)
            % Set the weights of the Dense layer.
            %
            % [INPUTS]
            % kernels: The kernel weights.
            % biases: The biases.
            %
            % [OUTPUTS]
            % this: The instance of the Dense class.

            this.kernels=kernels;
            this.biases=biases;
        end
        
        function output = call(this,inputs)
            % Dense forward pass.
            %
            % [INPUTS]
            % inputs: The input data of shape [batch x features]
            %
            % [OUTPUTS]
            % output: The output data of shape [batch x units]

            
            output=inputs*this.kernels;
            output=output+this.biases;
            if not(isempty(this.activation))
                output=this.activation(output);
            end
        end
         
    end
    
end

