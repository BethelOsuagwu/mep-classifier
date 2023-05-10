classdef Dense
    % Dense
    
    properties
        kernels
        biases
        activation=@mepclassifier.Utils.sigmoid;
    end
    
    methods
        function this = Dense(kernels,biases,activation)
            %Dense Construct an instance of this class
            %   
            this.kernels=kernels;
            this.biases=biases;
            
            if nargin >=3
                this.activation=activation;
            end
        end
        
        function output = call(this,inputs)
            % Dense forward pass.
            
            output=inputs*this.kernels;
            output=output+this.biases;
            if not(isempty(this.activation))
                output=this.activation(output);
            end
        end
         
    end
    
end

