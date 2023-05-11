classdef LSTM
    % LSTM
    
    properties(SetAccess=protected)
        kernels
        recurrentKernels
        biases
        activation=@tanh;
        recurrentActivation=@mepclassifier.Utils.sigmoid;
        returnSequences=false; % When true only the last time step is returned.
    end
    
    methods
        function this = LSTM(activation,recurrentActivation,returnSequences)
            %LSTM Construct an instance of this class
            %   
            %   [INPUTS]
            %   activation: Activation function to use. Default is tanh.
            %   recurrentActivation: Activation function to use for the recurrent step. Default is sigmoid.
            %   returnSequences: When true, the output of the LSTM is a sequence. Default is false.
            %
            %   [OUTPUTS]
            %   this: The instance of the LSTM class.

            if nargin>0
                this.activation=activation;
            end
            if nargin>1
                this.recurrentActivation=recurrentActivation;
            end
            if nargin>2
                this.returnSequences=returnSequences;
            end

        end
        function this=setWeights(this,kernels,recurrentKernels,biases)
        % Set the weights of the LSTM layer.
        %
        % [INPUTS]
        % kernels: The kernel weights.
        % recurrentKernels: The recurrent kernel weights.
        % biases: The biases.
        %
        % [OUTPUTS]
        % this: The instance of the LSTM class.

            this.kernels=kernels;
            this.recurrentKernels=recurrentKernels;
            this.biases=biases;
        end
        function output = call(this,inputData)
        % LSTM forward pass.
        %
        % [INPUTS]
        % inputData: The input data of shape [batch x time x features]
        %
        % [OUTPUTS]
        % output: The output data of shape [batch x units] or [batch x steps x units] if returnSequences is true.

            % LSTM forward pass.
            dataset=inputData;
            bias=this.biases;
            steps=size(dataset,2);
            batch_size=size(dataset,1);
            
            kernel_size=size(this.kernels);
            num_filters=kernel_size(end);
            units=num_filters/4;
            prev_state=zeros(batch_size,units);
            prev_carry=zeros(batch_size,units);
            states={prev_state,prev_carry};
            output=zeros(batch_size,steps,units);
            for step=1:steps
                inputs=squeeze(dataset(:,step,:));
                
                % for the case when the batch size=1, squeeze will remove
                % the batch dimension; this must be corrected here. 
                if iscolumn(inputs)
                    inputs=reshape(inputs,1,[]);
                end
                
                %
                [output(:,step,:),states]=this.rnnCell(inputs,states{1},states{2},this.kernels,this.recurrentKernels,bias);
            end
            
            
            
            if ~this.returnSequences % Pass on only the last time step
                output=output(:,end,:);
                output=squeeze(output);
            end
        end
        
        
    end
    % LSTM utilties
    methods(Access=protected)
        function [h, state] = rnnCell(this,inputs, prev_state, prev_carry, kernel, recurrent_kernel, bias)
        % Perform matrix multiplication for a single RNN cell.
        %
        % Inputs:
        % - inputs: 2D tensor with shape (batch_size, input_dim)
        % - prev_state: 2D tensor with shape (batch_size, units). 
        %               The previous output of the RNN.
        % - prev_carry: 2D tensor with shape (batch_size, units)
        % - kernel: 2D tensor with shape (input_dim, units)
        % - recurrent_kernel: 2D tensor with shape (units, units)
        % - bias: 1D tensor with shape (units,)
        %
        % Outputs:
        % - h: Output - 2D tensor with shape (batch_size, units)
        % - state: List containing 2 2D tensors each with shape (batch_size, units). 
        %          The first tensor is the output and the second tensor is the carry.


            z = inputs*kernel;

            z = z + prev_state * recurrent_kernel;
            z = z + bias;

            z = mat2cell(z, size(z, 1), ones(1, 4)*size(z, 2)/4);
            [c, o] = this.computeOutputAndCarry(z, prev_carry);
            h = o .* this.activation(c);
            state = {h, c};

        end

        function [c, o] = computeOutputAndCarry(this,z, prev_carry)
            % Compute the output and carry of a single RNN cell.
            %
            % Inputs:
            % - z: List/tuple containing 4 2D tensors each with shape (batch_size, units).
            %      The 4 tensors are the result of splitting the output of the matrix
            %      multiplication of the inputs and the weights.
            % - prev_carry: 2D tensor with shape (batch_size, units)
            %
            % Outputs:
            % - c: 2D tensor with shape (batch_size, units). 
            %      The carry of the RNN cell.
            % - o: 2D tensor with shape (batch_size, units). 
            %      The output of the RNN cell prior to multiplying by activation(c)


            recurrent_activation=this.recurrentActivation;

            [z0, z1, z2, z3] = mepclassifier.Utils.cellDestructure(z);

            i = recurrent_activation(z0);
            f = recurrent_activation(z1);
            c = f .* prev_carry + i .* this.activation(z2);
            o = recurrent_activation(z3);

        end
    end
end

