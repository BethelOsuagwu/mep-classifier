classdef Utils
    %Utilities.
    
    methods
        function obj = Utils()
        end
    end
    methods(Access=public,Static)
        function y = sigmoid(x)
        % Sigmoid function.
            y = 1 ./ (1 + exp(-x));
        end
        
        function y=relu(x)
            % Relu function
            y=max(x,zeros(size(x)));
        end
        
        function out = crossCorrelation1DSame(x, kernel)
            % Perform a 1D cross correlation, with padding=same.
            % [INPUTS]
            % x array: shape = [samples,]
            % kernel array: shape = [kernel_size,] 
            % [OUTPUTS]
            % out array: Same shape as input x.
            
            x=reshape(x,1,[]);
            kernel=reshape(kernel,1,[]);
            
            % compute padding size
            k_size = length(kernel);
            pad_size = floor((k_size - 1) / 2); % pad size that produces similar results as in TensorFlow

            % pad the input array
            x_padded = mepclassifier.Utils.padConstant1D(x, pad_size, 'both');

            % initialize output array
            out = zeros(size(x));

            % perform convolution
            for i = 1:numel(out)
                sz=i+k_size-1;
                if sz>length(x_padded)
                    sz=length(x_padded);
                end
                d = x_padded(i:sz);
                if length(d) < k_size
                    % This is a little messy but needed to produce
                    % tensorflow-like result.
                    d = mepclassifier.Utils.padConstant1D(d, k_size-length(d), 'post');
                end
                out(i) = d * kernel';
            end
        end
        
        function padded=padConstant1D(x,pad_size,direction)
            % Pads a vector with contant of the given size in the specified
            % drection.
            % [INPUTS]
            % x array: vector to pad.
            % pad_size int: Padding size
            % direction string: Padding direction, specified as one of the following values:
            %   both:pads start and end; post:only end; pre: only start.
            % [OUTPUTS]
            % paddad array: The padded array
            
            is_row=isrow(x);
            
            % Force x as row vector
            x=reshape(x,1,[]); 
            
            start_size=[];
            rear_size=[];
            switch(direction)
                case 'both'
                    start_size=pad_size;
                    rear_size=pad_size;
                case 'post'
                    rear_size=pad_size;
                case 'pre'
                    start_size=pad_size;
                otherwise
                    error('Unknown pad_size, %s',pad_size);
            end
            
            start=zeros(1,start_size);
            rear=zeros(1,rear_size);
            
            padded=[start,x,rear];
                    
            if ~is_row       
                padded=padded';
            end
        end
        
        function varargout= cellDestructure(c)
            % Destructure an cell array
            % [INPUTS]
            % c cell: Cell array.
            % [OUTPUTS]
            % varargout cell: c
            varargout=c;
        end
    end
end

