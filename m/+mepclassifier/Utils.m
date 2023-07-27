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
            % Perform a 1D cross correlation(which is convolution in deep 
            % learnng), with padding=same.
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
            
            
            if nargin<3
                direction='both';
            end
            
            % Early return
            if pad_size==0
                padded=x;
                return
            end
            
            
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
        
        function out_data = conv1d_transpose(data, kernel, padding, strides)
            % Function to imitate basic features of Keras conv1d_transpose operation in MATLAB
            % 
            % Parameters
            % ----------
            % data : ndarray
            %     Data. Shape=(N,)
            % kernel : array
            %     Filter. Shape=(k,) where k is the filter size.
            % padding : string, optional
            %     Padding {valid,same}. The default is "valid".
            % strides : int, optional
            %     Strides. The default is 1.
            % 
            % Raises
            % ------
            % ValueError
            %     If the stride is not allowed.
            % 
            % Returns
            % -------
            % ndarray
            %     The transpose convolution result for each batch and channel.
            if nargin<3
                padding="valid";
            end
            if nargin < 4
                strides = 1;
            end
        
            if strides ~= 1
                error('Non unitary stride is not supported');
            end
        
            s = strides;
            o = length(data);
            k = length(kernel);
            p = 0;
        
            if strcmp(padding, 'same')
                p = floor((k - 1) / 2);
            elseif strcmp(padding, 'valid')
                p = 0;
            end
        
            i = (o - 1) * s - 2 * p + k;
            input_size = i;
            output_size = i;
            pad_size = ((output_size - 1) * s - input_size + k) / 2;
            pad_size = int32(ceil(pad_size));
        
            kernel = rot90(kernel,2);  % Flip the filter
            data_padded = mepclassifier.Utils.padConstant1D(data, pad_size);  % Pad the input array
            out_data = mepclassifier.Utils.convolve1d(data_padded, kernel, 'same');  % Perform the convolution
        
            if mod(k, 2) == 0
                out_data = out_data(1:end-1);  % Adjust output size if the kernel size is even
            end
        end
        
        function test_conv1d_transpose()
            disp('TEST: conv1d_transpose')
            
            disp('_________________')
            disp('Test that it works when filter length is even and padding=same')
            actual=mepclassifier.Utils.conv1d_transpose([-2,1,4],[1,2],'same',1);
            expected=[-2,-3,6,8];
            if(length(actual)==length(expected) && all(actual(:)==expected(:)))
                disp('Pass');
            else
                error('Fail\n Expected:%g;\n Actual:%g',expected,actual);
            end
            
            disp('_________________')
            disp('Test that it works when filter length is odd and padding=same')
            actual=mepclassifier.Utils.conv1d_transpose([-2,1,4],[1,2,1],'same',1);
            expected=[-2,-3,4,9,4];
            if(length(actual)==length(expected) && all(actual(:)==expected(:)))
                disp('Pass');
            else
                error('Fail\n Expected:%g;\n Actual:%g',expected,actual);
            end
            
            disp('Test that it works when padding is "valid"')
            actual=mepclassifier.Utils.conv1d_transpose([-2,1,4],[1,2,1],'valid',1);
            expected=[-2,-3,4,9,4];
            if(length(actual)==length(expected) && all(actual(:)==expected(:)))
                disp('Pass');
            else
                error('Fail\n Expected:%g;\n Actual:%g',expected,actual);
            end
        end
        
        function out = convolve1d(x, kernel, padding, strides)
            % Function to perform elementary convolution as it is defined in deep learning
            % 
            % Parameters
            % ----------
            % x : array
            %     Data. shape= (N,)
            % kernel : array
            %     1D filter
            % padding : int, optional
            %     The padding scheem {valid,same}. The default is "valid".
            % strides : int, optional
            %     The stride. Onli stride=1 is currently supported. The default is 1.
            % 
            % Raises
            % ------
            % ValueError
            %     If stride is not 1.
            % 
            % Returns
            % -------
            % out : array
            %     1D array. Shape determined by convolution procedure. 
            
            if nargin<3
                padding="valid";
            end
            if nargin < 4
                strides = 1;
            end
        
            if strides ~= 1
                error('Non unitary stride is not supported');
            end
            
            % Force into row vectors
            x=reshape(x,1,[]);
            kernel=reshape(kernel,1,[]);
        
            %
            k_size = length(kernel);
        
            s = strides;
            i = length(x);
            k = length(kernel);
            p = 0;
        
            if strcmp(padding, 'same')
                p = floor((k - 1) / 2);
                o = i;
            elseif strcmp(padding, 'valid')
                p = 0;
                o = floor((i + 2 * p - k) / s) + 1;
            end
        
            x_padded = mepclassifier.Utils.padConstant1D(x, p);  % Pad the input array
            
            %For even filter len , we need to add extra 'post' padding when
            %padding is 'same'
            if strcmp(padding, 'same')
                if mod(length(kernel),2)==0
                    x_padded = mepclassifier.Utils.padConstant1D(x_padded, 1,'post');
                end
            end
        
            out = zeros(1, o);  % Initialize the output array
        
            for idx = 1:length(out)
                d = x_padded(idx:idx + k_size - 1);  % Extract a segment from the pad input
        
%                 if strcmp(padding, 'same')
%                     if length(d) < length(kernel)
%                         % This is a little messy but needed to produce
%                         % tensorflow-like result.
%                         d = mepclassifier.Utils.padConstant1D(d, [0, length(kernel) - length(d)], 'post');  % Zero pad if needed for 'same' padding
%                     end
%                 end
        
                out(idx) = sum(d .* kernel);  % Perform the convolution
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

