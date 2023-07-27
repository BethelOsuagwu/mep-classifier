classdef MaxPooling1D
    % MaxPooling 1D
    
    properties(SetAccess=protected)
        pool_size=2 % int: The size of the pool mask
        strides=2 % int: the amount of strides
        padding='valid' %one of {'valid','same'}
    end
    
    methods
        function this = MaxPooling1D(pool_size,padding,strides)
            %Construct an instance of this class
            %  [INPUTS]
            %   pool_size int:optional
            %   padding string: {'valid','same'}. Optional.
            %   strides int:optional
            %
            %   [OUTPUTS]
            %   this: The instance of the class.

            % Check if strides < 1
            if strides < 1
                error('Strides cannot be less than 1');
            end

            % Check padding option
            if ~strcmp(padding, 'valid')
                error('The given padding option ,%s, is not implemented',padding);
            end

            if nargin >=1
                this.pool_size=pool_size;
            end
            if nargin >=2
                this.padding=padding;
            end
            if nargin >=3
                this.strides=strides;
            end

        end
        
        function outputs = call(this,inputs)
            % Forward pass.
            % 
            % [INPUTS]
            % inputs: The input data of shape [batch x steps x features]
            %
            % [OUTPUTS]
            % outputs: The output data of shape [batch x steps x features]

            
            data=inputs;
            pool_size=double(this.pool_size);%for some reason, this number come in as int32, so need to cast to double for any division to avoide automatic rounding.
            padding=this.padding;
            strides=double(this.strides);%for some reason, this number come in as int32, so need to cast to double for any division to avoide automatic rounding.
            
            

            

            input_steps = size(data, 2);
            output_steps = floor((input_steps - pool_size) / strides) + 1; % https://arxiv.org/pdf/1603.07285v1.pdf % But a wrong formula @see keras.layers.MaxPooling1D

            batch_size = size(data, 1);
            chans = size(data, 3);
            outputs = zeros(batch_size, output_steps, chans);

            % Vectorised
            outstep = 0;
            for instep = 1:strides:(input_steps - pool_size + 1)
                result = data(:, instep:(instep + pool_size - 1), :);
                outstep = outstep + 1; % OR: outstep = fix((instep - 1) / strides) + 1;

                result = max(result,[],2);
                outputs(:, outstep, :) = result;
            end

            
            %Non vectorised
            %{
            for b = 1:batch_size
                for chan = 1:chans
                    outstep = 0;
                    for instep = 1:strides:(input_steps - pool_size + 1)
                        result = data(b, instep:(instep + pool_size - 1), chan);
                        outstep = outstep + 1; % OR: outstep = fix((instep - 1) / strides) + 1;

                        result = max(result);
                        outputs(b, outstep, chan) = result;
                    end
                end
            end
            %}
        end

    end
    
    methods(Static)
        function test_call()
            disp('TEST: MaxPoolinf1D call method')
            
            % this should be shape=(5,3,1) but matlab does not like
            % trailing singleton dimention
            data=[[[0.5488135 ],[0.71518937],[0.60276338]],
                   [[0.54488318],[0.4236548 ],[0.64589411]],
                   [[0.43758721],[0.891773  ],[0.96366276]],        
                   [[0.38344152],[0.79172504],[0.52889492]],
                   [[0.56804456],[0.92559664],[0.07103606]]];%shape(5,3,1)
            
               
            %% 
            disp('_________________')
            disp('Test that it works when padding=valid,pool_size=2 and strides=1')
            padding='valid';
            pool_size = 2;
            strides = 1;
            
            actual=mepclassifier.MaxPooling1D(pool_size,padding,strides).call(data);
            expected=[[[0.71518934],[0.71518934]],
                     [[0.5448832 ],[0.6458941 ]],
                     [[0.891773  ],[0.96366274]],
                     [[0.79172504],[0.79172504]],
                     [[0.92559665],[0.92559665]]];% shape (5,2,1)

            if(all((size(actual)==size(expected))) && all(abs(actual(:)-expected(:)) <0.00001) )
                disp('Pass');
            else
                error('Fail\n Expected:%s;\n Actual:%s\n',mat2str(expected),mat2str(actual));
            end
                      
            %% 
            disp('_________________')
            disp('Test that it works when padding=valid,pool_size=2 and strides=2')
            padding='valid';
            pool_size = 2;
            strides = 2;
            actual=mepclassifier.MaxPooling1D(pool_size,padding,strides).call(data);
            expected=[[[0.71518934]],
                      [[0.5448832 ]],
                      [[0.891773  ]],
                      [[0.79172504]],
                      [[0.92559665]]];% shape=(5,1,1)
            
            if(all((size(actual)==size(expected))) && all(abs(actual(:)-expected(:)) <0.00001) )
                disp('Pass');
            else
                error('Fail\n Expected:%s;\n Actual:%s\n',mat2str(expected),mat2str(actual));
            end
            
            %%
            disp('_________________')
            disp('Test that it works when there are more than one input channel')
            data= cat(3, ...
                [0.5488135 , 0.71518937;
                 0.60276338, 0.54488318;
                 0.4236548 , 0.64589411], ...
                [0.43758721, 0.891773  ;
                 0.96366276, 0.38344152;
                 0.79172504, 0.52889492], ...
                [0.56804456, 0.92559664;
                 0.07103606, 0.0871293 ;
                 0.0202184 , 0.83261985], ...
                [0.77815675, 0.87001215;
                 0.97861834, 0.79915856;
                 0.46147936, 0.78052918], ...
                [0.11827443, 0.63992102;
                 0.14335329, 0.94466892;
                 0.52184832, 0.41466194]);
             data=permute(data,[3,1,2]);%shape(5,3,2)
             
            padding='valid';
            pool_size = 2;
            strides = 2;
            
            actual=mepclassifier.MaxPooling1D(pool_size,padding,strides).call(data);
            expected = cat(3, ...
                            [0.60276335, 0.71518934], ...
                            [0.96366274, 0.891773  ], ...
                            [0.56804454, 0.92559665], ...
                            [0.9786183 , 0.87001216], ...
                            [0.14335328, 0.9446689 ]);
            expected=permute(expected,[3,1,2]);% shape (5,1,2)

            if(all((size(actual)==size(expected))) && all(abs(actual(:)-expected(:)) <0.00001) )
                disp('Pass');
            else
                disp('Expected:');
                disp(expected)
                disp('Actual:');
                disp(actual);
                error('Fail\n ');
            end
        end
    end
end

