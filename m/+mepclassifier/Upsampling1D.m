classdef Upsampling1D
    % Upsampling 1D
    
    properties(SetAccess=protected)
        rate=2 % int: The size of the upsampling
    end
    
    methods
        function this = Upsampling1D(rate)
            %Construct an instance of this class
            %  [INPUTS]
            %   rate int:optional
            %   [OUTPUTS]
            %   this: The instance of the class.

            if nargin >=1
                this.rate=rate;
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
            rate=this.rate;
            
            batch_size=size(data, 1);
            steps=size(data, 2);
            features=size(data,3);
            

            outputs = zeros(batch_size,steps*rate,features);
            for b = 1:batch_size
                block = data(b, :,:);
                rows=[];
                for r =1:steps
                    row=reshape(block(1,r,:),1,features);
                    rows=[rows; repmat(row, [rate,1])];
                end
                
                outputs(b,:,:)=rows;
                
            end
        end
    end
    
    methods(Static)
        function test_call()
            disp('TEST: Upsample1D call method')
            
            % Create data with shape (2,4,3)
            data=[1,2,3,4;5,6,7,8;9,10,11,12]'; 
            data(:,:,2)=data.^2; 
            data=permute(data,[3,1,2]);
                      
               
            %% 
            disp('_________________')
            disp('Test that it works when rate=2')
            rate=2;
            
            actual=mepclassifier.Upsampling1D(rate).call(data);
            
            expected=[1,1,2,2,3,3,4,4;5,5,6,6,7,7,8,8;9,9,10,10,11,11,12,12]';
            expected(:,:,2)=expected.^2; 
            expected=permute(expected,[3,1,2]);

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

