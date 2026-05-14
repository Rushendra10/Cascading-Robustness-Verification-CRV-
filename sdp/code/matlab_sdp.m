function [val] = matlab_sdp(matlab_folder)

cd(matlab_folder)
addpath('/home/ubuntu/mosek/mosek/10.2/toolbox/r2017aom');
addpath('YALMIP-master');
addpath('YALMIP-master/extras');
addpath('YALMIP-master/solvers');

S = load('sizes.mat'); 
sizes = double(S.sizes);
B = load('biases.mat'); 
biases = {};
biases{1} = (squeeze(B.biases)).'; 
W = load('weights.mat'); 
weights = {};
weights{1} = squeeze(W.weights);  
O = load('opt_params.mat'); 
opt_params = O.opt_params;
lower = [];
lower = [squeeze(opt_params.lower)];  
upper = [];
upper = [squeeze(opt_params.upper)]; 
final_constant = double(opt_params.final_constant); 
final_linear = double(opt_params.final_linear);

size_big_matrix = 1 + sum(sizes);

M = sdpvar(size_big_matrix, size_big_matrix); 
constraints = [M>=0, M(1, 1) == 1]; 
x = M(1, 2: 1+sizes(1)).';
X = M(2: 1 + sizes(1), 2: 1 + sizes(1)); 

%Input constraints 
constraints = [constraints, x>=double(cell2mat(lower(1)))]; 
constraints = [constraints, x<=double(cell2mat(upper(1)))]; 
constraints = [constraints, (diag(X) - (double(cell2mat(lower(1))) + double(cell2mat(upper(1)))).*x + double(cell2mat(lower(1))).*double(cell2mat(upper(1))) <= 1E-5)]; 
current_pos_matrix = 1;
W_1 = double(cell2mat(weights(1)));
b_1 = double(cell2mat(biases(1)));
input_span = 1 + current_pos_matrix: current_pos_matrix + sizes(1);
output_span = 1 + current_pos_matrix + sizes(1): current_pos_matrix + sizes(1) + sizes(2);
input_linear = M(1, input_span).'; 
output_linear = M(1, output_span).'; 
output_quadratic = M(output_span, output_span); 
cross_terms = M(input_span, output_span); 

% ReLU linear constraints 
constraints = [constraints, output_linear >= W_1*input_linear + b_1]; 
constraints = [constraints, output_linear >=0]; 
% ReLU quadratic constraints 1
temp_matrix = W_1*cross_terms; 
constraints = [constraints, diag(output_quadratic) == diag(temp_matrix) + output_linear.*b_1]; 
% layerwise constraints 
constraints = [constraints, (diag(output_quadratic) - (double(cell2mat(lower(2))) + double(cell2mat(upper(2)))).*output_linear + double(cell2mat(lower(2))).*double(cell2mat(upper(2))) <= 1E-5)]; 

current_pos_matrix = current_pos_matrix + sizes(1);
% New constraint 1 
constraints = [constraints, diag(output_quadratic) - diag(temp_matrix) - b_1.*output_linear - double(cell2mat(lower(2))).*output_linear + (W_1*input_linear).*double(cell2mat(lower(2))) + double(cell2mat(lower(2))).*b_1<=1E-5]; 
% Constructing the objective 
s = size(final_linear);
size(final_constant);
dim_final = s(2);
y_final = M(1, 1 + current_pos_matrix: current_pos_matrix + dim_final).'; 
obj = final_linear*y_final + final_constant;
diagnostics = optimize(constraints, -obj, sdpsettings('dualize', 1, 'solver', 'mosek'))
val = value(obj);
time = diagnostics.solvertime

save(char(string('SDP_optimum.mat')), 'val')

file_name = fullfile('code' ,'logs', 'output.txt');
fileID = fopen(file_name, 'a');
disp("fileID is:")
disp(fileID);
fprintf(fileID, '%g\n', val);
fprintf(fileID, '%g\n', time);

fclose(fileID);

end
