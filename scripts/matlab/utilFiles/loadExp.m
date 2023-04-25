function [dataset, CNN, method, repetitions] = loadExp(dir, exp, CNN)

fileName = fullfile(dir,'outputFiles', exp, CNN, 'info.txt');
fileID = fopen(fileName,'r');
text = split(fscanf(fileID, '%s'),',');
fclose(fileID);

dataset = text{1};
CNN = text{2};
method = text{3};
repetitions = text{4};

end