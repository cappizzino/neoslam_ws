data_int = select(bag,'Topic','/int_scores');
msg_int = readMessages(data_int,'DataFormat','struct');

matrix(1,1) = msg_int{1}.Data;

for i=2:length(msg_int)
    if length(msg_int{i}.Data)<length(msg_int{i+1}.Data)
        matrix(end+1,end+1) = msg_int{i}.Data;
    end
end
