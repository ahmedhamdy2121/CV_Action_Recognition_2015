diary on
for i = 1:1 % 1:3 (datasets)
    tic
    for j = 1:1 % 1:5 (skeleton representation) 
        skeletal_action_classification(i,j)
    end
    toc
end
diary off