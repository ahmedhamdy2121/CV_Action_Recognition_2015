diary on
for i = 1:1 % 1:3 (datasets)
    for j = 1:1 % 1:5 (skeleton representation)
		tic	
        skeletal_action_classification(i,j)
		toc
    end
end
diary off