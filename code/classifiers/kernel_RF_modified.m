% Random Forest

function [total_accuracy, class_wise_accuracy, confusion_matrix] =...
    kernel_RF_modified(K_train_train, K_test_train,...
    training_labels, test_labels, C_val)    
    
    unique_classes = unique(training_labels);
    n_classes = length(unique_classes);
        
    class_wise_accuracy = zeros(n_classes, 1);    
    confusion_matrix = zeros(n_classes, n_classes); 
    
    for i = 1:n_classes          
        
        % extra
        %extra_options.classwt = [1 2];
        extra_options.nodesize = 2;
        extra_options.proximity = 1;
        extra_options.oob_prox = 0;
        
        rf_model = classRF_train(K_train_train{i}, training_labels, 100000, 0, extra_options);

        predicted_labels = classRF_predict(K_test_train{i}, rf_model);

        class_wise_accuracy(i) = length(find(predicted_labels == test_labels)) / length(test_labels);
                
    end
    
    total_accuracy = sum(class_wise_accuracy) / n_classes;
      
end
