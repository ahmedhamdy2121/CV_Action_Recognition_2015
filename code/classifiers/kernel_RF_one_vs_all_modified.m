% Random Forest

function [total_accuracy, class_wise_accuracy, confusion_matrix] =...
    kernel_RF_one_vs_all_modified(K_train_train, K_test_train,...
    training_labels, test_labels, C_val)    
    
    unique_classes = unique(training_labels);
    n_classes = length(unique_classes);
         
    n_train_samples = length(training_labels);
    n_test_samples = length(test_labels);
        
    class_wise_accuracy = zeros(n_classes, 1);    
    confusion_matrix = zeros(n_classes, n_classes); 
    
    for i = 1:n_classes          
        class = unique_classes(i);

        c_tr_ind = (training_labels == class);
        c_tt_ind = (test_labels == class);
        
        tr_labels = -1*ones(n_train_samples, 1);        
        tr_labels(c_tr_ind) = 1;   
        tt_labels = -1*ones(n_test_samples, 1);
        tt_labels(c_tt_ind) = 1;
          
        temp = find(test_labels == class);
        
        % extra
        extra_options.classwt = [1 2];
        extra_options.nodesize = 2;
        extra_options.proximity = 1;
        extra_options.oob_prox = 0;
        
        rf_model = classRF_train(K_train_train{i}, tr_labels, 100, 0, extra_options);

        predicted_labels = classRF_predict(K_test_train{i}, rf_model);

        class_wise_accuracy(i) =...
            length(find(predicted_labels == tt_labels))...
            / length(temp);
        
        confusion_matrix(i, :) = hist(predicted_labels, unique_classes) / length(temp);
        
    end
    
    total_accuracy = sum(class_wise_accuracy) / n_test_samples;
      
end
