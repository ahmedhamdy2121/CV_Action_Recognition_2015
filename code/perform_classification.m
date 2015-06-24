function [] = perform_classification(root_dir, subject_labels, action_labels,...
    tr_subjects, te_subjects)

    results_dir = [root_dir, '/results'];
    mkdir(results_dir);

    % number of persons
    n_tr_te_splits = size(tr_subjects, 1);
    disp (['Number of persons: ' num2str(n_tr_te_splits)])
    
    % number of actions
    n_classes = length(unique(action_labels));  
    disp (['Number of actions: ' num2str(n_classes)])
   
    C_val = 1;
          
    loadname = 'linear_kernel';        

    dir = [root_dir, '/dtw_warped_pyramid_lf_fourier_kernels'];
    loadfile_tag = 'warped_pyramid_lf_fourier_kernels';    
    
    % total accuracy
    total_accuracy = zeros(n_tr_te_splits, 1);  
    
    % accuracy for action per person
    cw_accuracy = zeros(n_tr_te_splits, n_classes);
    
    confusion_matrices = cell(n_tr_te_splits, 1);
    
    % parallel
    parfor i = 1:n_tr_te_splits         
        tr_subject_ind = ismember(subject_labels, tr_subjects(i,:));
        te_subject_ind = ismember(subject_labels, te_subjects(i,:));        
        tr_labels = action_labels(tr_subject_ind);
        te_labels = action_labels(te_subject_ind);

        K_train_train = cell(n_classes, 1);
        K_test_train = cell(n_classes, 1);
        for class = 1:n_classes
            
            data = load ([dir, '/', loadfile_tag, '_split_',...
                num2str(i), '_class_', num2str(class)], loadname);

            K = data.(loadname);

            K_train_train{class} = K(tr_subject_ind, tr_subject_ind);
            K_test_train{class} = K(te_subject_ind, tr_subject_ind);
        end

        % I will change this line here only
        [total_accuracy(i), cw_accuracy(i,:), confusion_matrices{i}] =...
            kernel_RF_modified(K_train_train,...
            K_test_train, tr_labels, te_labels, C_val);

    end

    avg_total_accuracy = mean(total_accuracy);
    avg_cw_accuracy = mean(cw_accuracy);

    avg_confusion_matrix = zeros(size(confusion_matrices{1}));
    for j = 1:length(confusion_matrices)
        avg_confusion_matrix = avg_confusion_matrix + confusion_matrices{j};
    end
    avg_confusion_matrix = avg_confusion_matrix / length(confusion_matrices);
    
    save ([results_dir, '/classification_results.mat'],...
        'total_accuracy', 'cw_accuracy', 'avg_total_accuracy',...
        'avg_cw_accuracy', 'confusion_matrices', 'avg_confusion_matrix');
    
    disp ('Results:')
    disp (['Total Accuracy: ' num2str(avg_total_accuracy)])
    
    disp ('Accuracy per action: ')
    
    if n_classes == 10
        %% for UTKinect
        action_names = {'walk', 'sit down', 'stand up', 'pick up', 'carry', 'throw', 'push', 'pull', 'wave hands', 'clap hands'};
    elseif n_classes == 9
        %% for Florence 3D
        action_names = {'wave', 'drink from bottle', 'answer phone', 'clap', 'tight lace', 'sit down', 'stand up', 'read watch', 'bow'};
    end
    
    for j = 1:n_classes
        disp ([action_names{j} ' >> ' num2str(total_accuracy(j))])
    end
    
end
