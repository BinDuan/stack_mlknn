source ("./mlknn.R");

mlknn_stack_train <- function (X_train, Y_train, l0_k, l1_k, s0 = 1, s1 = 1, l0_dist = "euclidean", l1_dist = "euclidean")
{
  
  retval <- list ();
  
  mlknn_m   <- mlknn_train (X = X_train, Y = Y_train, k = l0_k, s = 1, dist = l0_dist)
  l1_target <- c ();
    
  pred_mlknn_train <- predict (mlknn_m, X_train, type = "prob"); 
  l1_target <- Y_train;
  retval$l0_pred_type <- "prob";
  
  pred_l1_val <- list ();
  evals_for_k <- list ();
  
  this_l1_m <- NA;
  this_l1_m_list <- NA;
  bn_dep <- NA;
  l1_lab_deps <- list ();

  this_l1_m <- mlknn_train (X = pred_mlknn_train, Y = l1_target, k = l1_k, s = s1, dist = l1_dist);
  
  retval$l0_m <- mlknn_m;
  retval$l1_m <- this_l1_m;
  retval$l0_k <- l0_k;
  retval$l1_k <- l1_k;
  retval$s0 <- s0;
  retval$s1 <- s1;
  retval$type_of_l1 <- type_of_l1;
  retval$use_bn <- use_bn;
  retval$bn_method <- bn_method;
  retval$bn_dep      <- bn_dep;
  retval$l1_lab_deps <- l1_lab_deps;
  retval$this_l1_m_list <- this_l1_m_list;
  
  class (retval) <- "mlknn_stack_t";
  
  return (retval);
}

predict.mlknn_stack_t <- function (model, X_test, type = "raw")
{
  l1_data    <- predict (model$l0_m, X_test, model$l0_pred_type);
  
  pred_final <- NA;
  if (model$use_bn == FALSE)
  {
    pred_final <- predict (model$l1_m, l1_data, type = type);
  }
  else
  {
    pred_final <- as.data.frame (matrix (NA, nrow = nrow (X_test), ncol = length (model$this_l1_m_list)));
    colnames (pred_final) <- names (model$this_l1_m_list);
    
    for (this_label in names (model$this_l1_m_list))
    {
      cat ("Processing Level 1, Label \"", this_label, "\"\n");
      this_dep <- model$l1_lab_deps[[this_label]];
      # If it is empty, make it depend on itself.
      if (length (this_dep))
      {
        this_dep <- this_label;
      }
      # pred_final[,this_label] <- predict (model$this_l1_m_list[[this_label]], l1_data[,this_dep,drop=F], type = "class")[,];
      pred_final[,this_label] <- apply (predict (model$this_l1_m_list[[this_label]], l1_data, type = "prob"), 1, which.max) - 1;
    }
  }
  
  return (pred_final);
}

