mlknn_train <- function (X, Y, k, s = 1, cls_prob = FALSE, feedback = FALSE, dist = "euclidean")
{
  if (mlcheck_data (X, Y) == FALSE) { stop ("Returning"); }
  
 
  if ((k < 1) || (k%%1 != 0))
  {
    stop ("The value of \"k\" should be a non-zero positive integer\n");
  }
  
  if ((s < 1))
  {
    stop ("The value of \"s\" should be a non-zero positive integer\n");
  }
  
  mlknn_model_info <- list ();
  mlknn_model_info$data  <- X;
  mlknn_model_info$label <- Y;
  mlknn_model_info$cls_prob <- cls_prob;
  mlknn_model_info$all_lab_names <- colnames (Y);
  mlknn_model_info$mlknn_params <- list (k=k, s=s);
  mlknn_model_info$distance <- dist;
  
  if (feedback == TRUE)
  { cat ("MLkNN Setup done\n"); }
  
  class (mlknn_model_info) <- "mlknn_ml";
  
  return (mlknn_model_info);
}

predict.mlknn_ml <- function (model, X, type = "raw", feedback = FALSE)
{
  if (class (model) != "mlknn_ml")
  {
    stop ("An instance of type \"mlknn_ml\" was expected");
  }
  
  Y <- mlknn_internal (train_data  = model$data, 
		       train_label = model$label, 
		       test_data   = X, 
		       k           = model$mlknn_params$k, 
		       s           = model$mlknn_params$s,
		       dist        = model$distance);
  
  if (type == "prob")
  {
    Y <- as.data.frame (Y$prob);
  }
  else if (type == "raw")
  {
    Y <- as.data.frame (Y$labels);
  }
  
  colnames (Y) <- model$all_lab_names;
  
  return (Y);
}



mlknn_internal <- function (train_data, train_label, test_data, k, s, weights=NULL, p=2, dist = "euclidean")
{
  # TODO: Assert the train_label is binary vector
  tot_labs <- ncol (train_label);
  m        <- nrow (train_data);
  C        <- matrix (c (0), nrow = m, ncol = tot_labs, byrow = TRUE);
  
  # Assert the length of weights is ncol (train_data) 
  # TODO: Remove if not needed
  if (is.null (weights))
  {
    weights <- rep (1, ncol (train_data));
  }
  
  if (length (weights) != ncol (train_data))
  {
    stop ("Incorrect length of weights: ", length (weights), ". Expected length: ", ncol (train_data),  "\n");
  }
  
  for (i in 1:ncol(train_label))
  {
    if (is.factor(train_label[,i]))
    {
      train_label[,i] <- as.integer (train_label[,i]);
    }
  }
  
  # Computing prior probabilities for P(H_{b}^{l})
  p_h1 <- rep (0, tot_labs);
  p_h0 <- rep (0, tot_labs);
  for (l in 1:tot_labs)
  {
    p_h1[l] <- (s + sum (train_label[,l])) / (2 * s + m);
  }
  p_h0 = 1 - p_h1;
  
  # Computing posterior probabilities P(E_{j}^{l}|H_{b}^{l}) 
  E_h1 <- matrix (c (0), nrow = tot_labs, ncol = (k+1), byrow = TRUE);
  E_h0 <- matrix (c (0), nrow = tot_labs, ncol = (k+1), byrow = TRUE);
  
  # Get a pairwise distance map
  #   w_map <- as.matrix (dist (train_data, diag = TRUE, upper = TRUE));
  # TODO: Use nn2, it is faster as it implements kd tree knn search.
  w_map <- as.matrix (daisy (train_data, metric = dist, weights = weights)); # The "dist" is set by the function arg
  
  rownames (w_map) <- rownames (train_data);
  colnames (w_map) <- rownames (train_data);
  diag (w_map)     <- Inf;
  
  for (l in 1:tot_labs)
  {
    cat ("\r                              ");
    
    count             <- rep (0, (k+1));
    count_dash        <- rep (0, (k+1));
    
    for (i in 1:m)
    {
      cat ("\rlabel = ", l, "data = ", i);
      sorted_obj <- sort (w_map[,i], method = "shell", decreasing = FALSE, index.return = TRUE);
      knn_set    <- sorted_obj$ix[1:k];
      delta      <- sum (train_label[knn_set,l]);
      C[i,l]     <- delta;
      if (train_label[i,l] == 1)
      {
        count[delta+1] = count[delta+1] + 1;
      }
      else
      {
        count_dash[delta+1] = count_dash[delta+1] + 1;
      }
    }
    
    count_sum      <- sum (count);
    count_dash_sum <- sum (count_dash);
    for (j in 0:k) # Note the range, 0:k . we are mapping 0 to 1, and so on.
    {
      E_h1[l,j+1] <- (s + count[j+1])      / (s * (k + 1) + count_sum);
      E_h0[l,j+1] <- (s + count_dash[j+1]) / (s * (k + 1) + count_dash_sum);
    }
  }
  
  
  cat ("\n");
  m_test    <- nrow (test_data);
  test_pred <- matrix (c (0), nrow = m_test, ncol = tot_labs, byrow = TRUE);
  r_pred    <- matrix (c (0), nrow = m_test, ncol = tot_labs, byrow = TRUE);
  for (i in 1:m_test)
  {
    dist_list       <- rdist (train_data, test_data[i,,drop=FALSE]);
    sorted_obj      <- sort (dist_list, method="shell", index.return = TRUE);
    tdata_i_knn_set <- sorted_obj$ix[1:k];
    
    C_test <- rep (0, tot_labs);
    for (l in 1:tot_labs)
    {
      C_test[l] <- sum (train_label[tdata_i_knn_set,l]);
      pl_1 <- (p_h1[l] * E_h1[l,C_test[l]+1]);
      pl_0 <- (p_h0[l] * E_h0[l,C_test[l]+1]);
      if (pl_1 > pl_0)
      {
        test_pred[i,l] <- 1;
      }
      else if (pl_1 < pl_0)
      {
        test_pred[i,l] <- 0;
      }
      # Random stuff
      else
      {
        random_int <- ceiling (runif (1) * 1000);
        test_pred[i, l] <- (random_int %% 2); # If odd, then 1 else 0.
      }
      
      r_pred[i,l]    <-  (pl_1/(pl_1 + pl_0));
    }
    cat ("\rpredict ", i);
  }

  cat ("\n");
  list (labels=test_pred, prob=r_pred);  
}
