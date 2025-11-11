# ======================= Setup =======================
rm(list = ls())

# set working directory
setwd("C:/Users/zhwja/OneDrive - The University of Chicago/Health Policy_RA/Periodontal_ML_final")


library(dplyr)
library(survey)
library(pROC)
library(glmnet)
library(Matrix)
library(caret)
library(ranger)     # RF with case.weights
library(xgboost)

# --------------------- Load data ---------------------
df <- read.csv("data_imputed_avg.csv", check.names = FALSE)

# Outcome (Severe vs not). If not present, derive from severity label
if (!"severe_PD" %in% names(df)) {
  stopifnot("imputed_PD_severity" %in% names(df))
  df$severe_PD <- as.integer(df$imputed_PD_severity == "Severe")
}
df$severe_PD <- as.integer(df$severe_PD)

# Detect survey design variables
wt_var <- dplyr::case_when(
  "WTMEC6YR" %in% names(df) ~ "WTMEC6YR",
  "WTMEC4YR" %in% names(df) ~ "WTMEC4YR",
  "WTMEC2YR" %in% names(df) ~ "WTMEC2YR",
  TRUE ~ NA_character_
)
if (is.na(wt_var)) stop("No NHANES exam weight found (WTMEC2YR/4YR/6YR).")

psu_var    <- if ("SDMVPSU"  %in% names(df)) "SDMVPSU"  else stop("Need SDMVPSU.")
strata_var <- if ("SDMVSTRA" %in% names(df)) "SDMVSTRA" else stop("Need SDMVSTRA.")

# ------------------ Covariates (current set) ------------------
cand_demo_soc <- c("age","sex","race","education",
                   "marital_status","poverty_level","employment",
                   "US_Citizen","insurance_type","Food_Security_4lvl",
                   "food_exp_per_capita")
cand_behavior <- c("smoking_status","imputed_drinking_status","Rountine_Care")

cand_hei      <- intersect(c("HEI_MC2D","HEI_NCI2D_total"), names(df))
cand_hei      <- cand_hei[1]  # pick one available summary score

cand_chronic  <- c("hypertension","diabetes","high_chol","arthritis",
                   "BMI_category", "depression_status","gout","CVD")

covariates_raw <- c(cand_demo_soc, cand_behavior, cand_hei, cand_chronic)
covariates     <- intersect(covariates_raw, names(df))

# Factor-ize categorical predictors
to_factor <- intersect(c("sex","race","education","marital_status","employment",
                         "US_Citizen","insurance_type","Food_Security_4lvl",
                         "smoking_status","imputed_drinking_status","Rountine_Care",
                         "BMI_category","depression_status","CVD","arthritis",
                         "hypertension","diabetes","high_chol","gout"), covariates)
df[to_factor] <- lapply(df[to_factor], factor)

# Analysis frame
df_all <- df %>%
  select(all_of(covariates), severe_PD, all_of(c(wt_var, psu_var, strata_var))) %>%
  na.omit()

# ==== Helper: stratified K-folds by outcome only ====
set.seed(42)
K <- 5
folds_train_idx <- caret::createFolds(df_all$severe_PD, k = K, returnTrain = TRUE)
folds_test_idx  <- lapply(folds_train_idx, function(tr) setdiff(seq_len(nrow(df_all)), tr))

# ==== Helpers: weighted AUC + design-aware evaluation (same as before) ====
coerce_binary01 <- function(labels, positive = c("1","Yes")) {
  y <- labels
  if (is.factor(y)) ch <- as.character(y) else if (is.character(y)) ch <- y else return(as.numeric(y))
  if (all(ch %in% c("0","1"))) return(as.numeric(ch))
  if (all(ch %in% c("No","Yes"))) return(as.numeric(ch == "Yes"))
  as.numeric(ch == positive[1])
}

weighted_auc <- function(labels, scores, weights) {
  y <- coerce_binary01(labels)
  d <- data.frame(y = y, s = as.numeric(scores), w = as.numeric(weights))
  d <- d[complete.cases(d), ]
  W1 <- sum(d$w[d$y==1]); W0 <- sum(d$w[d$y==0]); if (W1==0||W0==0) return(NA_real_)
  agg <- d |>
    dplyr::group_by(s) |>
    dplyr::summarise(w1 = sum(w[y==1]), w0 = sum(w[y==0]), .groups="drop") |>
    dplyr::arrange(s) |>
    dplyr::mutate(cum_w0_below = dplyr::lag(cumsum(w0), default = 0),
                  contrib = w1 * cum_w0_below + 0.5 * w1 * w0)
  as.numeric(sum(agg$contrib)/(W1*W0))
}

design_boot_auc <- function(df_pred, label, score, wname, psu, strata, B = 400) {
  d <- df_pred[, c(label, score, wname, psu, strata)]; names(d) <- c("y","score","w","psu","strata")
  d <- d[complete.cases(d), ]
  W1 <- sum(d$w[coerce_binary01(d$y)==1]); W0 <- sum(d$w[coerce_binary01(d$y)==0])
  if (W1==0 || W0==0) return(list(est=NA_real_, ci=c(NA,NA), reps=rep(NA_real_, B)))
  strata_list <- split(d, d$strata)
  aucs <- numeric(B)
  for (b in seq_len(B)) {
    boot_blocks <- lapply(strata_list, function(sub){
      psus <- unique(sub$psu); draw <- sample(psus, length(psus), replace=TRUE)
      do.call(rbind, lapply(draw, function(p) sub[sub$psu==p, , drop=FALSE]))
    })
    boot_df <- do.call(rbind, boot_blocks)
    aucs[b] <- weighted_auc(boot_df$y, boot_df$score, boot_df$w)
  }
  est <- weighted_auc(d$y, d$score, d$w); se <- stats::sd(aucs, na.rm=TRUE)
  ci <- est + c(-1,1)*1.96*se
  list(est=est, ci=ci, reps=aucs)
}

get_weighted_roc <- function(y, score, w, fpr_grid) {
  ro <- pROC::roc(y, score, weights = w, quiet = TRUE, direction = "<")
  fpr <- 1 - rev(ro$specificities); tpr <- rev(ro$sensitivities)
  approx(x=fpr, y=tpr, xout=fpr_grid, ties="ordered", rule=2)$y
}

plot_weighted_roc_with_bands <- function(df_pred, score_col,
                                         label_col="severe_PD", weight_col=wt_var,
                                         psu_col=psu_var, strata_col=strata_var,
                                         B_auc=400, B_band=200, main_title=NULL) {
  d <- df_pred[, c(label_col, score_col, weight_col, psu_col, strata_col)]
  names(d) <- c("y","score","w","psu","strata")
  auc_res <- design_boot_auc(d, "y","score","w","psu","strata", B=B_auc)
  set.seed(2027); fpr_grid <- seq(0,1,length.out=101)
  tpr_hat <- get_weighted_roc(d$y, d$score, d$w, fpr_grid)
  strata_list <- split(d, d$strata)
  tpr_mat <- matrix(NA_real_, nrow=B_band, ncol=length(fpr_grid))
  for (b in seq_len(B_band)) {
    boot_blocks <- lapply(strata_list, function(sub){
      psus <- unique(sub$psu); draw <- sample(psus, length(psus), replace=TRUE)
      do.call(rbind, lapply(draw, function(p) sub[sub$psu==p, , drop=FALSE]))
    })
    boot_df <- do.call(rbind, boot_blocks)
    tpr_mat[b,] <- get_weighted_roc(boot_df$y, boot_df$score, boot_df$w, fpr_grid)
  }
  tpr_lo <- apply(tpr_mat, 2, function(z) stats::quantile(z, 0.025, na.rm=TRUE))
  tpr_hi <- apply(tpr_mat, 2, function(z) stats::quantile(z, 0.975, na.rm=TRUE))
  if (is.null(main_title)) main_title <- paste0("Survey-weighted ROC: ", score_col)
  plot(fpr_grid, tpr_hat, type="n", xlab="1 − Specificity (FPR)", ylab="Sensitivity (TPR)", main=main_title)
  abline(a=0,b=1,col="gray75",lty=2)
  polygon(c(fpr_grid, rev(fpr_grid)), c(tpr_lo, rev(tpr_hi)), border=NA, col=adjustcolor("#2ca02c", 0.20))
  lines(fpr_grid, tpr_hat, lwd=2, col="#2ca02c")
  txt <- sprintf("Weighted AUC = %.3f\n95%% CI: %.3f–%.3f", auc_res$est, auc_res$ci[1], auc_res$ci[2])
  usr <- par("usr"); text(x=usr[2]-0.02*(usr[2]-usr[1]), y=usr[3]+0.04*(usr[4]-usr[3]),
                          labels=txt, adj=c(1,0), cex=0.9)
  invisible(auc_res)
}


# ==== LOGISTIC (FULL) — OOF via simple K-folds, weighted glm ====
oof_logit_full <- rep(NA_real_, nrow(df_all))
form_full <- as.formula(paste("severe_PD ~", paste(covariates, collapse = " + ")))
for (k in seq_len(K)) {
  tr <- folds_train_idx[[k]]; te <- folds_test_idx[[k]]
  w_tr <- df_all[[wt_var]][tr]; w_tr <- w_tr / mean(w_tr, na.rm=TRUE)
  fit <- glm(form_full, data = df_all[tr,], family = binomial(), weights = w_tr)
  oof_logit_full[te] <- as.numeric(predict(fit, newdata = df_all[te,], type = "response"))
}
df_all$pred_logit_full_oof <- oof_logit_full

# ==== LOGISTIC (LASSO-screened) — screen once, then OOF weighted glm ====
x_all <- model.matrix(severe_PD ~ . - 1, data = df_all[, c("severe_PD", covariates), drop=FALSE])
y_all <- df_all$severe_PD
w_all <- df_all[[wt_var]] / mean(df_all[[wt_var]], na.rm=TRUE)

set.seed(42)
cv_las <- cv.glmnet(x_all, y_all, family="binomial", alpha=1, nfolds=10, weights=w_all)
coef_las <- coef(cv_las, s = cv_las$lambda.1se)
sel_terms <- setdiff(rownames(coef_las)[as.numeric(coef_las)!=0], "(Intercept)")
assign_vec  <- attr(x_all, "assign")
term_labels <- attr(terms(severe_PD ~ ., data = df_all[, c("severe_PD", covariates), drop=FALSE]), "term.labels")
col_map     <- data.frame(col = colnames(x_all), base = term_labels[assign_vec], stringsAsFactors = FALSE)
sel_bases   <- unique(col_map$base[col_map$col %in% sel_terms])

if (length(sel_bases)==0) {
  coef_las <- coef(cv_las, s=cv_las$lambda.min)
  sel_terms <- setdiff(rownames(coef_las)[as.numeric(coef_las)!=0], "(Intercept)")
  sel_bases <- unique(col_map$base[col_map$col %in% sel_terms])
}
form_las <- as.formula(paste("severe_PD ~", paste(sel_bases, collapse = " + ")))

oof_logit_las <- rep(NA_real_, nrow(df_all))
for (k in seq_len(K)) {
  tr <- folds_train_idx[[k]]; te <- folds_test_idx[[k]]
  w_tr <- df_all[[wt_var]][tr]; w_tr <- w_tr / mean(w_tr, na.rm=TRUE)
  fit <- glm(form_las, data = df_all[tr,], family = binomial(), weights = w_tr)
  oof_logit_las[te] <- as.numeric(predict(fit, newdata = df_all[te,], type = "response"))
}

df_all$pred_logit_lasso_oof <- oof_logit_las


## ---- NESTED LASSO OOF (replaces the single-pass LASSO) ----
library(glmnet)

# Inputs assumed to exist from current script:
# df_all, covariates, wt_var, K, folds_train_idx, folds_test_idx

oof_logit_lasso_nested <- rep(NA_real_, nrow(df_all))

for (k in seq_len(K)) {
  tr <- folds_train_idx[[k]]
  te <- folds_test_idx[[k]]
  
  dtr <- df_all[tr, c("severe_PD", covariates, wt_var), drop = FALSE]
  dte <- df_all[te, c("severe_PD", covariates), drop = FALSE]
  
  # model matrix (one-hot, no intercept) on the **training** data only
  x_tr <- model.matrix(severe_PD ~ . - 1, data = dtr[, c("severe_PD", covariates), drop = FALSE])
  y_tr <- dtr$severe_PD
  w_tr <- dtr[[wt_var]] / mean(dtr[[wt_var]], na.rm = TRUE)
  
  # 10-fold CV LASSO *inside* the training fold
  set.seed(42 + k)
  cvk <- cv.glmnet(x_tr, y_tr, family = "binomial", alpha = 1,
                   nfolds = 10, weights = w_tr)
  
  # pick 1-SE (fallback to lambda.min if empty)
  beta <- coef(cvk, s = cvk$lambda.1se)
  sel_terms <- setdiff(rownames(beta)[as.numeric(beta) != 0], "(Intercept)")
  if (length(sel_terms) == 0) {
    beta <- coef(cvk, s = cvk$lambda.min)
    sel_terms <- setdiff(rownames(beta)[as.numeric(beta) != 0], "(Intercept)")
  }
  
  # map back to base variables to form a clean formula
  assign_vec  <- attr(x_tr, "assign")
  term_labels <- attr(terms(severe_PD ~ ., data = dtr[, c("severe_PD", covariates), drop = FALSE]), "term.labels")
  col_map     <- data.frame(col = colnames(x_tr), base = term_labels[assign_vec], stringsAsFactors = FALSE)
  sel_bases   <- unique(col_map$base[col_map$col %in% sel_terms])
  
  # if still empty, use intercept-only (rare)
  if (length(sel_bases) == 0) {
    p_hat <- sum(w_tr[y_tr == 1]) / sum(w_tr)
    oof_logit_lasso_nested[te] <- p_hat
    next
  }
  
  # fit *weighted* logistic on training fold using the selected bases
  form_fold <- as.formula(paste("severe_PD ~", paste(sel_bases, collapse = " + ")))
  fit_fold  <- glm(form_fold, data = dtr, family = binomial(), weights = w_tr)
  
  # predict on the held-out fold
  oof_logit_lasso_nested[te] <- as.numeric(predict(fit_fold, newdata = dte, type = "response"))
}

df_all$pred_logit_lasso_nested_oof <- oof_logit_lasso_nested


# ==== RANDOM FOREST (FULL) — caret CV with weights; OOF from caret ====
df_rf <- df_all
df_rf$severe_PD <- factor(df_rf$severe_PD, levels=c(0,1), labels=c("No","Yes"))
ctrl_rf <- trainControl(method="cv", number=K, classProbs=TRUE, summaryFunction=twoClassSummary,
                        index=folds_train_idx, savePredictions="final", verboseIter=FALSE)
grid_rf <- expand.grid(mtry = pmax(2, round(c(0.5, 1, 2) * sqrt(length(covariates)))),
                       splitrule = c("gini","extratrees"),
                       min.node.size = c(5,10,20))
set.seed(42)
rf_tuned <- train(x = df_rf[,covariates], y = df_rf$severe_PD,
                  method="ranger", trControl=ctrl_rf, tuneGrid=grid_rf, metric="ROC",
                  importance="impurity", num.trees=1500,
                  weights = df_all[[wt_var]]/mean(df_all[[wt_var]], na.rm=TRUE))
preds_rf <- rf_tuned$pred
best <- rf_tuned$bestTune
rows_best <- with(preds_rf, mtry==best$mtry & splitrule==best$splitrule & min.node.size==best$min.node.size)
df_all$pred_rf_oof <- NA_real_
df_all$pred_rf_oof[preds_rf$rowIndex[rows_best]] <- preds_rf$Yes[rows_best]

# ==== XGBOOST (FULL) — caret CV with weights; OOF from caret ====
x_all_xgb <- model.matrix(~ . - 1, data = df_all[, covariates, drop=FALSE])
nzv <- nearZeroVar(x_all_xgb)
if (length(nzv)) x_all_xgb <- x_all_xgb[, -nzv, drop=FALSE]
y_all_fac <- factor(df_all$severe_PD, levels=c(0,1), labels=c("No","Yes"))
ctrl_xgb <- trainControl(method="cv", number=K, classProbs=TRUE, summaryFunction=twoClassSummary,
                         index=folds_train_idx, savePredictions="final", verboseIter=FALSE)
grid_xgb <- expand.grid(
  nrounds=c(400,800),
  max_depth=c(3,5,7),
  eta=c(0.05,0.10),
  gamma=c(0,1),
  colsample_bytree=c(0.8),
  min_child_weight=c(1,5),
  subsample=c(0.8)
)
# coerce numeric
grid_xgb <- within(grid_xgb, {
  nrounds <- as.integer(nrounds); max_depth <- as.integer(max_depth)
  eta <- as.numeric(eta); gamma <- as.numeric(gamma)
  colsample_bytree <- as.numeric(colsample_bytree)
  min_child_weight <- as.numeric(min_child_weight)
  subsample <- as.numeric(subsample)
})
set.seed(42)
xgb_tuned <- train(x = x_all_xgb, y = y_all_fac, method="xgbTree",
                   trControl=ctrl_xgb, tuneGrid=grid_xgb, metric="ROC",
                   weights = df_all[[wt_var]]/mean(df_all[[wt_var]], na.rm=TRUE))
preds_xgb <- xgb_tuned$pred
bestx <- xgb_tuned$bestTune
rows_bestx <- with(preds_xgb, nrounds==bestx$nrounds & max_depth==bestx$max_depth &
                     eta==bestx$eta & gamma==bestx$gamma &
                     colsample_bytree==bestx$colsample_bytree &
                     min_child_weight==bestx$min_child_weight & subsample==bestx$subsample)
df_all$pred_xgb_oof <- NA_real_
df_all$pred_xgb_oof[preds_xgb$rowIndex[rows_bestx]] <- preds_xgb$Yes[rows_bestx]

# ==== Design-aware evaluation on OOF predictions ====
auc_logit_full  <- design_boot_auc(df_all, "severe_PD", "pred_logit_full_oof",  wt_var, psu_var, strata_var, B=400)
# auc_logit_lasso <- design_boot_auc(df_all, "severe_PD", "pred_logit_lasso_oof", wt_var, psu_var, strata_var, B=400)
auc_logit_lasso_nested <- design_boot_auc(df_all, "severe_PD", "pred_logit_lasso_nested_oof", wt_var, psu_var, strata_var, B=400)
auc_rf          <- design_boot_auc(df_all, "severe_PD", "pred_rf_oof",          wt_var, psu_var, strata_var, B=400)
auc_xgb         <- design_boot_auc(df_all, "severe_PD", "pred_xgb_oof",         wt_var, psu_var, strata_var, B=400)



cat(sprintf("\nLOGIT (FULL, OOF)   AUC = %.4f (95%% CI %.4f–%.4f)\n",
            auc_logit_full$est, auc_logit_full$ci[1], auc_logit_full$ci[2]))
# cat(sprintf("LOGIT (LASSO, OOF)  AUC = %.4f (95%% CI %.4f–%.4f)\n",
#             auc_logit_lasso$est, auc_logit_lasso$ci[1], auc_logit_lasso$ci[2]))
cat(sprintf("LOGIT (LASSO, OOF)  AUC = %.4f (95%% CI %.4f–%.4f)\n",
            auc_logit_lasso_nested$est, auc_logit_lasso_nested$ci[1], auc_logit_lasso_nested$ci[2]))
cat(sprintf("RANDOM FOREST (OOF) AUC = %.4f (95%% CI %.4f–%.4f)\n",
            auc_rf$est, auc_rf$ci[1], auc_rf$ci[2]))
cat(sprintf("XGBOOST (OOF)       AUC = %.4f (95%% CI %.4f–%.4f)\n",
            auc_xgb$est, auc_xgb$ci[1], auc_xgb$ci[2]))

# par(mfrow=c(2,2))
plot_weighted_roc_with_bands(df_all, "pred_logit_full_oof",  main_title="Survey-weighted ROC – Logistic (Full, OOF)")
# plot_weighted_roc_with_bands(df_all, "pred_logit_lasso_oof", main_title="Survey-weighted ROC – Logistic (LASSO, OOF)")
plot_weighted_roc_with_bands(df_all, "pred_logit_lasso_nested_oof", main_title="Survey-weighted ROC – Logistic (LASSO, OOF)")
plot_weighted_roc_with_bands(df_all, "pred_rf_oof", main_title="Survey-weighted ROC – Random Forest (OOF)")
plot_weighted_roc_with_bands(df_all, "pred_xgb_oof", main_title="Survey-weighted ROC – XGBoost (OOF)")
# par(mfrow=c(1,1))

# # Choose which logistic to show; change to "pred_logit_lasso_nested_oof" if you prefer
# logit_col <- "pred_logit_lasso_nested_oof"
# 
# set.seed(2028)
# curve_logit <- compute_curve_and_band(df_all, logit_col, "severe_PD", wt_var, psu_var, strata_var)
# curve_rf    <- compute_curve_and_band(df_all, "pred_rf_oof", "severe_PD", wt_var, psu_var, strata_var)
# curve_xgb   <- compute_curve_and_band(df_all, "pred_xgb_oof","severe_PD", wt_var, psu_var, strata_var)
# 
# ## ---------- plot all on one figure ----------
# # Aesthetic choices
# col_line <- c(logit="#1f77b4", rf="#2ca02c", xgb="#d62728")          # blue/green/red
# col_band <- c(logit=adjustcolor(col_line["logit"], 0.15),
#               rf   =adjustcolor(col_line["rf"],    0.15),
#               xgb  =adjustcolor(col_line["xgb"],   0.15))
# 
# plot(curve_logit$fpr, curve_logit$tpr, type="n",
#      xlab="1 – Specificity (FPR)", ylab="Sensitivity (TPR)",
#      main="Survey-weighted ROC with design-aware bands (OOF)")
# 
# abline(a=0, b=1, col="gray75", lty=2)
# 
# # bands first (so lines draw on top)
# polygon(c(curve_logit$fpr, rev(curve_logit$fpr)), c(curve_logit$lo, rev(curve_logit$hi)),
#         border=NA, col=col_band["logit"])
# polygon(c(curve_rf$fpr,    rev(curve_rf$fpr)),    c(curve_rf$lo,    rev(curve_rf$hi)),
#         border=NA, col=col_band["rf"])
# polygon(c(curve_xgb$fpr,   rev(curve_xgb$fpr)),   c(curve_xgb$lo,   rev(curve_xgb$hi)),
#         border=NA, col=col_band["xgb"])
# 
# # ROC lines
# lines(curve_logit$fpr, curve_logit$tpr, lwd=2.2, col=col_line["logit"])
# lines(curve_rf$fpr,    curve_rf$tpr,    lwd=2.2, col=col_line["rf"])
# lines(curve_xgb$fpr,   curve_xgb$tpr,   lwd=2.2, col=col_line["xgb"])
# 
# # Legend with AUC (95% CI)
# leg_txt <- c(
#   sprintf("Logistic (LASSO)  AUC %.3f (%.3f–%.3f)", curve_logit$auc, curve_logit$ci[1], curve_logit$ci[2]),
#   sprintf("Random Forest     AUC %.3f (%.3f–%.3f)", curve_rf$auc,    curve_rf$ci[1],    curve_rf$ci[2]),
#   sprintf("XGBoost          AUC %.3f (%.3f–%.3f)", curve_xgb$auc,   curve_xgb$ci[1],   curve_xgb$ci[2])
# )
# legend("bottomright", bty="n",
#        legend=leg_txt,
#        lwd=2.2, col=unname(col_line), cex=0.9)
# 




## ---- Weighted calibration utilities ----
# optional: install.packages("isotone")
library(isotone)
suppressWarnings({ if (!requireNamespace("isotone", quietly = TRUE)) message("`isotone` not installed; skipping isotonic overlay.") })

# Weighted bin stats
weighted_mean <- function(x, w) sum(w * x, na.rm = TRUE) / sum(w, na.rm = TRUE)

calibration_bins <- function(df, p_col, y_col = "severe_PD", w_col, psu_col, strata_col,
                             G = 10, B = 200, seed = 2029) {
  d <- df[, c(p_col, y_col, w_col, psu_col, strata_col)]
  names(d) <- c("p","y","w","psu","strata")
  d <- d[complete.cases(d), ]
  d$bin <- cut(d$p, breaks = quantile(d$p, probs = seq(0, 1, length.out = G + 1)),
               include.lowest = TRUE, labels = FALSE)
  
  # point estimates per bin
  est <- d %>%
    group_by(bin) %>%
    summarise(
      pbar = weighted_mean(p, w),
      ybar = weighted_mean(y, w),
      wsum = sum(w),
      .groups = "drop"
    )
  
  # design-aware bootstrap CIs for ybar in each bin
  set.seed(seed)
  strata_list <- split(d, d$strata)
  boot_mat <- matrix(NA_real_, nrow = B, ncol = G)
  
  for (b in seq_len(B)) {
    boot_blocks <- lapply(strata_list, function(sub){
      psus <- unique(sub$psu)
      draw <- sample(psus, size = length(psus), replace = TRUE)
      do.call(rbind, lapply(draw, function(p) sub[sub$psu == p, , drop = FALSE]))
    })
    boot_df <- do.call(rbind, boot_blocks)
    tmp <- boot_df %>%
      group_by(bin) %>%
      summarise(ybar = weighted_mean(y, w), .groups = "drop")
    boot_mat[b, tmp$bin] <- tmp$ybar
  }
  
  ci_lo <- apply(boot_mat, 2, function(z) quantile(z, 0.025, na.rm = TRUE))
  ci_hi <- apply(boot_mat, 2, function(z) quantile(z, 0.975, na.rm = TRUE))
  
  est$lo <- ci_lo[est$bin]
  est$hi <- ci_hi[est$bin]
  
  list(points = est, data = d)
}

# Zoomed calibration plot (no Platt/Isotonic), with user-settable y_max.
# - cal: object returned by calibration_bins()
# - x_max: show only bins with p̄ <= x_max (default 0.5)
# - y_max: upper y-limit (NULL = auto from data)
# - min_weight_frac: drop bins whose total survey weight < this fraction of total (stabilizes tails)
# - loess_frac: smoothing span for the loess guide line
plot_calibration_zoom <- function(cal, x_max = 0.5, y_max = NULL,
                                  main = NULL, min_weight_frac = 0,
                                  loess_frac = 0.8) {
  est <- cal$points
  
  # Optionally drop ultra-light bins by total survey weight
  if (min_weight_frac > 0) {
    tot_w <- sum(cal$data$w, na.rm = TRUE)
    est <- est[est$wsum >= min_weight_frac * tot_w, , drop = FALSE]
  }
  
  # Keep only requested probability range
  est <- est[est$pbar <= x_max, , drop = FALSE]
  if (nrow(est) == 0) stop("No bins in requested x-range; increase x_max.")
  
  # y-axis limit (auto if y_max = NULL)
  y_upper <- if (is.null(y_max)) max(1, est$hi, na.rm = TRUE) else y_max
  
  if (is.null(main)) main <- sprintf("Calibration (zoom 0–%d%%)", round(100 * x_max))
  
  plot(est$pbar, est$ybar, pch = 19,
       xlim = c(0, x_max), ylim = c(0, y_upper),
       xlab = "Predicted probability", ylab = "Observed event rate (weighted)",
       main = main)
  
  # 45° line for perfect calibration
  abline(0, 1, col = "gray70", lty = 2)
  
  # Design-aware 95% CIs
  segments(est$pbar, pmax(0, est$lo), est$pbar, pmin(y_upper, est$hi),
           col = adjustcolor("gray30", 0.9))
  
  # Loess smooth (visual guide)
  lo <- suppressWarnings(lowess(est$pbar, est$ybar, f = loess_frac))
  lines(lo, lwd = 2, col = "#1f77b4")
  
  legend("topleft", bty = "n",
         legend = c("Binned (weighted)", "Loess smooth"),
         pch = c(19, NA), lty = c(NA, 1),
         col = c("black", "#1f77b4"), cex = 0.9)
}


# You already built this earlier:
cal_logit_nested <- calibration_bins(
  df = df_all, p_col = "pred_logit_lasso_nested_oof",
  y_col = "severe_PD", w_col = wt_var, psu_col = psu_var, strata_col = strata_var,
  G = 10, B = 200, seed = 2031
)
cal_logit_full <- calibration_bins(
  df = df_all, p_col = "pred_logit_full_oof",
  y_col = "severe_PD", w_col = wt_var, psu_col = psu_var, strata_col = strata_var,
  G = 10, B = 200, seed = 2031
)

plot_calibration_zoom(cal_logit_nested, x_max = 0.40, y_max = 0.40,
                      main = "Calibration – Logistic (LASSO Reduced, OOF)")
plot_calibration_zoom(cal_logit_full, x_max = 0.40, y_max = 0.40,
                      main = "Calibration – Logistic (FULL, OOF)")

