# remove object from my environment
rm(list = ls())

# set working directory
setwd("C:/Users/zhwja/OneDrive - The University of Chicago/Health Policy_RA/Periodontal_ML_final")

# Load necessary libraries
library(dplyr)
library(survey)
library(pROC)
library(glmnet)
library(Matrix)
library(caret)
library(broom)

# ---- Load data ----
df <- read.csv("data_imputed_avg.csv", check.names = FALSE)

# ---- Outcome: Severe vs not ----
if (!"severe_PD" %in% names(df)) {
  stopifnot("imputed_PD_severity" %in% names(df))
  df$severe_PD <- as.integer(df$imputed_PD_severity == "Severe")
}
df$severe_PD <- as.integer(df$severe_PD)

# ---- Choose survey design variables (auto-detect weights) ----
wt_var <- dplyr::case_when(
  "WTMEC6YR" %in% names(df) ~ "WTMEC6YR",
  "WTMEC4YR" %in% names(df) ~ "WTMEC4YR",
  "WTMEC2YR" %in% names(df) ~ "WTMEC2YR",
  TRUE ~ NA_character_
)
if (is.na(wt_var)) stop("Could not find a NHANES weight (e.g., WTMEC2YR/4YR/6YR).")

psu_var   <- if ("SDMVPSU"  %in% names(df)) "SDMVPSU"  else stop("Need SDMVPSU.")
strata_var<- if ("SDMVSTRA" %in% names(df)) "SDMVSTRA" else stop("Need SDMVSTRA.")

# ======================= Predictors =======================
# Exclude missing_teeth (to use PD status to predict it later).
# We also avoid periodontal features/targets to prevent leakage.

cand_demo_soc <- c("age","sex","race","education",
                   "marital_status","poverty_level","employment",
                   "US_Citizen","insurance_type",
                   "Food_Security_4lvl","food_exp_per_capita")

cand_behavior <- c("smoking_status","imputed_drinking_status", "Rountine_Care")

# ---- Choose HEI with fallback ----
cand_hei <- intersect(c("HEI_MC2D","day1_HEI_Q","HEI_NCI2D_total"), names(df))
if (length(cand_hei) == 0) warning("No HEI variable found; continuing without HEI.")

# Chronic comorbidities
cand_chronic <- c("hypertension","diabetes","high_chol","arthritis",
                  "BMI_category","depression_status","gout","CVD")

# # Optional HEI component foods (keep if you want fine-grained diet; can increase p a lot)
# cand_food_groups <- c(
#   "total_fruit_cup","whole_fruit_cup","total_veg_cup","greens_beans_cup",
#   "whole_grains_oz","dairy_cup","total_protein_oz","seafood_plant_oz",
#   "refined_grains_oz","total_sodium_mg","total_added_sugars_g",
#   "total_sfa_g","total_kcal"
# )

# Build union of candidate predictors that actually exist in df
covariates_raw <- c(cand_demo_soc, cand_behavior, cand_hei[1], cand_chronic)
covariates_raw <- intersect(covariates_raw, names(df))


# Explicitly drop leakage/undesired cols
drop_patterns <- c("^missing_teeth$", "^tooth_count$",               # you requested
                   "^imputed_tooth_", "^tooth_\\d{2}_(ICAL|PD)$",     # tooth/site info
                   "^imputed_?PD", "PD_severity", "imputed_severity") # targets
keep <- covariates_raw[ !Reduce(`|`, lapply(drop_patterns, function(p) grepl(p, covariates_raw, ignore.case = TRUE))) ]

# Factor-ize common categoricals
to_factor <- intersect(c("sex","race","education","marital_status","employment",
                         "US_Citizen","insurance_type","Food_Security_4lvl",
                         "smoking_status","imputed_drinking_status","Rountine_Care",
                         "BMI_category","depression_status","CVD","arthritis",
                         "hypertension","diabetes","high_chol","gout"), keep)
df[to_factor] <- lapply(df[to_factor], function(x) as.factor(x))

# ======================= LASSO screening (with weights) =======================
# Build model frame and drop NAs for LASSO stage
df_lasso <- df %>%
  select(all_of(keep), severe_PD, all_of(c(wt_var, psu_var, strata_var))) %>%
  na.omit()

# Model matrix (one-hot, no intercept)
x <- model.matrix(severe_PD ~ . - 1, data = df_lasso[, c("severe_PD", keep), drop = FALSE])
y <- df_lasso$severe_PD

# Normalize weights for numerical stability in glmnet
w <- df_lasso[[wt_var]]
w <- w / mean(w, na.rm = TRUE)

set.seed(42)
cv_lasso <- cv.glmnet(x, y, family = "binomial", alpha = 1,
                      nfolds = 10, weights = w)

# Plot cross-validation results
plot(cv_lasso)

# Chosen lambda (more stable: 1-SE rule)
lambda_star <- cv_lasso$lambda.1se
lasso_coef  <- coef(cv_lasso, s = "lambda.1se")
sel_terms   <- rownames(lasso_coef)[as.numeric(lasso_coef) != 0]
sel_terms   <- setdiff(sel_terms, "(Intercept)")

# Map dummy columns back to base variables (original terms)
assign_vec  <- attr(x, "assign")
term_labels <- attr(terms(severe_PD ~ ., data = df_lasso[, c("severe_PD", keep), drop = FALSE]), "term.labels")
col_map     <- data.frame(col = colnames(x), base = term_labels[assign_vec], stringsAsFactors = FALSE)
selected_base_vars <- unique(col_map$base[col_map$col %in% sel_terms])

cat("Selected base variables (LASSO, lambda.1se):\n")
print(selected_base_vars)

# ======================= Survey-weighted refit =======================
# Build reduced dataset for svyglm (re-include rows with complete data on reduced set)
svy_vars <- c(wt_var, psu_var, strata_var, "severe_PD")
df_reduced <- df %>%
  select(all_of(selected_base_vars), all_of(svy_vars)) %>%
  na.omit()

nhanes_design <- svydesign(
  ids = as.formula(paste0("~", psu_var)),
  strata = as.formula(paste0("~", strata_var)),
  weights = as.formula(paste0("~", wt_var)),
  data = df_reduced,
  nest = TRUE
)

f_reduced <- as.formula(paste("severe_PD ~", paste(selected_base_vars, collapse = " + ")))
logit_lasso_refit <- svyglm(f_reduced, design = nhanes_design, family = quasibinomial())

cat("\nSurvey-weighted logistic regression (refit on LASSO-selected vars):\n")
print(summary(logit_lasso_refit))

# ======================= Evaluation (weighted AUC if possible) =======================
# Predictions on the analysis frame used by survey design
df_reduced$pred_prob <- as.numeric(predict(logit_lasso_refit, type = "response"))

# Weighted AUC = weighted Mann–Whitney U with proper tie handling
weighted_auc <- function(labels, scores, weights) {
  stopifnot(length(labels)==length(scores), length(scores)==length(weights))
  d <- data.frame(y=as.integer(labels), s=as.numeric(scores), w=as.numeric(weights))
  d <- d[!is.na(d$y) & !is.na(d$s) & !is.na(d$w), ]
  # Aggregate by unique score to handle ties cleanly
  agg <- d %>%
    dplyr::group_by(s) %>%
    dplyr::summarise(w1 = sum(w[y==1]), w0 = sum(w[y==0]), .groups="drop") %>%
    dplyr::arrange(s)
  
  # Cumulative weight of negatives strictly below a score
  agg <- agg %>%
    dplyr::mutate(
      cum_w0_below = dplyr::lag(cumsum(w0), default = 0),
      # Contribution: positives at this score paired with all negatives below,
      # plus half of ties (positives vs negatives with equal scores)
      contrib = w1 * cum_w0_below + 0.5 * w1 * w0
    )
  
  W1 <- sum(agg$w1); W0 <- sum(agg$w0)
  if (W1 == 0 || W0 == 0) return(NA_real_)
  auc <- sum(agg$contrib) / (W1 * W0)
  as.numeric(auc)
}

# Using df_reduced data frame
wAUC <- weighted_auc(df_reduced$severe_PD, df_reduced$pred_prob, df_reduced[[wt_var]])
cat(sprintf("Weighted AUC (custom): %.4f\n", wAUC))

# Un-weighted AUC as reference
auc_unw <- as.numeric(pROC::auc(pROC::roc(df_reduced$severe_PD, df_reduced$pred_prob, quiet = TRUE))) 
cat(sprintf("\nUnweighted AUC (fallback): %.4f\n", auc_unw))

# -------Including Design-aware Weights----------
set.seed(2026)

design_boot_auc <- function(df, psu, strata, weight, label_col, score_col,
                            B = 400) {
  # Prepare index lists by strata -> PSUs
  df <- df[, c(psu, strata, weight, label_col, score_col)]
  names(df) <- c("psu","strata","w","y","score")
  
  strata_list <- split(df, df$strata)
  aucs <- numeric(B)
  
  for (b in seq_len(B)) {
    sampled_blocks <- lapply(strata_list, function(sub) {
      psus <- unique(sub$psu)
      # sample PSUs with replacement within each stratum
      draw <- sample(psus, size = length(psus), replace = TRUE)
      # concatenate all selected PSUs (with multiplicity)
      do.call(rbind, lapply(draw, function(p) sub[sub$psu == p, , drop=FALSE]))
    })
    boot_df <- do.call(rbind, sampled_blocks)
    
    aucs[b] <- weighted_auc(boot_df$y, boot_df$score, boot_df$w)
  }
  
  est <- weighted_auc(df$y, df$score, df$w)
  se  <- stats::sd(aucs, na.rm = TRUE)
  ci  <- est + c(-1,1) * 1.96 * se
  list(est = est, se = se, ci = ci, replicates = aucs)
}

boot_res <- design_boot_auc(
  df   = df_reduced,
  psu  = psu_var,
  strata = strata_var,
  weight = wt_var,
  label_col = "severe_PD",
  score_col = "pred_prob",
  B = 400
)

cat(sprintf("Design-aware weighted AUC: %.4f (95%% CI %.4f–%.4f)\n",
            boot_res$est, boot_res$ci[1], boot_res$ci[2]))

## --- (C) Design-aware ROC bands (PSU/strata bootstrap) ---
set.seed(2027)
B <- 200
fpr_grid <- seq(0, 1, length.out = 101)

get_weighted_roc <- function(y, score, w, fpr_grid) {
  ro <- pROC::roc(y, score, weights = w, quiet = TRUE, direction = "<")
  fpr <- 1 - rev(ro$specificities); tpr <- rev(ro$sensitivities)
  approx(x = fpr, y = tpr, xout = fpr_grid, ties = "ordered", rule = 2)$y
}

# point estimate curve on common grid
tpr_hat <- get_weighted_roc(df_reduced$severe_PD, df_reduced$pred_prob,
                            df_reduced[[wt_var]], fpr_grid)

# bootstrap curves
dfb <- df_reduced[, c(psu_var, strata_var, wt_var, "severe_PD", "pred_prob")]
names(dfb) <- c("psu","strata","w","y","score")
strata_list <- split(dfb, dfb$strata)

tpr_mat <- matrix(NA_real_, nrow = B, ncol = length(fpr_grid))
for (b in seq_len(B)) {
  boot_blocks <- lapply(strata_list, function(sub) {
    psus <- unique(sub$psu)
    draw <- sample(psus, size = length(psus), replace = TRUE)
    do.call(rbind, lapply(draw, function(p) sub[sub$psu == p, , drop=FALSE]))
  })
  boot_df <- do.call(rbind, boot_blocks)
  tpr_mat[b, ] <- get_weighted_roc(boot_df$y, boot_df$score, boot_df$w, fpr_grid)
}

tpr_lo <- apply(tpr_mat, 2, function(z) stats::quantile(z, 0.025, na.rm = TRUE))
tpr_hi <- apply(tpr_mat, 2, function(z) stats::quantile(z, 0.975, na.rm = TRUE))

## --- (D) Plot: weighted ROC + bands + AUC with CI in the corner ---
plot(fpr_grid, tpr_hat, type = "n",
     xlab = "1 − Specificity (FPR)", ylab = "Sensitivity (TPR)",
     main = "Survey-weighted ROC with design-aware bands")
abline(a = 0, b = 1, col = "gray75", lty = 2)

# band
polygon(c(fpr_grid, rev(fpr_grid)),
        c(tpr_lo,    rev(tpr_hi)),
        border = NA, col = adjustcolor("#2ca02c", alpha.f = 0.18))

# curve
lines(fpr_grid, tpr_hat, lwd = 2, col = "#2ca02c")

# AUC text (top-left corner)
txt <- sprintf("Weighted AUC = %.3f\n95%% CI: %.3f–%.3f",
               boot_res$est, boot_res$ci[1], boot_res$ci[2])
usr <- par("usr")
text(x = usr[2] - 0.02*(usr[2]-usr[1]),
     y = usr[3] + 0.04*(usr[4]-usr[3]),
     labels = txt, adj = c(1,0), cex = 0.7)


#-------------Get top 5 strongest predictors---------------
# 1) All terms in the refit model (drop intercept)
terms_full <- attr(terms(formula(logit_lasso_refit)), "term.labels")

# 2) Design-aware partial Wald test for each term
wald_tab <- lapply(terms_full, function(tt) {
  rt <- regTermTest(logit_lasso_refit, as.formula(paste("~", tt)))
  # rt typically has elements: Ftest, ddf (num/den df), p
  data.frame(
    term = tt,
    F    = as.numeric(rt$Ftest),
    df1  = as.numeric(rt$ddf[1]),
    df2  = as.numeric(rt$ddf[2]),
    p    = as.numeric(rt$p),
    stringsAsFactors = FALSE
  )
}) %>% bind_rows() %>% arrange(desc(F))

# 3) Top 5 strongest predictors (design-aware)
top5_wald <- head(wald_tab, 5)
print(top5_wald)



# ======================= Outputs you might save =======================
# write.csv(data.frame(variable = selected_base_vars),
#           "lasso_selected_variables.csv", row.names = FALSE)

