import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu, ks_2samp, f_oneway
from sklearn.preprocessing import StandardScaler

def chi_square(array1, array2):
    """
    Perform a Chi-square test for independence between two categorical variables.

    Parameters
    ----------
    array1 : array-like
        The first categorical variable.
    array2 : array-like
        The second categorical variable.

    Returns
    -------
    chi2 : float
        The test statistic.
    p : float
        The p-value of the test.
    dof : int
        The degrees of freedom.
    expected : ndarray
        The expected frequencies in each category.
    contingency_table : pandas.DataFrame
        The contingency table of the observed frequencies.
    """
    contingency_table = pd.crosstab(array1, array2)
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p, dof, expected, contingency_table


def anova_test(df, target_column, var):
    """
    Perform a one-way ANOVA test to compare means of a numerical variable across different categories of a categorical variable.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    target_column : str
        The name of the categorical target column.
    var : str
        The name of the numerical variable to test.

    Returns
    -------
    f_stat : float
        The test statistic.
    p_value : float
        The p-value of the test.
    """
    groups = [df[df[target_column] == category][var].dropna().values for category in df[target_column].unique()]
    f_stat, p_value = f_oneway(*groups)
    return f_stat, p_value


def numerical_test(array1, array2):
    """
    Perform statistical tests to compare two numerical variables.

    Parameters
    ----------
    array1 : array-like
        The first numerical variable.
    array2 : array-like
        The second numerical variable.

    Returns
    -------
    t_stat : float
        The t-test statistic.
    t_p : float
        The p-value of the t-test.
    ks_stat : float
        The Kolmogorov-Smirnov test statistic.
    ks_p : float
        The p-value of the Kolmogorov-Smirnov test.
    """
    scaler = StandardScaler()
    array = np.concatenate([array1, array2])
    array = scaler.fit_transform(array.reshape(-1, 1)).flatten()

    array1_norm = array[:len(array1)]
    array2_norm = array[len(array1):]

    t_stat, t_p = ttest_ind(array1_norm, array2_norm)
    ks_stat, ks_p = ks_2samp(array1_norm, array2_norm)

    return t_stat, t_p, ks_stat, ks_p


def statistical_tests_step1(df, target_column, analysis_columns):
    """
    Perform statistical tests to identify relevant variables influencing the target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    target_column : str
        The name of the target column.
    analysis_columns : list of str
        The list of columns to analyze.

    Returns
    -------
    output : dict
        A dictionary containing the results of the statistical tests for each variable.
    relevant_columns : list of str
        A list of columns that are statistically significant.
    """
    output = dict()
    relevant_columns = list()
    target_unique_values = df[target_column].nunique()

    for var in analysis_columns:
        try:
            if pd.api.types.is_numeric_dtype(df[var]):
                if target_unique_values == 2:
                    group1 = df[df[target_column] == df[target_column].unique()[0]][var].values
                    group2 = df[df[target_column] == df[target_column].unique()[1]][var].values
                    t_stat, t_p, ks_stat, ks_p = numerical_test(group1, group2)
                    output[var] = {"t_stat": t_stat, "t_p": t_p, "ks_stat": ks_stat, "ks_p": ks_p}
                    if t_p <= 0.05 and ks_p <= 0.05:
                        print(f"Relevant : {var}")
                        relevant_columns.append(var)
                else:
                    f_stat, p_value = anova_test(df, target_column, var)
                    output[var] = {"f_stat": f_stat, "p_value": p_value}
                    if p_value <= 0.05:
                        print(f"Relevant : {var}")
                        relevant_columns.append(var)

            elif pd.api.types.is_object_dtype(df[var]):
                chi2, p, dof, expected, contingency_table = chi_square(df[var], df[target_column])
                output[var] = {"chi2": chi2, "p": p, "dof": dof, "expected": expected, "contingency_table": contingency_table}
                if p <= 0.05:
                    print(f"Relevant : {var}")
                    relevant_columns.append(var)
            else:
                raise NotImplementedError
        except Exception as e:
            print(f"Error processing {var}: {e}")
            continue

    return output, relevant_columns

def statistical_tests_step2(df, target_column, analysis_columns, significance_level=0.05, abs_corr_threshold=0.5):
    """
    Perform statistical tests to identify relevant variables influencing a CONTINUOUS NUMERICAL target variable.
    
    This function is specifically designed for regression tasks where the target is continuous.
    
    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame containing the data.
    target_column : str
        The name of the continuous numerical target column.
    analysis_columns : list of str
        The list of columns to analyze.
    significance_level : float, optional
        The significance level for hypothesis testing (default: 0.05).
    
    Returns
    -------
    output : dict
        A dictionary containing the results of the statistical tests for each variable.
    relevant_columns : list of str
        A list of columns that are statistically significant.
    
    Notes
    -----
    For numerical features:
        - Pearson correlation: Tests linear relationship
        - Spearman correlation: Tests monotonic relationship
        - A feature is considered significant if BOTH tests have p-value <= significance_level
    
    For categorical features:
        - Kruskal-Wallis H-test: Non-parametric test for comparing groups
        - Tests if the target distribution differs across categories
        - A feature is considered significant if p-value <= significance_level
    """
    output = dict()
    relevant_columns = list()
    
    # Verify target is numerical
    if not pd.api.types.is_numeric_dtype(df[target_column]):
        raise ValueError(f"Target column '{target_column}' must be numerical for statistical_tests_step2. "
                        f"Use statistical_tests_step1 for categorical targets.")
    
    target_values = df[target_column].dropna()
    
    for var in analysis_columns:
        try:
            # Skip if the variable is the same as target
            if var == target_column:
                continue
                
            # Get non-null values for this variable
            valid_mask = df[var].notna() & df[target_column].notna()
            
            if valid_mask.sum() < 3:  # Need at least 3 observations
                print(f"Skipping {var}: insufficient non-null observations")
                continue
            
            if pd.api.types.is_numeric_dtype(df[var]):
                # For numerical features: use correlation tests
                var_values = df.loc[valid_mask, var].values
                target_vals = df.loc[valid_mask, target_column].values
                
                # Pearson correlation (linear relationship)
                pearson_corr, pearson_p = pearsonr(var_values, target_vals)
                
                # Spearman correlation (monotonic relationship)
                spearman_corr, spearman_p = spearmanr(var_values, target_vals)
                
                output[var] = {
                    "pearson_corr": pearson_corr,
                    "pearson_p": pearson_p,
                    "spearman_corr": spearman_corr,
                    "spearman_p": spearman_p,
                    "test_type": "correlation"
                }
                
                # Consider significant if both tests are significant and abs correlation exceeds threshold
                if (
                    pearson_p <= significance_level and 
                    spearman_p <= significance_level and 
                    (abs(pearson_corr) >= abs_corr_threshold or abs(spearman_corr) >= abs_corr_threshold)
                ):
                    print(
                        f"Relevant : {var} (Pearson r={pearson_corr:.4f}, Spearman ρ={spearman_corr:.4f}, "
                        f"abs_corr_threshold={abs_corr_threshold})"
                    )
                    relevant_columns.append(var)
            elif pd.api.types.is_object_dtype(df[var]) or df[var].nunique() < 20:
                # For categorical features: use Kruskal-Wallis test
                # Group the target values by categories
                categories = df.loc[valid_mask, var].unique()
                
                if len(categories) < 2:
                    print(f"Skipping {var}: only one category")
                    continue
                
                # Create groups for each category
                groups = []
                for cat in categories:
                    cat_mask = valid_mask & (df[var] == cat)
                    if cat_mask.sum() > 0:
                        groups.append(df.loc[cat_mask, target_column].values)
                
                if len(groups) < 2:
                    print(f"Skipping {var}: insufficient groups")
                    continue
                
                # Kruskal-Wallis H-test (non-parametric ANOVA)
                h_stat, p_value = kruskal(*groups)
                
                output[var] = {
                    "h_stat": h_stat,
                    "p_value": p_value,
                    "n_categories": len(categories),
                    "test_type": "kruskal"
                }
                
                if p_value <= significance_level:
                    print(f"Relevant : {var} (H={h_stat:.4f}, p={p_value:.4e})")
                    relevant_columns.append(var)
            
            else:
                print(f"Skipping {var}: unknown data type")
                
        except Exception as e:
            print(f"Error processing {var}: {e}")
            continue
    
    return output, relevant_columns
