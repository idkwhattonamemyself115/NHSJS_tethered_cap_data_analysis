import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# File paths and directories
FILE_PATH = 'data/raw_survey_data.xlsx'
OUTPUT_DIR = 'output'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data loading
df_raw = pd.read_excel(FILE_PATH)
df = df_raw.copy()

print('Original shape:', df.shape)

# Helper functions
def to_num(series):
    return pd.to_numeric(series, errors='coerce')

def get_col_starting_with(prefix):
    for c in df.columns:
        if str(c).startswith(prefix):
            return c
    return None

def calc_cronbach_alpha(df_items):
    items = df_items.dropna()
    k = items.shape[1]
    if k < 2:
        return None
    variances = items.var(axis=0, ddof=1)
    total_var = items.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - (variances.sum() / total_var))

def calc_vif(df_data, xvars):
    tmp = df_data[xvars].dropna()
    vif_list = []
    for i in range(tmp.shape[1]):
        vif_val = variance_inflation_factor(tmp.values, i)
        vif_list.append({'variable': xvars[i], 'vif': vif_val})
    return pd.DataFrame(vif_list)

def run_tukey_hsd(df_data, group_col, outcome_col, alpha=0.05):
    tmp = df_data[[group_col, outcome_col]].dropna()
    res = pairwise_tukeyhsd(endog=tmp[outcome_col], groups=tmp[group_col], alpha=alpha)
    summary_data = res._results_table.data
    return pd.DataFrame(summary_data[1:], columns=summary_data[0])

def run_ttest(data, group_col, outcome_col, group1=1, group0=0):
    tmp = data[[group_col, outcome_col]].dropna()
    g1 = tmp.loc[tmp[group_col] == group1, outcome_col]
    g0 = tmp.loc[tmp[group_col] == group0, outcome_col]
    if len(g1) < 2 or len(g0) < 2:
        return None
    tstat, pval = stats.ttest_ind(g1, g0, equal_var=False, nan_policy='omit')
    return {
        'group_col': group_col,
        'outcome_col': outcome_col,
        'mean_group1': g1.mean(),
        'mean_group0': g0.mean(),
        'n_group1': len(g1),
        'n_group0': len(g0),
        't_stat': tstat,
        'p_value': pval
    }

def run_anova(data, group_col, outcome_col):
    tmp = data[[group_col, outcome_col]].dropna()
    groups = [g[outcome_col].values for _, g in tmp.groupby(group_col)]
    if len(groups) < 2 or any(len(g) < 2 for g in groups):
        return None
    fstat, pval = stats.f_oneway(*groups)
    return {
        'group_col': group_col,
        'outcome_col': outcome_col,
        'num_groups': len(groups),
        'f_stat': fstat,
        'p_value': pval
    }

def run_spearman(data, x, y):
    tmp = data[[x, y]].dropna()
    if len(tmp) < 5:
        return None
    rho, pval = stats.spearmanr(tmp[x], tmp[y])
    return {
        'var1': x,
        'var2': y,
        'n': len(tmp),
        'spearman_rho': rho,
        'p_value': pval
    }

def fit_ols(data, y, xvars):
    tmp = data[[y] + xvars].dropna()
    if len(tmp) < 20:
        return None
    X = sm.add_constant(tmp[xvars], has_constant='add')
    model = sm.OLS(tmp[y], X).fit(cov_type='HC3')
    return model

def fit_logit(data, y, xvars):
    tmp = data[[y] + xvars].dropna()
    if len(tmp) < 20:
        return None
    X = sm.add_constant(tmp[xvars], has_constant='add')
    model = sm.Logit(tmp[y], X).fit(disp=False)
    return model

def fit_ordered_logit(data, y, xvars):
    tmp = data[[y] + xvars].dropna()
    if len(tmp) < 20:
        return None
    endog = tmp[y].astype(int)
    exog = tmp[xvars]
    model = OrderedModel(endog, exog, distr='logit')
    res = model.fit(method='bfgs', disp=False)
    return res

def model_to_table(result, model_name):
    if result is None:
        return pd.DataFrame()
    return pd.DataFrame({
        'term': result.params.index,
        'coef': result.params.values,
        'std_err': result.bse.values,
        'p_value': result.pvalues.values,
        'model': model_name
    })

# Survey column mapping
q1_age = get_col_starting_with('1、')
q2_gender = get_col_starting_with('2、')
q3_edu = get_col_starting_with('3、')
q4_income = get_col_starting_with('4、')
q5_employ = get_col_starting_with('5、')
q6_env = get_col_starting_with('6、')
q7_country = get_col_starting_with('7、')
q8_freq = get_col_starting_with('8、')

q9_plastic = '9、您最常购买的瓶装水包装是？（多选题）(塑料水瓶)'
q9_aluminum = '9、(铝罐)'
q9_glass = '9、(玻璃瓶)'
q9_reusable = '9、(可循环利用水瓶)'

q10_factor = get_col_starting_with('10、')
q11_price_sens = get_col_starting_with('11、')
q12_more_exp = get_col_starting_with('12、')
q13_reuse = get_col_starting_with('13、')
q14_cap_handle = get_col_starting_with('14、')
q15_cap_recycle_know = get_col_starting_with('15、')
q16_env_att = get_col_starting_with('16、')
q17_resp = get_col_starting_with('17、')
q18_inconv = get_col_starting_with('18、')
q19_heard = get_col_starting_with('19、')
q20_conven = get_col_starting_with('20、')
q21_stopbuy = get_col_starting_with('21、')
q22_ifall = get_col_starting_with('22、')
q23_recycle_together = get_col_starting_with('23、')
q24_support = get_col_starting_with('24、')

# Variable transformations and coding
df['cap_handling'] = to_num(df[q14_cap_handle])
df['know_caps_not_recycled_raw'] = to_num(df[q15_cap_recycle_know])
df['heard_tethered_cap_raw'] = to_num(df[q19_heard])
df['stop_buy_threshold'] = to_num(df[q21_stopbuy])
df['if_all_tethered'] = to_num(df[q22_ifall])
df['policy_support'] = to_num(df[q24_support])

df['know_caps_not_recycled'] = np.where(df['know_caps_not_recycled_raw'] == 1, 1,
                                 np.where(df['know_caps_not_recycled_raw'] == 2, 0, np.nan))

df['heard_tethered_cap'] = np.where(df['heard_tethered_cap_raw'] == 1, 1,
                             np.where(df['heard_tethered_cap_raw'] == 2, 0, np.nan))

df['price_sensitivity'] = to_num(df[q11_price_sens])
df['env_concern'] = to_num(df[q16_env_att])
df['personal_responsibility'] = to_num(df[q17_resp])
df['accept_inconvenience'] = to_num(df[q18_inconv])
df['tethered_cap_convenience'] = to_num(df[q20_conven])
df['recycle_together_likelihood'] = to_num(df[q23_recycle_together])

df['age_group'] = to_num(df[q1_age])
df['gender_code'] = to_num(df[q2_gender])
df['education_level'] = to_num(df[q3_edu])
df['income_group'] = to_num(df[q4_income])
df['employment_status'] = to_num(df[q5_employ])
df['living_environment'] = to_num(df[q6_env])
df['bottled_water_freq'] = to_num(df[q8_freq])
df['purchase_factor'] = to_num(df[q10_factor])
df['if_more_expensive'] = to_num(df[q12_more_exp])
df['reuse_frequency'] = to_num(df[q13_reuse])

df['country'] = df[q7_country].astype(str).str.strip()

df['female'] = np.where(df['gender_code'] == 1, 1,
                 np.where(df['gender_code'].notna(), 0, np.nan))

for raw_col, new_col in [
    (q9_plastic, 'buy_plastic'),
    (q9_aluminum, 'buy_aluminum'),
    (q9_glass, 'buy_glass'),
    (q9_reusable, 'buy_reusable')
]:
    if raw_col in df.columns:
        s = df[raw_col]
        df[new_col] = np.where(
            s.isna(), 0,
            np.where(s.astype(str).str.strip().isin(['0', '0.0', '', 'nan']), 0, 1)
        )

df['env_index'] = df[['env_concern', 'personal_responsibility', 'accept_inconvenience']].mean(axis=1)

df['support_binary'] = np.where(df['policy_support'] >= 4, 1,
                         np.where(df['policy_support'].isin([1, 2, 3]), 0, np.nan))

df['convenience_x_env'] = df['tethered_cap_convenience'] * df['env_index']

# Descriptive statistics & reliability analysis
desc_vars = [
    'env_concern', 'personal_responsibility', 'accept_inconvenience',
    'tethered_cap_convenience', 'recycle_together_likelihood',
    'policy_support', 'env_index'
]
desc = df[desc_vars].describe().T

alpha_val = calc_cronbach_alpha(df[['env_concern', 'personal_responsibility', 'accept_inconvenience']])
print(f"Cronbach's Alpha (EnvIndex): {alpha_val}")

# Multicollinearity diagnostics
vif_vars = ['tethered_cap_convenience', 'price_sensitivity', 'env_index', 'know_caps_not_recycled', 'heard_tethered_cap']
vif_df = calc_vif(df, vif_vars)

# Bivariate statistical analysis
corr_specs = [
    ('policy_support', 'tethered_cap_convenience'),
    ('policy_support', 'price_sensitivity'),
    ('policy_support', 'env_index'),
    ('recycle_together_likelihood', 'env_index'),
    ('recycle_together_likelihood', 'tethered_cap_convenience'),
]

corr_results = []
for x, y in corr_specs:
    res = run_spearman(df, x, y)
    if res:
        corr_results.append(res)
corr_df = pd.DataFrame(corr_results)

ttest_specs = [
    ('know_caps_not_recycled', 'recycle_together_likelihood'),
    ('know_caps_not_recycled', 'policy_support'),
    ('heard_tethered_cap', 'policy_support'),
    ('female', 'policy_support'),
    ('buy_plastic', 'policy_support'),
]

ttest_results = []
for g, y in ttest_specs:
    res = run_ttest(df, g, y)
    if res:
        ttest_results.append(res)
ttest_df = pd.DataFrame(ttest_results)

anova_specs = [
    ('stop_buy_threshold', 'policy_support'),
    ('if_all_tethered', 'policy_support'),
    ('cap_handling', 'recycle_together_likelihood'),
    ('if_all_tethered', 'tethered_cap_convenience'),
    ('purchase_factor', 'env_index'),
]

anova_results = []
for g, y in anova_specs:
    res = run_anova(df, g, y)
    if res:
        anova_results.append(res)
anova_df = pd.DataFrame(anova_results)

# Post-hoc Tukey HSD test
tukey_df = run_tukey_hsd(df, 'cap_handling', 'recycle_together_likelihood')

# Multivariate regression modeling
controls = [
    'age_group', 'female', 'education_level',
    'bottled_water_freq', 'reuse_frequency', 'buy_plastic'
]

main_x = [
    'tethered_cap_convenience',
    'price_sensitivity',
    'env_index',
    'know_caps_not_recycled',
    'heard_tethered_cap',
    'stop_buy_threshold'
]

m1 = fit_ols(df, 'policy_support', main_x)
m2 = fit_ols(df, 'policy_support', main_x + controls)

x3 = [
    'tethered_cap_convenience',
    'price_sensitivity',
    'env_index',
    'know_caps_not_recycled',
    'heard_tethered_cap'
] + controls
m3 = fit_ols(df, 'recycle_together_likelihood', x3)

x4 = [
    'tethered_cap_convenience',
    'price_sensitivity',
    'env_index',
    'know_caps_not_recycled',
    'heard_tethered_cap'
] + controls
m4 = fit_logit(df, 'support_binary', x4)

x5 = [
    'tethered_cap_convenience',
    'env_index',
    'convenience_x_env',
    'price_sensitivity',
    'know_caps_not_recycled',
    'heard_tethered_cap'
] + controls
m5 = fit_ols(df, 'policy_support', x5)

try:
    ordered_res = fit_ordered_logit(
        df, 'policy_support',
        ['tethered_cap_convenience', 'price_sensitivity', 'env_index',
         'know_caps_not_recycled', 'heard_tethered_cap']
    )
except Exception as e:
    print('Ordered logit failed:', e)
    ordered_res = None

reg_df = pd.concat([
    model_to_table(m1, 'OLS_policy_support_baseline'),
    model_to_table(m2, 'OLS_policy_support_controls'),
    model_to_table(m3, 'OLS_recycle_together'),
    model_to_table(m4, 'Logit_support_binary'),
    model_to_table(m5, 'OLS_policy_support_interaction'),
    model_to_table(ordered_res, 'OrderedLogit_policy_support') if ordered_res is not None else pd.DataFrame()
], ignore_index=True)

# Export results and clean data
clean_cols = [
    'age_group', 'gender_code', 'female', 'education_level', 'income_group',
    'employment_status', 'living_environment', 'country', 'bottled_water_freq',
    'buy_plastic', 'buy_aluminum', 'buy_glass', 'buy_reusable',
    'purchase_factor', 'price_sensitivity', 'if_more_expensive',
    'reuse_frequency', 'cap_handling', 'know_caps_not_recycled',
    'env_concern', 'personal_responsibility', 'accept_inconvenience',
    'heard_tethered_cap', 'tethered_cap_convenience', 'stop_buy_threshold',
    'if_all_tethered', 'recycle_together_likelihood', 'policy_support',
    'env_index', 'support_binary', 'convenience_x_env'
]

clean_path = os.path.join(OUTPUT_DIR, 'cleaned_tethered_cap_data.xlsx')
result_path = os.path.join(OUTPUT_DIR, 'tethered_cap_analysis_results.xlsx')

df[clean_cols].to_excel(clean_path, index=False)

with pd.ExcelWriter(result_path, engine='xlsxwriter') as writer:
    df[clean_cols].to_excel(writer, sheet_name='cleaned_data', index=False)
    desc.to_excel(writer, sheet_name='descriptive_stats')
    vif_df.to_excel(writer, sheet_name='vif_diagnostics', index=False)
    corr_df.to_excel(writer, sheet_name='correlations', index=False)
    ttest_df.to_excel(writer, sheet_name='t_tests', index=False)
    anova_df.to_excel(writer, sheet_name='anova_tests', index=False)
    tukey_df.to_excel(writer, sheet_name='tukey_hsd', index=False)
    reg_df.to_excel(writer, sheet_name='regressions', index=False)

print('Saved cleaned data to:', clean_path)
print('Saved results workbook to:', result_path)