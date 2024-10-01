import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.ensemble import RandomForestRegressor
import graphviz
from sklearn.model_selection import train_test_split, GridSearchCV
from general import Logger
import io
from sklearn.preprocessing import minmax_scale, LabelEncoder
import warnings


# Target variable name is assigned here and used in the code globally
target_col = "charges"
# unnacessary columns should be included here to get eliminated before starting the process
unnecessary_cols = []

def load_data(file_path : str, logger : Logger = None) -> pd.DataFrame:
    # opening the file and show 5 first rows
    logger.log("#################### First 5 rows of original data #######################")
    # open csv file and load it into a dataframe
    data = pd.read_csv(file_path)
    # eliminate unnecessary columns
    data = data.drop(columns=unnecessary_cols)
    # return the first 5 rows of the dataset
    logger.log(data.head())
    return data

def describe_data(data : pd.DataFrame, logger : Logger = None, suffix : str = "original"):
    # extract some descriptive analysis
    logger.log(f"\n#################### Describe {suffix} data specifications ###############")

    # show the rows and columns count plus columns data types
    info = io.StringIO()
    data.info(buf=info)
    logger.log(info.getvalue())

    # show statistic of the columns one by one
    describe = data.describe()

    # since median and mode are not included in describe method outputs, they are added manually to the output for numeric columns
    median = data.median(axis="index", numeric_only=True)
    numeric_mode = data.mode(numeric_only=True)
    for col in describe:
        describe.at["median",col] = median[col]
        describe.at["mode",col] = numeric_mode.loc[0,col]

    logger.log(describe)
    
    # for categorical columns only mode measure is available
    categorical_mode = data.mode().drop(columns=describe.columns)
    if not categorical_mode.empty:
        logger.log(f"\nModes of categorical variables are:\n{categorical_mode}")
        for col in categorical_mode:
            logger.log(f"\n{data.value_counts([col])}")

    logger.log("##########################################################################\n")

def handle_missing_values(data : pd.DataFrame, logger : Logger = None) -> pd.DataFrame:
    # examining data to discover missing values
    logger.log(f"Dataset has these missing values:\n{data.isna().sum()}\n")
    # Replace missing values via bfill method
    data = data.bfill()
    logger.log(f"Missing values are replaced by next valid value of that column using bfill method.\n")
    return data

def handle_duplicate_values(data : pd.DataFrame, logger : Logger = None) -> pd.DataFrame:
    # explorirng data to find duplicated rows
    logger.log(f"Duplicate rows are:\n{data.loc[data.duplicated()]}")
    logger.log(f"Duplicate rows count: {data.duplicated().sum()}")
    # eliminate duplicate rows from data
    data = data.drop_duplicates()
    # summary about data after removing duplicates
    logger.log(f"After removing duplicate rows the dataset has {data.shape[0]} rows.\n")
    return data

def handle_outlier_values(data : pd.DataFrame, output_folder : str = "output", logger : Logger = None) -> pd.DataFrame:
    # Distinguish and remove the outliers based on IQR analysis
    describe = data.describe()
    try:
        describe = describe.drop(columns=target_col)
    except:
        pass
    # Box plot to show outliers
    plot_box(data=data, output_folder=output_folder, logger=logger)

    # Detecting outliers with IQR method
    logger.log("Outliers based on IQR method and skewness before handling them:\n")

    outliers_index = {}
    for col in describe.columns:
        # acquiring q1 and q3 to establish allowed area
        q1 = describe.loc["25%", col]
        q3 = describe.loc["75%", col]
        iqr = q3 - q1
        # every value outside of allowed area is distiguished as an outlier
        outliers = data[col].loc[(data[col] > q3 + 1.5*iqr) | (data[col] < q1 - 1.5*iqr)]
        logger.log(f"Outliers of Column {col} count: {outliers.count()}  ---  skewness: {data[col].skew()}")
        # outlier index of each column
        if not outliers.empty:
            outliers_index[col] = outliers.index

    # Replace outliers with mode of that columns
    for col in outliers_index.keys():
        data.loc[outliers_index[col],col] = data.mode(numeric_only=True).loc[0,col]
    
    # Detecting outliers with IQR method after handling them
    logger.log("\nOutliers based on IQR method and skewness after handling them:\n")
    
    outliers_index = {}
    for col in describe.columns:
        # acquiring q1 and q3 to establish allowed area
        q1 = describe.loc["25%", col]
        q3 = describe.loc["75%", col]
        iqr = q3 - q1
        # every value outside of allowed area is distiguished as an outlier
        outliers = data[col].loc[(data[col] > q3 + 1.5*iqr) | (data[col] < q1 - 1.5*iqr)]
        logger.log(f"Outliers of Column {col} count: {outliers.count()}  ---  skewness: {data[col].skew()}")
        # outlier index of each column
        if not outliers.empty:
            outliers_index[col] = outliers.index

    return data


def encode_data(data : pd.DataFrame, logger : Logger) -> pd.DataFrame:
    # Encode categorical variables to numeric
    # Replace encoded variable using Label Encoding
    logger.log(f"\nBelow columns need encoding:")

    le = LabelEncoder()
    # for each columns in the dataset
    for col in data.columns:
        # if the column is categorical
        if data[col].dtype == "object":
            # encode the data into numeric
            data[col] = le.fit_transform(data[col])
            #unique values of encoded column
            logger.log(f"Values for {col} are {le.classes_}")

    return data


def scale_data(data : pd.DataFrame, logger : Logger = None) -> pd.DataFrame:
    # Scale features values
    # Scale all columns except target column
    scaling_cols = list(data.drop(target_col,axis="columns").columns) 

    # Extract columns which need Scaling
    need_scaling = data[scaling_cols]

    # Scale data
    ndarr = minmax_scale(need_scaling, axis=0) #axis=0 means column-wise

    scaled_data = pd.DataFrame()
    # Build dataframe based on the ndarray and original data
    scaled_data.index = data.index
    for col in data.columns:
        # all features values come from ndarr and the target values come from original data
        if col in scaling_cols:
            scaled_data[col] = ndarr[:,scaling_cols.index(col)]
        else:
            scaled_data[col] = data.loc[:,col]

    # summary about data after encoding and scaling
    logger.log(f"\nAfter encoding and Scaling dataset is:")
    logger.log(scaled_data)

    return scaled_data

def plot_correlation(data : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # Extract correlatoin between features themselves and also with label plus plot their heatmap
    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)
    correlations = data.corr()
    logger.log(f"correlations of features:\n{correlations}")
    sns.heatmap(correlations,annot=True)
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/correlations{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/correlations{('_' if suffix else '') + suffix}.png file saved ###################\n")
   

def plot_pair(data : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # show data exploration by plotting relationship between every two columns
    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)
    # data Exploration
    sns.pairplot(data)
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/data_exploration{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/data_exploration{('_' if suffix else '') + suffix}.png file saved ###################\n")


def plot_box(data : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # Boxplots of input of dataset (features) to show outliers
    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)

    # Distinguish and remove the outliers based on IQR analysis
    describe = data.describe()
    try:
        describe = describe.drop(columns=target_col)
    except:
        pass

    row_count = 1
    col_count = len(describe.columns)

    i = 1
    for col in describe.columns:
        # Creating inputs subplots
        plt.subplot(row_count, col_count, i)
        sns.boxplot(data[col])
        plt.title(f"{col} (input)")
        i += 1

    # setup the layout to fit in the figure
    plt.tight_layout(pad=1, h_pad=0.5, w_pad=0.5) 
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/input_outliers{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### input_outliers{('_' if suffix else '') + suffix}.png file saved ######################\n")


def plot_hist(data : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # histogram of input and output of dataset (features and label)
    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)

    row_count = 3
    col_count = 2

    i = 1
    for col in data.drop(columns=target_col).columns:
        # Creating inputs subplots
        plt.subplot(row_count, col_count, i)
        sns.histplot(data[col], kde=True)
        plt.title(f"{col} (feature)")
        i += 1

    # setup the layout to fit in the figure
    plt.tight_layout(pad=1, h_pad=0.5, w_pad=0.5) 
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/input_description{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/input_description{('_' if suffix else '') + suffix}.png file saved ###################\n")


def plot_line(data : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # histogram of input and output of dataset (features and label)
    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)

    row_count = 3
    col_count = 2

    i = 1
    for col in data.drop(columns=target_col).columns:
        # Creating inputs subplots
        plt.subplot(row_count, col_count, i)
        sns.lineplot(data, x=data[col], y=data[target_col])
        plt.title(f"{col} (feature)")
        i += 1

    # setup the layout to fit in the figure
    plt.tight_layout(pad=1, h_pad=0.5, w_pad=0.5) 
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/regression_lines{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/regression_lines{('_' if suffix else '') + suffix}.png file saved ###################\n")


def extract_train_test(data : pd.DataFrame, logger : Logger = None) -> list:
    # Seperate features and labels
    X = data.drop(columns=target_col)
    y = data.loc[:, target_col]

    # split train and test datasets
    train_test_sets = train_test_split(X, y, test_size=0.2, random_state=42)

    return train_test_sets

def hyperparameter_tuning_regression(data : pd.DataFrame, train_test_sets : list, estimator, param : dict, output_folder : str = "output", logger : Logger = None) -> dict: 
    logger.log("############ Hyperparameter tuning to optimized model parameters ############\n")
    # grid search to find best classifier parameters (Hyperparameter tuning)
    
    # Extract train and test sets
    X_train, X_test, y_train, y_test = train_test_sets
    # Some parameters do not match with each other. This code avoids unnecessary warnings
    warnings.filterwarnings("ignore")

    # Create and train gridsearch
    grid = GridSearchCV(estimator, param_grid=param, refit=True, n_jobs=-1, cv=10, scoring="r2")
    grid.fit(X_train, y_train)

    # Find the best paramters
    grid_best_params = grid.best_params_
    grid_best_score = grid.best_score_

    logger.log(f"Best Parameters: {grid_best_params}")
    logger.log(f"Best Score: {grid_best_score}")

    return grid_best_params


def linear_regression(data : pd.DataFrame, train_test_sets : list, grid_best_params : dict = {}, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    logger.log(f"################ regression by linear regression {"(After Optimization)" if grid_best_params else "(Before Optimization)"} #################\n")
    # Build a regression model to predict the test set

    # grid_best_params is not empty when we are train model after the optimization
    if grid_best_params:
        reg = LinearRegression(copy_X=grid_best_params["copy_X"], fit_intercept=grid_best_params["fit_intercept"], n_jobs=grid_best_params["n_jobs"], positive=grid_best_params["positive"])
    else:
        reg = LinearRegression()

    # Extract train and test datasets
    X_train, X_test, y_train, y_test = train_test_sets

    # train the model
    reg.fit(X_train, y_train)

    # Predict test dataset via model
    y_pred = reg.predict(X_test)

    # Evaluate the model
    evaluate_linear_regression(reg=reg, data=data, X_train=X_train, y_train=y_train, y_test=y_test, y_pred=y_pred, suffix="After Optimization" if grid_best_params else "Before Optimization", output_folder=output_folder, logger=logger)


def evaluate_linear_regression(reg : LinearRegression, data : pd.DataFrame, X_train : pd.DataFrame, y_train : pd.DataFrame, y_test : pd.DataFrame, y_pred : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # Evaluating model
    logger.log(f"################ Evaluating the model{"(" + suffix + ")" if suffix else ''} #################")

    # Extract mean_absolute_error
    mea = mean_absolute_error(y_test, y_pred)
    logger.log(f"mean absolute error: {mea}\n")

    # Extract mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    logger.log(f"mean squared error: {mse}\n")

    # Extract r2_score
    r2score = r2_score(y_test, y_pred)
    logger.log(f"r2_score: {r2score}\n")

    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)
    # Extract model coefficients and plot their barh
    # Creating a dataframe based on the  coefficients
    coef = pd.DataFrame(reg.coef_,index=reg.feature_names_in_, columns=["coefficients"])
    coef = coef.sort_values("coefficients",axis="index")
    logger.log(f"Coefficients:\n{coef}")
    plt.clf()
    # Create a bar plot to compare feature importances visually
    plt.barh(coef.index, coef.iloc[:,0])
    plt.xlabel("coefficients")
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/coefficients{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/coefficients{('_' if suffix else '') + suffix}.png file saved ###################\n")


def ridge_regression(data : pd.DataFrame, train_test_sets : list, grid_best_params : dict = {}, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    logger.log(f"################ regression by ridge regression {"(After Optimization)" if grid_best_params else "(Before Optimization)"} #################\n")
    # Build a regression model to predict the test set

    # grid_best_params is not empty when we are train model after the optimization
    if grid_best_params:
        reg = Ridge(alpha = grid_best_params["alpha"], copy_X=grid_best_params["copy_X"], fit_intercept=grid_best_params["fit_intercept"], positive=grid_best_params["positive"], solver = grid_best_params["solver"], random_state = grid_best_params["random_state"])
    else:
        reg = Ridge()

    # Extract train and test datasets
    X_train, X_test, y_train, y_test = train_test_sets

    # train the model
    reg.fit(X_train, y_train)

    # Predict test dataset via model
    y_pred = reg.predict(X_test)

    # Evaluate the model
    evaluate_ridge_regression(reg=reg, data=data, X_train=X_train, y_train=y_train, y_test=y_test, y_pred=y_pred, suffix="After Optimization" if grid_best_params else "Before Optimization", output_folder=output_folder, logger=logger)


def evaluate_ridge_regression(reg : ridge_regression, data : pd.DataFrame, X_train : pd.DataFrame, y_train : pd.DataFrame, y_test : pd.DataFrame, y_pred : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # Evaluating model
    logger.log(f"################ Evaluating the model{"(" + suffix + ")" if suffix else ''} #################")

    # Extract mean_absolute_error
    mea = mean_absolute_error(y_test, y_pred)
    logger.log(f"mean absolute error: {mea}\n")

    # Extract mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    logger.log(f"mean squared error: {mse}\n")

    # Extract r2_score
    r2score = r2_score(y_test, y_pred)
    logger.log(f"r2_score: {r2score}\n")

    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)
    # Extract model coefficients and plot their barh
    # Creating a dataframe based on the  coefficients
    coef = pd.DataFrame(reg.coef_,index=reg.feature_names_in_, columns=["coefficients"])
    coef = coef.sort_values("coefficients",axis="index")
    logger.log(f"Coefficients:\n{coef}")
    plt.clf()
    # Create a bar plot to compare feature importances visually
    plt.barh(coef.index, coef.iloc[:,0])
    plt.xlabel("coefficients")
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/coefficients{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/coefficients{('_' if suffix else '') + suffix}.png file saved ###################\n")


def decisiontree_regression(data : pd.DataFrame, train_test_sets : list, grid_best_params : dict = {}, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    logger.log(f"################ regression by decision tree  {"(After Optimization)" if grid_best_params else "(Before Optimization)"} #################\n")
    # Build a regression model to predict the test set

    # grid_best_params is not empty when we are train model after the optimization
    if grid_best_params:
        reg = DecisionTreeRegressor(max_depth=grid_best_params["max_depth"], min_samples_leaf=grid_best_params["min_samples_leaf"], min_samples_split=grid_best_params["min_samples_split"], random_state = grid_best_params["random_state"], max_features = grid_best_params["max_features"])
    else:
        reg = DecisionTreeRegressor()

    # Extract train and test datasets
    X_train, X_test, y_train, y_test = train_test_sets

    # train the model
    reg.fit(X_train, y_train)

    # Predict test dataset via model
    y_pred = reg.predict(X_test)

    # # Exporting decesion tree to png file
    # export_tree(reg=reg, feature_names_in_=reg.feature_names_in_, suffix=suffix, output_folder=output_folder, logger=logger)

    # Evaluate the model
    evaluate_decisiontree_regression(reg=reg, data=data, X_train=X_train, y_train=y_train, y_test=y_test, y_pred=y_pred, suffix="After Optimization" if grid_best_params else "Before Optimization", output_folder=output_folder, logger=logger)


def evaluate_decisiontree_regression(reg : DecisionTreeRegressor, data : pd.DataFrame, X_train : pd.DataFrame, y_train : pd.DataFrame, y_test : pd.DataFrame, y_pred : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # Evaluating model
    logger.log(f"################ Evaluating the model{"(" + suffix + ")" if suffix else ''} #################")

    # Extract mean_absolute_error
    mea = mean_absolute_error(y_test, y_pred)
    logger.log(f"mean absolute error: {mea}\n")

    # Extract mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    logger.log(f"mean squared error: {mse}\n")

    # Extract r2_score
    r2score = r2_score(y_test, y_pred)
    logger.log(f"r2_score: {r2score}\n")

    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)
    # Extract model feature_importances and plot their barh
    # Creating a dataframe based on the feature importances
    feature_importances = pd.DataFrame(reg.feature_importances_,index=reg.feature_names_in_, columns=["feature_importances"])
    feature_importances = feature_importances.sort_values("feature_importances",axis="index")
    logger.log(f"Importance of features:\n{feature_importances}")
    plt.clf()
    # Create a bar plot to compare feature importances visually
    plt.barh(feature_importances.index, feature_importances.iloc[:,0])
    plt.xlabel("feature_importances")
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/feature_importances{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/feature_importances{('_' if suffix else '') + suffix}.png file saved ###################\n")


def randomforest_regression(data : pd.DataFrame, train_test_sets : list, grid_best_params : dict = {}, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    logger.log(f"################ regression by random forest {"(After Optimization)" if grid_best_params else "(Before Optimization)"} #################\n")
    # Build a regression model to predict the test set

    # grid_best_params is not empty when we are train model after the optimization
    if grid_best_params:
        reg = RandomForestRegressor(max_depth=grid_best_params["max_depth"], min_samples_leaf=grid_best_params["min_samples_leaf"], min_samples_split=grid_best_params["min_samples_split"], random_state = grid_best_params["random_state"], max_features = grid_best_params["max_features"])
    else:
        reg = RandomForestRegressor()

    # Extract train and test datasets
    X_train, X_test, y_train, y_test = train_test_sets

    # train the models
    reg.fit(X_train, y_train)

    # Predict test dataset via model
    y_pred = reg.predict(X_test)

    # # Exporting decesion trees of the random forest to png file
    # est_count = 1
    # for estimator in reg.estimators_:
    #     export_tree(reg=estimator, feature_names_in_=reg.feature_names_in_, suffix=suffix + "_" + str(est_count), output_folder=output_folder, logger=logger)
    #     est_count += 1
    #     if est_count > 5:
    #         break

    # Evaluate the model
    evaluate_randomforest_regression(reg=reg, data=data, X_train=X_train, y_train=y_train, y_test=y_test, y_pred=y_pred, suffix="After Optimization" if grid_best_params else "Before Optimization", output_folder=output_folder, logger=logger)


def evaluate_randomforest_regression(reg : RandomForestRegressor, data : pd.DataFrame, X_train : pd.DataFrame, y_train : pd.DataFrame, y_test : pd.DataFrame, y_pred : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # Evaluating model
    logger.log(f"################ Evaluating the model{"(" + suffix + ")" if suffix else ''} #################")

    # Extract mean_absolute_error
    mea = mean_absolute_error(y_test, y_pred)
    logger.log(f"mean absolute error: {mea}\n")

    # Extract mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    logger.log(f"mean squared error: {mse}\n")

    # Extract r2_score
    r2score = r2_score(y_test, y_pred)
    logger.log(f"r2_score: {r2score}\n")

    # set the figure resolution and dpi
    fig = plt.figure(figsize=(16, 9), dpi=600)
    # Extract model feature_importances and plot their barh
    # Creating a dataframe based on the feature importances
    feature_importances = pd.DataFrame(reg.feature_importances_,index=reg.feature_names_in_, columns=["feature_importances"])
    feature_importances = feature_importances.sort_values("feature_importances",axis="index")
    logger.log(f"Importance of features:\n{feature_importances}")
    plt.clf()
    # Create a bar plot to compare feature importances visually
    plt.barh(feature_importances.index, feature_importances.iloc[:,0])
    plt.xlabel("feature_importances")
    # Save the file with proper dpi
    plt.savefig(fname=f"{output_folder}/feature_importances{('_' if suffix else '') + suffix}.png", format="png", dpi = fig.dpi)
    logger.log(f"##################### {output_folder}/feature_importances{('_' if suffix else '') + suffix}.png file saved ###################\n")

def export_tree(reg : DecisionTreeRegressor, feature_names_in_ : np.ndarray, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # export decision tree into dot file notaion
    dot_data = export_graphviz(reg, out_file =None ,feature_names=feature_names_in_, impurity=True, filled=True, rounded=True)
    # create png file based on dot data
    graph = graphviz.Source(dot_data, format="png") 
    # render and save png file
    graph.render(filename=f"{output_folder}/decision_tree{('_' if suffix else '') + suffix}", view=False)
    logger.log(f"##################### {output_folder}/decision_tree{('_' if suffix else '') + suffix}.png file saved #################################\n")



def xgboost_regression(data : pd.DataFrame, train_test_sets : list, grid_best_params : dict = {}, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    logger.log(f"################ regression by xgboost {"(After Optimization)" if grid_best_params else "(Before Optimization)"} #################\n")
    # Build a regression model to predict the test set

    # grid_best_params is not empty when we are train model after the optimization
    if grid_best_params:
        reg = XGBRegressor(max_depth=grid_best_params["max_depth"], learning_rate=grid_best_params["learning_rate"], subsample=grid_best_params["subsample"], min_child_weight=grid_best_params["min_child_weight"], reg_alpha=grid_best_params["reg_alpha"], reg_lambda=grid_best_params["reg_lambda"], gamma=grid_best_params["gamma"])
    else:
        reg = XGBRegressor()

    # Extract train and test datasets
    X_train, X_test, y_train, y_test = train_test_sets

    # train the model
    reg.fit(X_train, y_train)

    # Predict test dataset via model
    y_pred = reg.predict(X_test)

    # Evaluate the model
    evaluate_xgboost_regression(reg=reg, data=data, X_train=X_train, y_train=y_train, y_test=y_test, y_pred=y_pred, suffix="After Optimization" if grid_best_params else "Before Optimization", output_folder=output_folder, logger=logger)


def evaluate_xgboost_regression(reg : LinearRegression, data : pd.DataFrame, X_train : pd.DataFrame, y_train : pd.DataFrame, y_test : pd.DataFrame, y_pred : pd.DataFrame, suffix : str = "", output_folder : str = "output", logger : Logger = None):
    # Evaluating model
    logger.log(f"################ Evaluating the model{"(" + suffix + ")" if suffix else ''} #################")

    # Extract mean_absolute_error
    mea = mean_absolute_error(y_test, y_pred)
    logger.log(f"mean absolute error: {mea}\n")

    # Extract mean_squared_error
    mse = mean_squared_error(y_test, y_pred)
    logger.log(f"mean squared error: {mse}\n")

    # Extract r2_score
    r2score = r2_score(y_test, y_pred)
    logger.log(f"r2_score: {r2score}\n")


def main() -> int:
    # Folder path to save outputs
    output_folder = "output"
    logger = Logger()

    # Load dataset 
    # SAMAN TEYMOURI PAMLP Report
    df = load_data("input/insurance.csv", logger=logger)

    # describe dataset characteristics
    describe_data(df, logger=logger)

    # Show data exploration
    plot_pair(df, "before_data_cleaning", output_folder=output_folder,logger=logger)

    # Plot before data cleaning
    plot_hist(df, "before_data_cleaning", output_folder, logger=logger)

    # Show line plot of features and target
    plot_line(df,output_folder=output_folder,logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("loading and describing the data")

    logger.log("#################### Data cleaning and dataset changes ###################")
    # Data Cleaning Steps

    # Handle missing values
    df = handle_missing_values(df, logger=logger)

    # Find duplicates and remove them
    df = handle_duplicate_values(df, logger=logger)

    # Find outliers and remove them
    df = handle_outlier_values(df, output_folder, logger=logger)

    # Encode data
    df = encode_data(df, logger=logger)

    # describe dataset characteristics
    describe_data(df, logger=logger, suffix="cleaned (except scaling step)")

    # Show correlation between features with each other and with the target variable
    plot_correlation(df, output_folder=output_folder,logger=logger)

    # Show data exploration after data cleaning
    plot_pair(df, "after_data_cleaning", output_folder=output_folder,logger=logger)

    # Plot histograms after data cleaning
    plot_hist(df, "after_data_cleaning", output_folder, logger=logger)
    
    # Scale data
    df = scale_data(df, logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("cleaning the data")

    # Extracting train and test sets
    train_test_sets = extract_train_test(df, logger=logger)


    # ================== Modeling via linear regression ================
    # Perform classification using linear regression (before optimization)
    logger.log("#################### Modeling via linear regression ###################")

    linear_regression(df, train_test_sets=train_test_sets, output_folder=output_folder + "/linearregression_model/before_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("regression via linear regression before optimization")

    # Optimizing the parameters of the linear regression model using hyperparameter tuning
    grid_best_params = hyperparameter_tuning_regression(data=df, train_test_sets=train_test_sets, output_folder=output_folder + "/linearregression_model/before_optimization", estimator=LinearRegression(), param = 
        {"copy_X": [True,False],
         "fit_intercept": [True,False], 
         "n_jobs": [1,5,10,15,None],
         "positive": [False]
    }, logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("hyperparameter tuning of linear regression")

    # Perform classification using linear regression (after optimization)
    linear_regression(df, train_test_sets=train_test_sets, grid_best_params=grid_best_params, output_folder=output_folder + "/linearregression_model/after_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("regression via linear regression after optimization")


    # ================== Modeling via ridge regression ================
    # Perform classification using ridge regression (before optimization)
    logger.log("#################### Modeling via ridge regression ###################")

    ridge_regression(df, train_test_sets=train_test_sets, output_folder=output_folder + "/ridgeregression_model/before_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("regression via ridge regression before optimization")

    # Optimizing the parameters of the ridge regression model using hyperparameter tuning
    grid_best_params = hyperparameter_tuning_regression(data=df, train_test_sets=train_test_sets, output_folder=output_folder + "/ridgeregression_model/before_optimization", estimator=Ridge(), param = {
         "alpha": [0.0001, 0.001, 0.01, 1, 10, 100 , 1000],
         "copy_X": [True,False],
         "fit_intercept": [True,False], 
         "positive": [False],
         "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
         "random_state": [0, 1, 10, 20, 42, None]
    }, logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("hyperparameter tuning of ridge regression")

    # Perform classification using ridge regression (after optimization)
    ridge_regression(df, train_test_sets=train_test_sets, grid_best_params=grid_best_params, output_folder=output_folder + "/ridgeregression_model/after_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("regression via ridge regression after optimization")
    

    # ================== Modeling via decision tree ================
    # Perform classification using decision tree (before optimization)
    logger.log("#################### Modeling via decision tree ###################")

    decisiontree_regression(df, train_test_sets=train_test_sets, output_folder=output_folder + "/decisiontree_model/before_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("regression via decision tree before optimization")

    # Optimizing the parameters of the decision tree model using hyperparameter tuning
    grid_best_params = hyperparameter_tuning_regression(data=df, train_test_sets=train_test_sets, estimator=DecisionTreeRegressor(), param = {
        "max_depth": [10,50,100,200,None],
        "min_samples_leaf": [1, 2],
        "min_samples_split": [2, 3, 4],
        "random_state": [0, 1, 10, 20, 42, None],
        "max_features" : ["sqrt", "log2", None]
    }, logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("hyperparameter tuning of decision tree")

    # Perform classification using decision tree (after optimization)
    decisiontree_regression(df, train_test_sets=train_test_sets, grid_best_params=grid_best_params, output_folder=output_folder + "/decisiontree_model/after_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("regression via decision tree after optimization")

    
    # ================== Modeling via random forest ================
    # Perform classification using random forest (before optimization)
    logger.log("#################### Modeling via random forest ###################")

    randomforest_regression(df, train_test_sets=train_test_sets, output_folder=output_folder + "/randomforest_model/before_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("regression via random forest before optimization")

    # Optimizing the parameters of the random forest model using hyperparameter tuning
    grid_best_params = hyperparameter_tuning_regression(data=df, train_test_sets=train_test_sets, estimator=RandomForestRegressor(), param = {
        "max_depth": [10,50,100,200,None],
        "min_samples_leaf": [1, 2],
        "min_samples_split": [2, 3, 4],
        "random_state": [0, 1, 10, 20, 42, None],
        "max_features" : ["sqrt", "log2"]
    }, logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("hyperparameter tuning of random forest")

    # Perform classification using random forest (after optimization)
    randomforest_regression(df, train_test_sets=train_test_sets, grid_best_params=grid_best_params, output_folder=output_folder + "/randomforest_model/after_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("regression via random forest after optimization")

    # ================== Modeling via xgboost regression ================
    # Perform classification using xgboost regression (before optimization)
    logger.log("#################### Modeling via xgboost regression ###################")

    xgboost_regression(df, train_test_sets=train_test_sets, output_folder=output_folder + "/xgboost_model/before_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("regression via xgboost regression before optimization")

    # Optimizing the parameters of the xgboost regression model using hyperparameter tuning
    grid_best_params = hyperparameter_tuning_regression(data=df, train_test_sets=train_test_sets, output_folder=output_folder + "/xgboost_model/before_optimization", estimator=XGBRegressor(), param = 
        {"max_depth": [3, 6, 9], 
         "learning_rate": [0.03, 0.3, 0.9],
         "subsample": [0.5, 1],
         "min_child_weight": [1, 10, 15],
         "reg_alpha": [0, 0.5, 1],
         "reg_lambda": [0, 0.5, 1],
         "gamma": [0, 1, 10],
    }, logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("hyperparameter tuning of xgboost regression")

    # Perform classification using xgboost regression (after optimization)
    xgboost_regression(df, train_test_sets=train_test_sets, grid_best_params=grid_best_params, output_folder=output_folder + "/xgboost_model/after_optimization",logger=logger)

    # Print elapsed time for the below job
    logger.print_elapsed_time("regression via xgboost regression after optimization")

    # Save log file
    logger.save_file(f"{output_folder}/output_log.txt")

    print("\n##########################################################################\n" +
            f"Please check <{output_folder}> folder in the program path for outputs.\n" +
            "##########################################################################\n")

    return 0

if __name__ == "__main__":
    main()
