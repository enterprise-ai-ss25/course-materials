# Let's connect everything together and create a training pipeline here
from src.config import TEST_SIZE, DATA_PATH, DATA_RANDOM_STATE, MODEL_RANDOM_STATE
from src.data_loader import load_data
from src.preprocessing import split_data, impute_train_data, impute_test_data, remove_categorical_columns
from src.model import create_forest_regressor, train_model, get_predictions
from src.utils import compute_mae, compute_mape

# create a pipeline function
def run_pipeline():
    # read data
    data = load_data(DATA_PATH)
    X = data.drop("price", axis=1)
    labels = data["price"]
    # split data
    X_train, X_test, Y_train, Y_test = split_data(X, labels, test_size=TEST_SIZE, random_state=DATA_RANDOM_STATE)

    # impute numeric train data
    numeric_imputer, numeric_train = impute_train_data(X_train, ["area"], "mean")
    # add the imputed columns back to the train set
    X_train["area"] = numeric_train

    # add the imputed columns in the test set
    numeric_test = impute_test_data(X_test, ["area"], numeric_imputer)
    X_test["area"] = numeric_test

    # for simplicity, let's just remove all categorical columns
    X_train = remove_categorical_columns(X_train)
    X_test = remove_categorical_columns(X_test)

    # create a model
    model = create_forest_regressor(MODEL_RANDOM_STATE)
    # train model
    model = train_model(model, X_train, Y_train)

    # get predictions
    in_sample_prediction = get_predictions(model, X_train)
    out_sample_prediction = get_predictions(model, X_test)

    # compute the metrics
    in_mae = compute_mae(Y_train, in_sample_prediction)
    in_mape = compute_mape(Y_train, in_sample_prediction)
    out_mae = compute_mae(Y_test, out_sample_prediction)
    out_mape = compute_mape(Y_test, out_sample_prediction)

    print(f"In-sample: Mean Absolute Error: {in_mae:.2f}\t Mean Absolute Percentage Error: {in_mape:.2f}")
    print(f"Out-sample: Mean Absolute Error: {out_mae:.2f}\t Mean Absolute Percentage Error: {out_mape:.2f}")

# indicate that if you run this file, only the code logic from run_pipeline() is executed
# it's quite a complicated matter, so check out this: https://stackoverflow.com/questions/419163/what-does-if-name-main-do
if __name__ == "__main__":
    run_pipeline()