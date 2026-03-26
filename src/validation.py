import great_expectations as gx
import pandas as pd
import sys

# Define path to our cleaned data
CLEAN_DATA_PATH = "data/gold_standard_bike_data.csv"

def run_data_validation():
    print(f"Loading data from {CLEAN_DATA_PATH} for validation...")
    
    try:
        # Load the CSV
        df = pd.read_csv(CLEAN_DATA_PATH)
    except FileNotFoundError:
        print("Error: Gold Standard dataset not found. Please run ingestion.py first.")
        sys.exit(1)

    print("\nSetting up Great Expectations (v1.0 API)...")
    
    # 1. Create an ephemeral context (runs in-memory)
    context = gx.get_context(mode="ephemeral")
    
    # 2. Connect the Pandas dataframe to the context
    data_source = context.data_sources.add_pandas("bike_data")
    data_asset = data_source.add_dataframe_asset(name="bike_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("bike_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    
    # 3. Create an Expectation Suite
    suite = gx.ExpectationSuite(name="bike_quality_suite")
    
    # 4. Add our 3 specific expectations based on your PRD
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="datetime"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="cnt"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="cnt", min_value=0))
    
    print("Running Quality Checks...")
    
    # 5. Validate the batch against the suite
    validation_results = batch.validate(suite)
    
    # Extract results
    success = validation_results.success
    results = validation_results.results
    
    # Print individual checks
    print(f"Check 1 - No missing timestamps: {'Passed' if results[0].success else 'Failed'}")
    print(f"Check 2 - No missing 'cnt' values: {'Passed' if results[1].success else 'Failed'}")
    print(f"Check 3 - 'cnt' is strictly positive: {'Passed' if results[2].success else 'Failed'}")

    # Pipeline Gatekeeper Logic
    if success:
        print("\n SUCCESS: All data contracts passed! The data is clean and ready for modeling.")
    else:
        print("\n FAILURE: Data contract violated. Pipeline stopped.")
        # Exit with an error code to stop any future automated pipelines
        sys.exit(1)

if __name__ == "__main__":
    run_data_validation()