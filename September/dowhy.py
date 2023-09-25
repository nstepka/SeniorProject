
import numpy as np
import pandas as pd
import logging
from IPython.display import Image, display
import os
import dowhy
from dowhy import CausalModel
from econml.metalearners import TLearner
from sklearn.ensemble import HistGradientBoostingRegressor

import logging
logging.getLogger().setLevel(logging.CRITICAL)


all_columns = nashvilleDF.columns.tolist()

defined_nodes = ["beds", "bedrooms", "accommodates", "price", "number_of_reviews", 
                 "review_scores_rating", "neighbourhood_cleansed_num", "pool", 
                 "host_is_superhost", "bathrooms", "pets_allowed", "hot_tub", "host_response_time"]


missing_nodes = [node for node in all_columns if node not in defined_nodes and node not in ["id"]]
print("Missing nodes:", missing_nodes)

missing_nodes = [node for node in all_columns if node not in defined_nodes and node not in ["id"]]
print("Missing nodes:", missing_nodes)


logging.getLogger("dowhy").setLevel(logging.ERROR)  # This will show only errors and suppress warnings

nashvilleDF = nashvilleDF.astype(float)

# Calculate total income for each property and add it directly to the DataFrame
nashvilleDF['total_income'] = (nashvilleDF['number_of_reviews'] / 0.5 * 4.3) * nashvilleDF['price']



output_directory = r"C:\Users\nstep\TSU\SeniorProject"


from dowhy import CausalModel

# Add all missing nodes as confounders affecting the price
extra_nodes_str = "; ".join([f'"{node}" -> price' for node in missing_nodes])
causal_graph_updated = f"""
digraph {{
  {extra_nodes_str}
  beds -> bedrooms;
  bedrooms -> accommodates;
  bedrooms -> price;
  accommodates -> price;
  number_of_reviews -> price;
  review_scores_rating -> price;
  neighbourhood_cleansed_num -> bedrooms;
  neighbourhood_cleansed_num -> accommodates;
  neighbourhood_cleansed_num -> price;
  pool -> price;
  host_is_superhost -> price;
  accommodates -> bathrooms;
  bathrooms -> price;
  "pets allowed" -> price;
  "hot tub" -> price;
  host_response_time -> price;
}}
"""

# Rest of the code remains unchanged


import dowhy


import graphviz

graph = graphviz.Digraph()
graph.node("beds")
graph.node("bedrooms")
graph.node("accommodates")
graph.node("price")
graph.node("number_of_reviews")
graph.node("review_scores_rating")
graph.node("neighbourhood_cleansed_num")
graph.node("pool")
graph.node("host_is_superhost")
graph.node("bathrooms")
graph.node("pets_allowed")
graph.node("hot_tub")
graph.node("host_response_time")
graph.edge("beds", "bedrooms")
graph.edge("bedrooms", "accommodates")
graph.edge("bedrooms", "price")
graph.edge("accommodates", "price")
graph.edge("number_of_reviews", "price")
graph.edge("review_scores_rating", "price")
graph.edge("neighbourhood_cleansed_num", "bedrooms")
graph.edge("bedrooms", "accommodates")
graph.edge("accommodates", "price")
graph.edge("pool", "price")
graph.edge("host_is_superhost", "price")
graph.edge("accommodates", "bathrooms")
graph.edge("bathrooms", "price")
graph.edge("pets_allowed", "price")
graph.edge("hot_tub", "price")

graph.render('causal_graph.dot', view=True)

nashvilleDF = nashvilleDF.astype(float)
# Load the Airbnb data
airbnb_df = nashvilleDF.copy


import dowhy
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

# Create a set of all columns from the dataframe
all_columns = set(nashvilleDF.columns)

# Create a set of primary variables that we've established relationships for
primary_variables = {
    "beds", "bedrooms", "accommodates", "price", "number_of_reviews", 
    "review_scores_rating", "neighbourhood_cleansed_num", "pool", "host_is_superhost",
    "bathrooms", "pets allowed", "hot tub", "host_response_time"
}

# Subtract primary variables from all columns to get the extra columns
extra_columns = all_columns - primary_variables

# Create relationships for these extra columns
extra_nodes_str = "; ".join([f'"{node}" -> price' for node in extra_columns])

# Create the updated causal graph
causal_graph_updated = f"""
digraph {{
  {extra_nodes_str}
  beds -> bedrooms;
  bedrooms -> accommodates;
  bedrooms -> price;
  accommodates -> price;
  number_of_reviews -> price;
  review_scores_rating -> price;
  neighbourhood_cleansed_num -> bedrooms;
  neighbourhood_cleansed_num -> accommodates;
  neighbourhood_cleansed_num -> price;
  pool -> price;
  host_is_superhost -> price;
  accommodates -> bathrooms;
  bathrooms -> price;
  "pets allowed" -> price;
  "hot tub" -> price;
  host_response_time -> price;
}}
"""

# Proceed with the causal analysis using the updated causal graph
causal_effects_updated = {}

for treatment in treatment_variables:
    causal_model = CausalModel(
        data=nashvilleDF, 
        treatment=treatment, 
        outcome=outcome_variable, 
        graph=causal_graph_updated
    )
    
    # Identify the effect
    identified_estimand = causal_model.identify_effect(proceed_when_unidentifiable=True)
    
    # Estimate the effect
    estimate = causal_model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
    )
    causal_effects_updated[treatment] = estimate.value

# Rank the treatments based on their effect sizes
ranked_treatments_updated = sorted(causal_effects_updated.items(), key=lambda x: abs(x[1]), reverse=True)

ranked_treatments_updated
