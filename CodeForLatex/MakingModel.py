# Import necessary libraries
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import dowhy
from dowhy import CausalModel
import re
from collections import Counter



def load_data(filepath):
    df = pd.read_csv(filepath, header=None, skiprows=0)
    df.columns = df.iloc[0]
    df = df[1:]
    return df



# Load and preprocess data
file_path = r"C:\Users\nstep\TSU\SeniorProject\DataSet\listings.csv"
nashvilleDF = load_data(file_path)

#nashvilleDf to lower case

nashvilleDF.head()

pd.options.display.max_columns = 200
# Colors sourced from here: https://usbrandcolors.com/airbnb-colors/
bnb_red = '#FF5A5F'
bnb_blue = '#00A699'
bnb_orange = '#FC642D'
bnb_lgrey = '#767676'
bnb_dgrey = '#484848'
bnb_maroon = '#92174D'
# Create diverging colormap for heatmaps
bnb_cmap = sns.diverging_palette(210,
                                 13,
                                 s=81,
                                 l=61,
                                 sep=3,
                                 n=16,
                                 as_cmap=True)

# Test colors
sns.palplot(sns.diverging_palette(210, 13, s=81, l=61, sep=3, n=16))



# Create color palette
bnb_palette = sns.color_palette(
    ["#FF5A5F", "#007989", "#8CE071", "#FC642D", "#92174D", "#01D1C1"])

# Test colors
sns.palplot(bnb_palette)


nashvilleDF.shape
#nashvilleDF.info()

#write nashvilleDF.info to csv 
info = nashvilleDF.info()



# Visualize price table, changing them to floats and replacing the commas with a blank
prices = nashvilleDF['price'].apply(lambda s: float(s[1:].replace(',','')))

# Drop listings with a price of zero
prices = prices[prices!=0]

# Log prices
log_prices = np.log(prices)

print(log_prices.describe())

def plot_hist(n, titles, ranges):
    """
    Quick helper function to plot histograms
    """
    fig, ax = plt.subplots(n, figsize = (8, 7.5))
    for i in range(n):
        d, bins, patches = ax[i].hist(ranges[i], 50, density = 1, color='red', alpha = 0.85)
        ax[i].set_title(titles[i])
        ax[i].set_xlabel("Daily Listing Price in Dollars")
        ax[i].set_ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# Plot histograms of price distribution
plot_hist(4, ['Distribution of Listing Prices: All Data', 'Distribution of Listing Prices: $0 - $1000', 
               'Distribution of Listing Prices: $0 - $350','Log Transformed Distribution of Listing Prices: All Data'], 
          [prices, prices[prices <= 1000], prices[prices < 350],log_prices])



# Convert relevant columns to datetime format
nashvilleDF.last_scraped = pd.to_datetime(
    nashvilleDF.last_scraped)

nashvilleDF.host_since = pd.to_datetime(
    nashvilleDF.host_since)

nashvilleDF.calendar_last_scraped = pd.to_datetime(
    nashvilleDF.calendar_last_scraped)

nashvilleDF.first_review = pd.to_datetime(
    nashvilleDF.first_review)

nashvilleDF.last_review = pd.to_datetime(
    nashvilleDF.last_review)



nashvilleDF.host_response_rate = nashvilleDF[
    'host_response_rate'].apply(lambda s: float(str(s).replace('%', '')))



nashvilleDF.property_type.value_counts()

sns.countplot(y=nashvilleDF['property_type'])
plt.ylabel('Property Types')
plt.xlabel('Number of Listings')
plt.title('Number of Listings by Propery Type')

# Step 1: Create a mask to identify rows with 'Private' in the 'property_type' column
private_mask = nashvilleDF['property_type'].str.contains('Private')

# Step 2: Update the values in the 'property_type' column based on the mask
nashvilleDF.loc[private_mask, 'property_type'] = 'Private room'

nashvilleDF.replace('Aparthotel','Hotel',inplace=True)
nashvilleDF.replace('Room in aparthotel','Hotel',inplace=True)
nashvilleDF.replace('Room in hotel','Hotel',inplace=True)
nashvilleDF.replace('Room in boutique hotel','Hotel',inplace=True)
nashvilleDF.replace('Entire serviced apartment','Entire condo',inplace=True)
nashvilleDF.replace('Entire loft','Entire condo',inplace=True)

private_mask = nashvilleDF['property_type'].str.contains('Shared')

nashvilleDF.loc[private_mask, 'property_type'] = 'Shared room'

# Display the updated value counts
updated_value_counts = nashvilleDF['property_type'].value_counts()
updated_value_counts

# Display the updated value counts
updated_value_counts_after_drop = nashvilleDF['property_type'].value_counts()
print(updated_value_counts_after_drop)


# List of property types to drop
drop_list = [
    "Entire bungalow", "Entire cottage", "Tiny home", "Camper/RV", "Entire cabin",
    "Entire vacation home", "Entire place", "Shared room", "Entire villa",
    "Farm stay", "Entire home/apt", "Earthen home", "Barn", "Entire chalet",
    "Bus", "Shipping container", "Boat", "Tent"
]

# Drop rows based on the property_type values in drop_list
nashvilleDF = nashvilleDF[~nashvilleDF['property_type'].isin(drop_list)]

# Display the updated value counts
updated_value_counts_after_drop = nashvilleDF['property_type'].value_counts()
print(updated_value_counts_after_drop)


updated_value_counts_after_drop = nashvilleDF['host_response_time'].value_counts()
print(updated_value_counts_after_drop)

# Host Response Time
# We will treat nan values as 

nashvilleDF[
    'host_response_time'] = nashvilleDF.host_response_time.map({
        'within an hour':
        1,
        'within a few hours':
        2,
        'within a day':
        3,
        'a few days or more':
        4,
        np.nan:
        5
    })




updated_value_counts_after_drop = nashvilleDF['host_response_time'].value_counts()
print(updated_value_counts_after_drop)



# Display the updated value counts
updated_value_counts_after_drop = nashvilleDF['host_is_superhost'].value_counts()
print(updated_value_counts_after_drop)


nashvilleDF['host_is_superhost'] = [
    1 if x == 't' else 0 for x in nashvilleDF.host_is_superhost
]
# Display the updated value counts
updated_value_counts_after_drop = nashvilleDF['host_is_superhost'].value_counts()
print(updated_value_counts_after_drop)


# Display the updated value counts
updated_value_counts_after_drop = nashvilleDF['instant_bookable'].value_counts()
print(updated_value_counts_after_drop)

nashvilleDF['instant_bookable'] = [
    1 if x == 't' else 0 for x in nashvilleDF.instant_bookable
]

# Display the updated value counts
updated_value_counts_after_drop = nashvilleDF['instant_bookable'].value_counts()
print(updated_value_counts_after_drop)


# Display the updated value counts
updated_value_counts_after_drop = nashvilleDF['has_availability'].value_counts()
print(updated_value_counts_after_drop)

nashvilleDF['has_availability'] = [
    1 if x == 't' else 0 for x in nashvilleDF.has_availability
]

# Display the updated value counts
updated_value_counts_after_drop = nashvilleDF['has_availability'].value_counts()
print(updated_value_counts_after_drop)

# Display the updated value counts
updated_value_counts_after_drop = nashvilleDF['host_has_profile_pic'].value_counts()
print(updated_value_counts_after_drop)

nashvilleDF['host_has_profile_pic'] = [
    1 if x == 't' else 0 for x in nashvilleDF.host_has_profile_pic
]

# Display the updated value counts
updated_value_counts_after_drop = nashvilleDF['host_has_profile_pic'].value_counts()
print(updated_value_counts_after_drop)


# Display the updated value counts
updated_value_counts_after_drop = nashvilleDF['host_identity_verified'].value_counts()
print(updated_value_counts_after_drop)


nashvilleDF['host_identity_verified'] = [
    1 if x == 't' else 0 for x in nashvilleDF.host_identity_verified
]

# Display the updated value counts
updated_value_counts_after_drop = nashvilleDF['host_identity_verified'].value_counts()
print(updated_value_counts_after_drop)


# Property Type
# One hot encode
prop_type_dummies = pd.get_dummies(nashvilleDF.property_type,
                                   prefix='prop')

# Merge with df
nashvilleDF = nashvilleDF.merge(prop_type_dummies,
                                left_index=True,
                                right_index=True)


nashvilleDF.property_type.value_counts()


nashvilleDF.room_type.value_counts()


# Room Type
# Create numerical column for room type
nashvilleDF['room_type_num'] = nashvilleDF.room_type.map({
    'Entire home/apt':
    3,
    'Private room':
    2,
    'Hotel room':
    1,
})

# One hot encode
room_type_dummies = pd.get_dummies(nashvilleDF.room_type,
                                   prefix='room')

# Drop "hotel room" column as base case
del room_type_dummies['room_Hotel room']

# Merge with df
nashvilleDF = nashvilleDF.merge(room_type_dummies,
                                                    left_index=True,
                                                    right_index=True)


nashvilleDF.room_type_num.value_counts()


# Combine 'Private room' and 'Hotel room' into 'Private/Hotel room'
nashvilleDF['room_type'] = nashvilleDF['room_type'].replace(['Private room', 'Hotel room'], 'Private/Hotel room')

# Display the updated value counts for room_type column
updated_value_counts_room_type = nashvilleDF['room_type'].value_counts()
print(updated_value_counts_room_type)

# Map the updated room types to numeric values
nashvilleDF['room_type_num'] = nashvilleDF.room_type.map({
    'Entire home/apt': 2,
    'Private/Hotel room': 1
})

# Display the updated value counts for room_type_num column
updated_value_counts_room_type_num = nashvilleDF['room_type_num'].value_counts()
updated_value_counts_room_type_num


nashvilleDF.head()

nashvilleDF[nashvilleDF['bedrooms'].isnull()]


# Extract numeric portion from the 'bathrooms_text' column and assign to 'bathrooms' column
nashvilleDF['bathrooms'] = nashvilleDF['bathrooms_text'].str.extract('(\d+\.?\d*)').astype(float)

# Drop the 'bathrooms_text' column
nashvilleDF = nashvilleDF.drop(columns=['bathrooms_text'])


#show nashvilleDF['bedrooms'] that are empty
nashvilleDF.bathrooms.value_counts()


nashvilleDF.neighbourhood.value_counts()


nashvilleDF.neighbourhood.fillna('Empty',inplace=True)

nashvilleDF.neighbourhood.value_counts()


#drop neighbourhood from nashvilleDF
nashvilleDF.drop('neighbourhood',axis=1,inplace=True)
nashvilleDF.neighbourhood_cleansed.value_counts()

# Extract the district number from the 'neighbourhood_cleansed' column and assign to 'neighbourhood_cleansed_num'
nashvilleDF['neighbourhood_cleansed_num'] = nashvilleDF['neighbourhood_cleansed'].str.extract('(\d+)').astype(int)

# Display the updated value counts for the 'neighbourhood_cleansed_num' column
updated_neighbourhood_cleansed_num_value_counts = nashvilleDF['neighbourhood_cleansed_num'].value_counts()
updated_neighbourhood_cleansed_num_value_counts


import re
from collections import Counter

# Format amenities column for analysis
nashvilleDF.amenities = nashvilleDF.amenities.apply(
    lambda x: [i.strip() for i in re.sub('[^a-zA-Z,\/\s\d-]*', '', x.lower()).split(sep=',')] if isinstance(x, str) else x)

# Create a flat list of all amenities entries
amenities_list = [item for sublist in nashvilleDF.amenities for item in sublist if isinstance(sublist, list)]

# Count amenities occurrences
amenity_counts = Counter(amenities_list).most_common()

# Examine the top 20 amenities
top_amenity_counts = amenity_counts[0:20]

# Look at the 90 least common amenities
lowest_amenity_counts = amenity_counts[-90:]

# Total unique amenities
total_unique_amenities = len(set(amenities_list))

lowest_amenity_counts, total_unique_amenities



# Make a list of amenities of interest
amenities_of_interest = [x[0] for x in amenity_counts[0:70]]

#print amenity_counts 1 per row
for amenity in amenity_counts:
    print(amenity)
    
print(amenity_counts)


combine_keywords = {
    "coffee": "coffee",
    "shampoo": "shampoo",
    "toaster": "toaster",
    "crib": "crib",
    "hot tub": "hot tub",
    "refrigerator": "refrigerator",
    "gym": "gym",
    "resort access": "resort access",
    "microwave": "microwave",
    "kitchen": "kitchen",
    "camera": "camera",
    "cable": "cable",
    "grill": "grill",
    "stove": "stove",
    "backyard": "backyard",
    "bluetooth": "bluetooth",
    "wifi": "wifi",
    "oven": "oven",
    "sono": "sono",
    "disney": "disney",
    "high chair": "high chair",
    "tv": "tv", 
    "hdtv": "tv", # Note: TV combines with HDTV
    "hbo": "hbo",
    "netflix": "netflix",
    "pool": "pool",
    "conditioner": "conditioner",
    "soap": "soap",
    "iron": "iron",
    "sony": "sony",
    "sound system": "sound system",
    "heating": " radiant heating",
    "stainless": "stainless",
    "washer": "washer",
    "dryer": "dryer",
    "free parking": "free parking",
    "baby monitor": "baby monitor",
    "baby bath": "baby bath",
    "body wash": "body wash",
    "changing table": "changing table",
    "books and toys": "books and toys",
    "clothing storage": "clothing storage",
    "exercise equipment": "exercise equipment",
    "carport": "carport",
    "free residential garage": "free residential garage",
    "game console": "game console",
    "fireplace": "fireplace",
    "paid parking garage": "paid parking garage",
    "paid parking lot off": "paid parking lot off",
    "paid parking lot on": "paid parking lot on",
    "paid parking on premises": "paid parking on premises",
    "paid valet parking": "paid valet parking",
    "hot water": "hot water",
    "free driveway": "free driveway",

}

# List to store new aggregated amenity counts
aggregated_amenity_counts = {}

# Loop through amenities and aggregate based on keywords
for amenity, count in amenity_counts:
    found = False
    for keyword, new_amenity in combine_keywords.items():
        if keyword in amenity.lower():
            # Special handling for "oven" to ensure it doesn't get overshadowed by "toaster"
            if keyword == "oven" and "toaster" in amenity.lower():
                continue
            # Special handling for "hot water" to ensure "hot water kettle" doesn't get combined
            if keyword == "hot water" and "kettle" in amenity.lower():
                continue
            if new_amenity not in aggregated_amenity_counts:
                aggregated_amenity_counts[new_amenity] = 0
            aggregated_amenity_counts[new_amenity] += count
            found = True
            break
    if not found:
        if amenity not in aggregated_amenity_counts:
            aggregated_amenity_counts[amenity] = 0
        aggregated_amenity_counts[amenity] += count

# Convert dictionary to list of tuples and sort
sorted_amenity_counts = sorted(aggregated_amenity_counts.items(), key=lambda x: x[1], reverse=True)

# Save sorted_amenity_counts to csv
df = pd.DataFrame(sorted_amenity_counts, columns=["Amenity", "Count"])
df.to_csv('new_amenity_counts.csv', index=False)

# Assuming nashvilleDF has a column 'amenities' that is either a string or list

# Ensure that the 'amenities' column is a list
nashvilleDF['amenities'] = nashvilleDF['amenities'].apply(lambda x: x if isinstance(x, list) else x.split(','))

# Define a function to aggregate amenities based on combine_keywords
def aggregate_amenities(amenities_list):
    aggregated_list = []
    for amenity in amenities_list:
        found = False
        for keyword, new_amenity in combine_keywords.items():
            if keyword in amenity.lower():
                aggregated_list.append(new_amenity)
                found = True
                break
        if not found:
            aggregated_list.append(amenity)
    return aggregated_list

# Apply the aggregate_amenities function to the amenities column
nashvilleDF['aggregated_amenities'] = nashvilleDF['amenities'].apply(aggregate_amenities)

# Create dummy variables
amenities_dummies = pd.get_dummies(nashvilleDF['aggregated_amenities'].apply(pd.Series).stack()).sum(level=0)

# Join the dummy variables to the original DataFrame and drop the amenities columns
nashvilleDF = nashvilleDF.join(amenities_dummies)
nashvilleDF = nashvilleDF.drop(["amenities", "aggregated_amenities"], axis=1)




# Convert columns to numeric and handle NaN values
columns_to_convert = [
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication', 'review_scores_location',
    'review_scores_value'
]

for col in columns_to_convert:
    nashvilleDF[col] = pd.to_numeric(nashvilleDF[col], errors='coerce')
    nashvilleDF[col].fillna(0, inplace=True)


# Define the size of the entire figure. (width, height)
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(10, 14))

ax1.hist(nashvilleDF.review_scores_rating, color='blue')
ax1.set_title('Rating')
ax2.hist(nashvilleDF.review_scores_accuracy, color='blue')
ax2.set_title('Accuracy')
ax3.hist(nashvilleDF.review_scores_cleanliness, color='blue')
ax3.set_title('Cleanliness')
ax4.hist(nashvilleDF.review_scores_checkin, color='blue')
ax4.set_title('Check In')
ax5.hist(nashvilleDF.review_scores_communication, color='blue')
ax5.set_title('Communication')
ax6.hist(nashvilleDF.review_scores_location, color='blue')
ax6.set_title('Location')
ax7.hist(nashvilleDF.review_scores_value, color='blue')
ax7.set_title('Value')

# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.5)

plt.tight_layout()  # This ensures that the titles and plots don't overlap
plt.show()


def findOutliers(df, column, lim_scalar=4):
    """
    Returns outliers above the max limit for a column in a dataframe
    Adjust outlier cutoff to q75 + 4*iqr to include more data
    ---
    input: DataFrame, column(series),lim_scalar(float)
    output: DataFrame
    """
    q25, q50, q75 = df[column].quantile(q=[0.25, 0.5, 0.75])
    iqr = q75 - q25
    # max limits to be considered an outlier
    max_ = q75 + lim_scalar * iqr
    # identify the points
    outlier_mask = [True if x > max_ else False for x in df[column]]
    print('{} outliers found out of {} data points, {}% of the data. {} is the max'.format(
        sum(outlier_mask), len(df[column]),
        100 * (sum(outlier_mask) / len(df[column])),max_))
    return outlier_mask
#average number of bathrooms in nashvilleDF
# Bathrooms
# Fill na with 1
nashvilleDF.bathrooms.fillna(1,inplace=True)
nashvilleDF.bathrooms.mean()
# Look at the distribution of the bathrooms column
plt.figure(figsize=(3,3))
plt.boxplot(nashvilleDF.bathrooms)
sns.despine()
plt.title('bathrooms distribution');



# Remove bathroom outliers
nashvilleDF = nashvilleDF[np.logical_not(
    findOutliers(nashvilleDF, 'bathrooms'))]
# Look at the distribution of the bathrooms column
plt.figure(figsize=(3,3))
plt.boxplot(nashvilleDF.bathrooms)
sns.despine()
plt.title('bathrooms distribution')



# Convert non-numeric values in the 'bedrooms' column to NaN
nashvilleDF['bedrooms'] = pd.to_numeric(nashvilleDF['bedrooms'], errors='coerce')

# Replace NaN values with 1
nashvilleDF['bedrooms'].fillna(1, inplace=True)

# Plot the boxplot for the cleaned 'bedrooms' column
plt.figure(figsize=(3,3))
plt.boxplot(nashvilleDF['bedrooms'])
sns.despine()
plt.title('beds distribution')
plt.show()




# Remove bedroom outliers
nashvilleDF = nashvilleDF[np.logical_not(
    findOutliers(nashvilleDF, 'bedrooms',lim_scalar=6))]


# Plot the boxplot for the cleaned 'bedrooms' column
plt.figure(figsize=(3,3))
plt.boxplot(nashvilleDF['bedrooms'])
sns.despine()
plt.title('beds distribution')
plt.show()


# Plot distribution of the number of bathrooms
plt.figure(figsize=(10, 6))
plt.hist(nashvilleDF['bathrooms'], bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Number of Bathrooms')
plt.xlabel('Number of Bathrooms')
plt.ylabel('Number of Listings')
plt.grid(axis='y')
plt.show()

# Plot distribution of the number of bathrooms
plt.figure(figsize=(10, 6))
plt.hist(nashvilleDF['bedrooms'], bins=20, color='blue', edgecolor='black')
plt.title('Distribution of Number of Bathrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Number of Listings')
plt.grid(axis='y')
plt.show()



# Calculate estimated number of bookings
nashvilleDF['est_bookings'] = nashvilleDF.number_of_reviews * 2

# Replace entries where unit is brand new with est_bookings = 1
nashvilleDF['est_bookings'] = [
    1 if nashvilleDF.first_review.iloc[idx] == nashvilleDF.last_review.iloc[idx] else x
    for idx, x in enumerate(nashvilleDF.est_bookings)
]
# Convert 'minimum_nights' to numeric and handle NaN values (assuming a default value of 1)
nashvilleDF['minimum_nights'] = pd.to_numeric(nashvilleDF['minimum_nights'], errors='coerce')
nashvilleDF['minimum_nights'].fillna(1, inplace=True)

# Calculate estimated number of nights booked per year
# Use 3 days as the average length of a stay
# Unless the minimum number of days is greater than 3, then use that number
nashvilleDF['est_booked_nights_per_year'] = [
    3 if x < 3 else x
    for x in nashvilleDF.minimum_nights  # avg stay length
] * nashvilleDF.reviews_per_month * 2 * 12
# Calculate estimated number of nights booked
# Use 3 days as the average length of a stay
# Unless the minimum number of days is greater than 3, then use that number

nashvilleDF['est_booked_nights'] = (
    [
        3 if x < 3 else x
        for x in nashvilleDF.minimum_nights
    ] *  # avg stay length
    nashvilleDF['est_bookings'])
# Convert 'est_booked_nights' to numeric and handle NaN values
nashvilleDF['est_booked_nights'] = pd.to_numeric(nashvilleDF['est_booked_nights'], errors='coerce')
nashvilleDF['est_booked_nights'].fillna(0, inplace=True)

# Occupancy Rate = total_booked_nights / total_available_nights
nashvilleDF['occupancy_rate'] = nashvilleDF['est_booked_nights'] / (
    (nashvilleDF.last_review - nashvilleDF.first_review).dt.days + 1)

# The next line seems redundant and is the same as the previous one. Consider removing it.
# Occupancy Rate = total_booked_nights / total_available_nights
nashvilleDF['occupancy_rate'] = nashvilleDF['est_booked_nights'] / (
    (nashvilleDF.last_review - nashvilleDF.first_review).dt.days + 1)
# Convert 'availability_365' to numeric and handle NaN values
nashvilleDF['availability_365'] = pd.to_numeric(nashvilleDF['availability_365'], errors='coerce')
nashvilleDF['availability_365'].fillna(0, inplace=True)


# Convert 'est_booked_nights_per_year' to numeric and handle NaN values
nashvilleDF['est_booked_nights_per_year'] = pd.to_numeric(nashvilleDF['est_booked_nights_per_year'], errors='coerce')
nashvilleDF['est_booked_nights_per_year'].fillna(0, inplace=True)

# Calculate occupancy rate
#nashvilleDF['occupancy_rate2'] = nashvilleDF['est_booked_nights_per_year'] / (nashvilleDF['availability_365'] + 1)
Getting rid of data that isnt numerical or helpful.

columns_to_drop = [
    "listing_url",
    "scrape_id",
    "source",
    "name",
    "description",
    "neighborhood_overview",
    "picture_url",
    "host_id",
    "host_url",
    "host_name",
    "host_location",
    "host_about",
    "host_thumbnail_url",
    "host_picture_url",
    "host_neighbourhood",
    "host_verifications",
    "neighbourhood_cleansed",
    "neighbourhood_group_cleansed",
    "latitude",
    "longitude",
    'property_type',
    'room_type',
    'est_booked_nights_per_year',
    'est_booked_nights',
    'occupancy_rate',
    'host_listings_count',
    'est_bookings',
    'calculated_host_listings_count',
    'calculated_host_listings_count_private_rooms',
    'minimum_nights_avg_ntm',
    'maximum_nights_avg_ntm',
    'calculated_host_listings_count_entire_homes',
    'host_total_listings_count',
    'availability_90',
    'availability_60', 'minimum_nights',
    'maximum_minimum_nights', 'review_scores_communication',
    'smaller', 'ge', 'irish spring', 'lotion','suave', 'dove', 'dove anti-stress moisturizing cream bar','dr teals', 'smaller fridge',
    'organic', 'olympic-sized','number_of_reviews_l30d', 'number_of_reviews_ltm',
    '5-10 years old', '2-5 years old',


]
nashvilleDF['price'] = nashvilleDF['price'].str.replace('$', '').str.replace(',', '').astype(float)

nashvilleDF = nashvilleDF.drop(columns=columns_to_drop, errors='ignore')


# Convert the 'price' column to numeric (assuming it contains strings representing numbers)
nashvilleDF['price'] = pd.to_numeric(nashvilleDF['price'], errors='coerce')

try:
    # Calculate the required statistics
    total_entries = len(nashvilleDF['price'])
    entries_500_or_less = len(nashvilleDF[nashvilleDF['price'] <= 500])
    entries_over_500 = len(nashvilleDF[nashvilleDF['price'] > 1000])
    
    results = total_entries, entries_500_or_less, entries_over_500
except Exception as e:
    error_message = str(e)
    results = None

results, error_message if results is None else None


#drop first_review	last_review last_scraped	host_since
nashvilleDF.drop(['first_review','last_review','last_scraped','host_since',
                  'calendar_updated','calendar_last_scraped','license'],axis=1,inplace=True)

#host_acceptance_rate has % on it, drop the % from it
nashvilleDF['host_acceptance_rate'] = nashvilleDF['host_acceptance_rate'].str.replace('%', '').astype(float)
missing_data = nashvilleDF.isnull().sum()
print(missing_data[missing_data > 0])



nashvilleDF = nashvilleDF.apply(pd.to_numeric, errors='coerce')

nashvilleDF["host_response_rate"] = nashvilleDF["host_response_rate"].fillna(nashvilleDF["host_response_rate"].median())
nashvilleDF["host_acceptance_rate"] = nashvilleDF["host_acceptance_rate"].fillna(nashvilleDF["host_acceptance_rate"].median())

# Impute missing values for beds with the median
nashvilleDF["beds"] = nashvilleDF["beds"].fillna(nashvilleDF["beds"].median())

# Convert reviews_per_month to an integer column
nashvilleDF["reviews_per_month"] = nashvilleDF["reviews_per_month"].astype("float64")

# Impute missing values for reviews_per_month and occupancy_rate with the mean
nashvilleDF["reviews_per_month"] = nashvilleDF["reviews_per_month"].fillna(nashvilleDF["reviews_per_month"].mean())
#limit price between 200-300
nashvilleDF = nashvilleDF[nashvilleDF['price'] >= 000]


nashvilleDF = nashvilleDF[nashvilleDF['price'] <= 1000]
missing_data = nashvilleDF.isnull().sum()
print(missing_data[missing_data > 0])
Series([], dtype: int64)
#write nashvilleDF to a csv file
nashvilleDF.to_csv('nashvilleDF.csv', index=False)
plt.figure(figsize=(10, 6))
sns.histplot(nashvilleDF['price'], bins=50, kde=True)
plt.title('Distribution of Price')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()


# Split data into features (X) and target (y)
X = nashvilleDF.drop('price', axis=1)
y = nashvilleDF['price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)
# Calculate and print R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2}")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

subsetdf = nashvilleDF.copy()
df = subsetdf.copy()

def prepare_data(df, target_col='price'):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def create_models():
    return {
        'GradientBoosting': GradientBoostingRegressor(),
        'RandomForest': RandomForestRegressor(),
        'Linear': LinearRegression(),
        'HistGradientBoosting': HistGradientBoostingRegressor(),
        'DecisionTree': DecisionTreeRegressor(),
        'XGBoost': xgb.XGBRegressor(),
    }

def fit_models(X_train, y_train, models):
    for name, model in models.items():
        model.fit(X_train.astype(float), y_train.astype(float))
    return models

def evaluate_models(X_test, y_test, models):
    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test.astype(float))

        r2 = r2_score(y_test.astype(float), y_pred)
        mse = mean_squared_error(y_test.astype(float), y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test.astype(float), y_pred)

        results.append({
            'Model': name,
            'R2': r2,
            'RMSE': rmse,
            'MSE': mse,
            'MAE': mae
        })
    return pd.DataFrame(results)

def plot_model_performance(results_df):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle('Model Performance Comparison')

    metrics = ['R2', 'RMSE', 'MSE', 'MAE']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]
        results_df.plot(x='Model', y=metric, kind='bar', ax=ax, legend=None)
        ax.set_ylabel(metric)
        ax.set_ylim(bottom=0)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def main(df):
    X_train, X_test, y_train, y_test = prepare_data(df)
    models = create_models()
    models = fit_models(X_train, y_train, models)
    results_df = evaluate_models(X_test, y_test, models)
    plot_model_performance(results_df)
    table = pd.pivot_table(results_df, index='Model', values=['R2', 'RMSE', 'MSE', 'MAE'])
    print(table)

main(df)








from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

# Initialize the HistGradientBoostingRegressor model
model = HistGradientBoostingRegressor()

# Specify the number of folds for k-fold cross-validation
num_folds = 10

# Initialize the KFold object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)


# Initialize an array to store the accuracy scores
scores = np.zeros(num_folds)

# Initialize an array to store feature importances for each fold
# The number of columns in X is assumed to be X.shape[1]
feature_importances = np.zeros((num_folds, X.shape[1]))

# Loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    # Split the data into training and test sets
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]

    # Train your model on the training set
    model.fit(X_train, y_train)

    # Evaluate your model on the test set
    scores[fold] = model.score(X_test, y_test)

    # Get feature importances using permutation importance
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    feature_importances[fold] = perm_importance.importances_mean

    # Print the results
    print(f'Fold {fold+1}: Accuracy = {scores[fold]}')

# Compute and print the average accuracy across all folds
print(f'Average Accuracy: {np.mean(scores)}')

# Compute the average feature importance across all folds
avg_feature_importances = feature_importances.mean(axis=0)

# Get the top 20 features based on average importance
top_features = np.argsort(avg_feature_importances)[-20:]
top_feature_names = X.columns[top_features]
top_feature_importances = avg_feature_importances[top_features]

top_feature_names, top_feature_importances






import pandas as pd
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from collections import defaultdict

# Define function to get feature importances
def get_feature_importance(data, features):
    X = data[features]
    y = data['price']
    
    model = HistGradientBoostingRegressor()
    model.fit(X, y)
    
    # Using permutation importance to get feature importances
    result = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    importances = result.importances_mean
    
    return dict(zip(features, importances))

# Iteratively get feature importances for random feature subsets
iterations = 100  # Number of iterations
num_features = 20

feature_columns = [col for col in nashvilleDF.columns if col != 'price']

collective_importances = defaultdict(float)

for _ in range(iterations):
    selected_features = np.random.choice(feature_columns, num_features, replace=False)
    
    importances = get_feature_importance(nashvilleDF, selected_features)
    
    for feature, importance in importances.items():
        collective_importances[feature] += importance

# Sort features by their collective importance
sorted_features = sorted(collective_importances.keys(), key=lambda x: collective_importances[x], reverse=True)

print(sorted_features)

from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

# Split the data into features and target
X = nashvilleDF.drop('price', axis=1)
y = nashvilleDF['price']

# Initialize the model
model = HistGradientBoostingRegressor()

# Fit the model
model.fit(X, y)

# Get feature importances using permutation importance
perm_importance = permutation_importance(model, X, y, n_repeats=30, random_state=42)

# Get the top 20 features
top_20_features = sorted(zip(perm_importance.importances_mean, X.columns), reverse=True)[:20]
top_20_features



# Categorizing amenities into defined buckets
kitchen_amenities = [
    "coffee", "kitchen", "refrigerator", "dishes and silverware", "microwave",
    "oven", "stove", "cooking basics", "freezer", "wine glasses", "toaster", 
    "hot water kettle", "grill", "blender", "barbecue utensils", "rice maker",
    "nespresso", "espresso machine", "french press", "bread maker", "dining table",
    "baking sheet", "gas", "we have a countertop burner"
]

cleaning_amenities = [
    "dryer", "washer", "shampoo", "iron", "hot water", "soap", "cleaning products", 
    "shower gel", "conditioner", "cleaning available during stay", "body wash", 
    "lotion", "suave", "irish spring", "la botegga", "dove anti-stress moisturizing cream bar"
]

safety_amenities = [
    "smoke alarm", "carbon monoxide alarm", "fire extinguisher", "first aid kit", 
    "keypad", "smart lock", "lockbox", "camera", "lock on bedroom door", "safe", 
    "baby safety gates", "outlet covers", "table corner guards", "window guards", 
    "ev charger", "ev charger - level 2", "ev charger - level 1", "tesla only"
]

household_amenities = [
    "radiant heating", "essentials", "hangers", "self check-in", "bed linens", 
    "free parking", "air conditioning", "private entrance", "extra pillows and blankets", 
    "dedicated workspace", "free street parking", "long term stays allowed", "clothing storage",
    "ceiling fan", "central air conditioning", "private patio or balcony", "room-darkening shades", 
    "luggage dropoff allowed", "single level home", "elevator", "fireplace", 
    "books and reading material", "laundromat nearby", "board games", "backyard", 
    "outdoor furniture", "patio or balcony", "outdoor dining area", "gym", 
    "fire pit", "high chair", "crib", "pool", "hot tub", "breakfast", "resort access", 
    "city skyline view", "exercise equipment", "portable fans", "childrenu2019s dinnerware",
    "building staff", "babysitter recommendations", "free driveway", "closet", 
    "shared patio or balcony", "record player", "trash compactor", "sun loungers",
    "closet", "wardrobe", "portable heater", "bidet", "changing table", "bikes",
    "hammock", "2-5 years old", "5-10 years old", "and 10 years old", "and 5-10 years old",
    "rooftop", "saltwater", "infinity", "beach essentials", "mosquito net", "heated", 
    "dvd player", "ev charger - level 1", "private sauna", "shared sauna", "window ac unit", 
    "host greets you", "piano", "baby monitor", "ping pong table", "private living room", 
    "portable air conditioning", "smoking allowed", "baby bath", "beach access", "ski-in/ski-out", 
    "wood-burning", "outdoor shower", "ac - split type ductless system", "window ac unit", 
    "all natural", "stainless", "portable", "sono", "but good size not standard large size", 
    "smaller", "some type", "two racks", "organic", "olympic-sized", "smaller fridge"
]

bedroom_amenities = [
    "hangers", "bed linens", "extra pillows and blankets", "room-darkening shades", 
    "crib", "single level home", "closet", "wardrobe", "and dresser", "changing table"
]

electronics_amenities = [
    "wifi", "tv", "ethernet connection", "cable", "roku", "hulu", "disney", "hbo", 
    "netflix", "amazon prime video", "bluetooth", "game console", "xbox 360", "and xbox one", 
    "sound system", "vizio soundbar", "chromecast"
]

extra_spaces_amenities = [
    "patio or balcony", "pool", "garden or backyard", "hot tub", "BBQ grill", 
    "private living room", "gym", "fire pit", "outdoor dining area", "backyard", 
    "outdoor furniture", "private patio or balcony", "gym", "fireplace", "rooftop", 
    "beach access", "ski-in/ski-out", "outdoor shower", "resort access", "waterfront", 
    "resort view", "harbor view", "beach view", "sea view", "ocean view", 
    "lake access", "mountain view", "valley view", "golf course view", "lake view", 
    "marina view", "public or shared beach access", "canal view", "boat slip", "bay view", 
    "park view"
]

review_amenities = [
    "number_of_reviews", "number_of_reviews_ltm", "number_of_reviews_l30d", "review_scores_rating",
    "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value"
]


# Return the categorized amenities
categorized_amenities = {
    "Kitchen Amenities": kitchen_amenities,
    "Cleaning Amenities": cleaning_amenities,
    "Safety Amenities": safety_amenities,
    "Household Amenities": household_amenities,
    "Bedroom Amenities": bedroom_amenities,
    "Electronics Amenities": electronics_amenities,
    "Extra Spaces Amenities": extra_spaces_amenities,
    "Review Amenities": review_amenities    

}
import networkx as nx
import matplotlib.pyplot as plt

# Create a new directed graph
G = nx.DiGraph()

# Add nodes for each amenity group
amenity_groups_names = [
    "Kitchen Amenities", "Cleaning Amenities", "Safety Amenities",
    "Household Amenities", "Bedroom Amenities", "Electronics Amenities",
    "Extra Spaces Amenities", "Review Amenities"
]
for group in amenity_groups_names:
    G.add_node(group)

# Add edges from each amenity group to 'price'
for group in amenity_groups_names:
    G.add_edge(group, 'price')

# Plot the DAG
plt.figure(figsize=(10, 6))

# Use shell_layout instead of spring_layout
shell_nestings = [amenity_groups_names, ['price']]  # Define the concentric circles
pos = nx.shell_layout(G, nlist=shell_nestings)

nx.draw_networkx(G, pos, with_labels=True, node_size=4000, node_color='skyblue', font_size=10, font_weight='bold', width=2, edge_color='gray')
plt.title("Causal DAG representing the effect of amenity groups on price")
plt.show()



from sklearn.model_selection import KFold
import numpy as np

# Load your data into a pandas DataFrame
data = nashvilleDF.copy()

# Split your data into features and target
X = data.drop('price', axis=1)
y = data['price']

# Define the number of folds
num_folds = 10

# Initialize the cross-validation method
kf = KFold(n_splits=num_folds)

# Initialize a linear regression model
model = HistGradientBoostingRegressor()

# Initialize an array to store the accuracy scores
scores = np.zeros(num_folds)

# Loop through each fold
for fold, (train_index, test_index) in enumerate(kf.split(X)):

    # Split the data into training and test sets
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]

    # Train your model on the training set
    model.fit(X_train, y_train)

    # Evaluate your model on the test set
    scores[fold] = model.score(X_test, y_test)

    # Print the results
    print(f'Fold {fold+1}: Accuracy = {scores[fold]}')

# Compute and print the average accuracy across all folds
print(f'Average Accuracy: {np.mean(scores)}')


model1_plot = nashvilleDF.groupby(['accommodates', 'bedrooms']).size().reset_index().pivot(columns='bedrooms',
                                                                                  index='accommodates', values=0)

model1_plot.plot(kind='bar', stacked=True, figsize=(12,5))
plt.title("Accomodates and Number of Bedrooms", size=20)
plt.xlabel("Accomodates", size=15)




# Filter rows where 'wifi' column equals 1
wifi_1_rows = nashvilleDF[nashvilleDF['wifi'] == 1]

# Calculate the percentage
percent_with_wifi_1 = (len(wifi_1_rows) / len(nashvilleDF)) * 100

print(f"Percentage of rows with 'wifi' = 1 in nashvilleDF: {percent_with_wifi_1:.2f}%")


# Filter rows where 'wifi' column equals 1
resort_1_rows = nashvilleDF[nashvilleDF['resort access'] == 1]

# Calculate the percentage
percent_with_resort_1 = (len(resort_1_rows) / len(nashvilleDF)) * 100

print(f"Percentage of rows with 'wifi' = 1 in nashvilleDF: {percent_with_resort_1:.2f}%")

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split, KFold
from joblib import Parallel, delayed
import numpy as np

def process_fold(fold, train_index, test_index, X, y):
    model = HistGradientBoostingRegressor()
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]
    sorted_features = [(X.columns[i], perm_importance.importances_mean[i]) for i in sorted_idx]
    return score, perm_importance.importances_mean, sorted_features

# Assuming nashvilleDF is defined
X = nashvilleDF.drop(['price', 'id'], axis=1)
y = nashvilleDF['price']
X.columns = X.columns.astype(str)

num_folds = 10
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
scores = np.zeros(num_folds)
feature_importances = np.zeros((num_folds, X.shape[1]))
results = Parallel(n_jobs=-1)(delayed(process_fold)(fold, train_index, test_index, X, y) 
                              for fold, (train_index, test_index) in enumerate(kf.split(X)))

accuracy_scores = []
for fold, (score, importance, sorted_features) in enumerate(results):
    scores[fold] = score
    feature_importances[fold] = importance
    
    # Append the score to accuracy_scores list
    accuracy_scores.append(score)

    print(f'Fold {fold+1}: Accuracy = {scores[fold]}')
    print("Features ranked by importance for fold:", fold+1)
    for name, imp in sorted_features:
        print(f"{name}: {imp}")

# Compute and print the average accuracy and standard deviation across all folds
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
print(f'Average Accuracy: {mean_accuracy}')
print(f'Standard Deviation of Accuracy: {std_accuracy}')

# Compute the average feature importance across all folds
avg_feature_importances = feature_importances.mean(axis=0)

# Get the top 20 features based on average importance
top_features = np.argsort(avg_feature_importances)[-20:]
top_feature_names = X.columns[top_features]
top_feature_importances = avg_feature_importances[top_features]

print("Top 20 Features and their Importances:")
print(list(zip(top_feature_names, top_feature_importances)))


all_top_features = []
for _, _, sorted_features in results:
    top_20 = [feature[0] for feature in sorted_features[:20]]
    all_top_features.extend(top_20)

# Calculate the average rank for each feature
feature_ranks = {feature: 0 for feature in set(all_top_features)}
for feature in all_top_features:
    feature_ranks[feature] += (all_top_features.count(feature) - all_top_features.index(feature) - 1)

average_ranks = {feature: rank / all_top_features.count(feature) for feature, rank in feature_ranks.items()}

# Sort by average rank
sorted_avg_ranks = sorted(average_ranks.items(), key=lambda x: x[1], reverse=True)

# Displaying the sorted average ranks
for feature, rank in sorted_avg_ranks:
    print(f"{feature}: {rank}")

# Plotting the graph
features = [item[0] for item in sorted_avg_ranks]
ranks = [item[1] for item in sorted_avg_ranks]

plt.figure(figsize=(12, 10))
plt.barh(features, ranks, color='skyblue')
plt.xlabel('Average Rank')
plt.ylabel('Feature Name')
plt.title('Average Rank of Top Features')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()