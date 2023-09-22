import pandas as pd
from flask import Flask, render_template, redirect, url_for, flash, request
from flask_wtf import FlaskForm
from flask import jsonify
from wtforms import FloatField, SubmitField, BooleanField, IntegerField, SelectField
from wtforms.validators import DataRequired
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')

app = Flask(__name__, static_folder='Statics')
app.config['SECRET_KEY'] = 'wdazD1dRmBGVwVSi'





# Load or train the model
try:
    trained_model = load('trained_model.joblib')
    mean_features = nashvilleDF.drop(['price', 'id'], axis=1).mean().to_dict()
except:
    nashvilleDF = pd.read_csv('nashvilleDF.csv')
    mean_features = nashvilleDF.mean().to_dict()
    X = nashvilleDF.drop(['price', 'id'], axis=1)
    y = nashvilleDF['price']
    trained_model = HistGradientBoostingRegressor()
    trained_model.fit(X, y)
    dump(trained_model, 'trained_model.joblib')



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

trained_model.fit(X_train, y_train)
y_pred = trained_model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)



class PredictionForm(FlaskForm):
    accommodates = FloatField('How many people do you want to stay?', validators=[DataRequired()])
    bathrooms = FloatField('Enter the number of bathrooms (e.g. 1.5)', validators=[DataRequired()])
    nights_staying = FloatField('How many nights are you staying?', validators=[DataRequired()])
    bedrooms = FloatField('Enter the number of bedrooms', validators=[DataRequired()])
    fireplace = BooleanField('Do you want a fireplace?')
    hot_tub = BooleanField('Do you want a hot tub?')
    cable = BooleanField('Do you want cable TV?')
    wifi = BooleanField('Do you want WiFi?') # Added WiFi field
    review_scores_value = FloatField('Enter desired review score (1-5 or percentage)', validators=[DataRequired()])
    property_type = SelectField('Choose Property Type', choices=[
        ('prop_Entire condo', 'Entire Condo'),
        ('prop_Entire guest suite', 'Entire Guest Suite'),
        ('prop_Entire guesthouse', 'Entire Guesthouse'),
        ('prop_Entire home', 'Entire Home'),
        ('prop_Entire rental unit', 'Entire Rental Unit'),
        ('prop_Entire townhouse', 'Entire Townhouse'),
        ('prop_Hotel', 'Hotel'),
        ('prop_Private room', 'Private Room')
    ])
    room_type_num = SelectField('Choose Room Type', choices=[
        (1, 'Entire home/apt'),
        (2, 'Private Room')
    ])
    submit = SubmitField('Predict Price')
    
# Route to the Prediction Model Page
@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    form = PredictionForm()
    estimated_price = None

    if form.validate_on_submit():
        user_data = {key: value for key, value in form.data.items() if key != 'nights_staying' and key != 'csrf_token'}

        # Convert specific fields to appropriate values
        for key, value in user_data.items():
            if value == 'Yes':
                user_data[key] = 1
            elif value == 'No':
                user_data[key] = 0
            elif isinstance(value, str) and "%" in value:
                user_data[key] = float(value.strip('%')) / 100

        # Handle one-hot encoding for property_type
        property_types = [
            'prop_Entire condo', 'prop_Entire guest suite', 'prop_Entire guesthouse',
            'prop_Entire home', 'prop_Entire rental unit', 'prop_Entire townhouse',
            'prop_Hotel', 'prop_Private room'
        ]
        selected_property_type = user_data.pop('property_type')
        for prop_type in property_types:
            user_data[prop_type] = 1 if prop_type == selected_property_type else 0

        # Ensure hot tub is correctly encoded
        if 'hot_tub' in user_data:
            user_data['hot tub'] = user_data.pop('hot_tub')

        # Populate missing features with their average values
        prepared_data = mean_features.copy()
        prepared_data.update(user_data)

        if 'id' in prepared_data: 
            del prepared_data['id']
        if 'price' in prepared_data: 
            del prepared_data['price']
        if 'submit' in prepared_data: 
            del prepared_data['submit']

        # Now, let's filter the nashvilleDF dataframe based on the user's criteria:

        # Start with the entire dataset and filter it step by step
        filtered_properties = nashvilleDF.copy()

        # 1. Accommodates
        filtered_properties = filtered_properties[filtered_properties['accommodates'] >= user_data['accommodates']]

        # 2. Bathrooms
        filtered_properties = filtered_properties[filtered_properties['bathrooms'] >= user_data['bathrooms']]

        # 3. Nights staying
        # Use nights_staying to filter the properties
        nights_staying_value = form.data['nights_staying']
        filtered_properties = filtered_properties[
            (filtered_properties['minimum_minimum_nights'] <= nights_staying_value) &
            (filtered_properties['maximum_maximum_nights'] >= nights_staying_value)
        ]

        # Now, drop the key from the dictionary since it's not needed anymore
        user_data.pop('nights_staying', None)


        # 4. Bedrooms
        filtered_properties = filtered_properties[filtered_properties['bedrooms'] >= user_data['bedrooms']]

        # 5. Amenities
        amenities = ['fireplace', 'hot tub', 'cable', 'wifi']
        for amenity in amenities:
            if user_data[amenity]:
                filtered_properties = filtered_properties[filtered_properties[amenity] == 1]

        # 6. Review scores
        min_review_score = user_data['review_scores_value'] - 1
        filtered_properties = filtered_properties[filtered_properties['review_scores_value'] > min_review_score]

        # 7. Property type
        # The filtering for property type is inherently done through the one-hot encoding process

        # 8. Room type
        room_type_mapping = {1: 'Entire home/apt', 2: 'Private Room'}
        selected_room_type = room_type_mapping[int(user_data['room_type_num'])]
        if selected_room_type == 'Entire home/apt':
            filtered_properties = filtered_properties[filtered_properties['room_Entire home/apt'] == 1]
        elif selected_room_type == 'Private Room':
            filtered_properties = filtered_properties[filtered_properties['room_Private room'] == 1]



        # Predict the price using the trained model
        try:
            estimated_price = trained_model.predict([list(prepared_data.values())])[0]
            estimated_price = round(estimated_price, 2)

            recommended_properties = filtered_properties.sort_values(by='review_scores_value', ascending=False).head(5)
            
           

            return render_template('price.html', 
                       estimated_price=estimated_price, 
                       mae=round(mae, 2), 
                       recommended_properties=recommended_properties)

        except Exception as e:
            print(f"Error during prediction: {str(e)}")

    return render_template('prediction.html', title='Prediction Model', form=form, estimated_price=estimated_price)



@app.route('/price')
def price():
    estimated_price = request.args.get('estimated_price', 'N/A')
    mae = request.args.get('mae', 'N/A')
    return render_template('price.html', estimated_price=estimated_price, mae=mae)


# Route to the About Page
@app.route('/about')
def about():
    return render_template('about.html', title='About Page')

if __name__ == '__main__':
    app.run(debug=True)
