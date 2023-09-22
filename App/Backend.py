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
    neighbourhood_cleansed_num = FloatField('Enter neighborhood number (1-20)', validators=[DataRequired()])
    nights_staying = FloatField('How many nights are you staying?', validators=[DataRequired()])
    bedrooms = FloatField('Enter the number of bedrooms', validators=[DataRequired()])
    fireplace = BooleanField('Do you want a fireplace?')
    hot_tub = BooleanField('Do you want a hot tub?')
    cable = BooleanField('Do you want cable TV?')
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

    if request.method == 'POST' and form.validate_on_submit() == False:
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

        # Compare features between model and user data
        model_features = set(X.columns)
        prepared_features = set(prepared_data.keys())


        try:
            estimated_price = trained_model.predict([list(prepared_data.values())])[0]
            estimated_price = round(estimated_price, 2)
            response_data = {
                'estimated_price': estimated_price,
                'mae': round(mae, 2)
            }
        except Exception as e:
            print(f"Error during prediction: {str(e)}")

        # If it's an AJAX request, return both the predicted price and the MAE
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({"estimated_price": str(estimated_price), "mae": str(mae)})

        return render_template('prediction.html', title='Prediction Model', form=form)
    
    print(f'Estimated Price: ${estimated_price}')
    
    #if request.method == 'POST':
    #    redirect("/price", code=302)

    return render_template('prediction.html', title='Prediction Model', form=form, estimated_price=estimated_price)
    # return redirect("/price", code=302)


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
