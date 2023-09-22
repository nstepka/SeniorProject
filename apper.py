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

        # Removing unwanted keys
        unwanted_keys = ['id', 'price', 'submit']
        for key in unwanted_keys:
            if key in prepared_data:
                del prepared_data[key]

        # Compare features between model and user data
        model_features = set(X.columns)
        prepared_features = set(prepared_data.keys())

        missing_features = model_features - prepared_features
        extra_features = prepared_features - model_features

        print("Missing features:", missing_features)
        print("Extra features:", extra_features)

        try:
            estimated_price = trained_model.predict([list(prepared_data.values())])[0]
            estimated_price = round(estimated_price, 2)
        except Exception as e:
            print(f"Error during prediction: {str(e)}")

        # If it's an AJAX request, return just the predicted price
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return str(estimated_price)

        flash(f'Estimated Price: ${estimated_price}', 'info')
        print(f'Estimated Price: ${estimated_price}')

    return render_template('prediction.html', title='Prediction Model', form=form, estimated_price=estimated_price)
