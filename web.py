import joblib
from trainrft import RandomForestModel

# Load the saved Random Forest model
rf_model = joblib.load('rf_model.joblib')


from flask import Flask, render_template, request, url_for
import requests
from bs4 import BeautifulSoup
import pandas as pd


app = Flask(__name__)

def scrape_features(players):
    import requests
    from bs4 import BeautifulSoup

    # Fetch the webpage content
    url = 'https://www.atptour.com/en/rankings/singles?rankRange=0-5000'  # Replace with the actual website URL
    response = requests.get(url)
    import time

# Wait for 5 seconds
    time.sleep(5)

# Access the HTML content
    html_content = response.text

# Print the HTML source
    print(html_content[:10000])

    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')

    # Find all rows with the class "lower row"
    lower_row_elements = soup.find_all('tr', class_='lower-row')

    # Iterate through the rows and isolate the one with the desired name
    target_name_0 = '-'.join(players[0].split()).lower()
    target_name_1 = '-'.join(players[1].split()).lower()  # Replace with the name you're looking for
    print(target_name_0)
    print(target_name_1)
    print(lower_row_elements)
    print("length of rows = ", len(lower_row_elements))
    for row in lower_row_elements:
        print(row.get_text())
        if target_name_0 in row.get_text():
            rank1 = row.find('td', class_='rank').get_text(strip=True)
            points1 = row.find('td', class_='points').get_text(strip=True)
            break
    for row in lower_row_elements:
        if target_name_1 in row.get_text():
            rank2 = row.find('td', class_='rank').get_text(strip=True)
            points2 = row.find('td', class_='points').get_text(strip=True)
            break
    print(rank1,points1, rank2, points2)

    return {'feature1': 0, 'feature2': 0}  # Replace with actual features



@app.route('/', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        player1 = request.form['player1']
        player2 = request.form['player2']
        rank1 = request.form['rank1']
        rank2 = request.form['rank2']
        points1 = request.form['points1']
        points2 = request.form['points2']
        year = request.form['year']
        Avgodds1 = request.form['Avgodds1']
        Avgodds2 = request.form['Avgodds2']
        bestodds1 = request.form['bestodds1']
        bestodds2 = request.form['bestodds2']
        series = request.form['series']
        round = request.form['round']
        
        # Scrape features
        print(player1, player2)
        #scrape_features([player1, player2])
        #features = scrape_features([player1, player2])
        features = ['AvgL', 'AvgW', 'Pointsdiff', 'LPts', 'WPts', 'LRank', 'WRank', 'B365L', 'B365W', 'Date', 'Round_encoded', 'Series_encoded', 'Info', 'Surface_Hard', 'Surface_Clay']
        # Create a DataFrame with one row and features as columns
        df = pd.DataFrame({
            'AvgL': [float(Avgodds2)],
            'AvgW': [float(Avgodds1)],
            'Pointsdiff': [float(points1) - float(points2)],
            'LPts': [float(points2)],
            'WPts': [float(points1)],
            'LRank': [float(rank2)],
            'WRank': [float(rank1)],
            'B365L': [float(bestodds2)],
            'B365W': [float(bestodds1)],
            'Date': [int(year) - 2013],
            'Round_encoded': [float(round)],
            'Series_encoded': [float(series) / 250],
            'Info': [0]
        })

        # Load the MinMaxScaler
        scaler = joblib.load('min_max_scaler.pkl')

        # Apply scaling to relevant columns
        columns_to_scale = ['WPts', 'LPts', 'Pointsdiff']
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])

        # Convert DataFrame to a list of features
        features = df.iloc[0].tolist()
        # Load the model
        print(features)
        model = RandomForestModel()  # Adjust based on your actual model structure
        
        # Make prediction
        prediction = model.predict(pd.DataFrame([features]))
        
        return render_template('result.html', prediction=prediction)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)