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

    # Parse the page content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all rows with the class "lower row"
    rows = soup.find_all(class_='lower-row')

    # Iterate through the rows and isolate the one with the desired name
    target_name_0 = players[0]
    target_name_1 = players[1]  # Replace with the name you're looking for
    for row in rows:
        if target_name_0 in row.get_text():
            rank1 = row.find('td', class_='rank').get_text(strip=True)
            points1 = row.find('td', class_='points').get_text(strip=True)
            break
    for row in rows:
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
        
        # Scrape features
        print(player1, player2)
        #features = scrape_features([player1, player2])
        
        # Load the model
        #model = RandomForestModel()  # Adjust based on your actual model structure
        
        # Make prediction
        #prediction = model.predict(pd.DataFrame([features]))
        
        return render_template('result.html', prediction=1)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)