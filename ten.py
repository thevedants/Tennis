import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Initialize the webdriver (you may need to specify the path to your chromedriver)
driver = webdriver.Chrome()

# Navigate to the initial website
initial_url = "https://www.tennisexplorer.com/"  # Replace with the actual URL
driver.get(initial_url)

# Wait for the form to be present on the page
form = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.TAG_NAME, "form"))
)

# Find the input field and submit button
input_field = form.find_element(By.TAG_NAME, "input")
submit_button = form.find_element(By.TAG_NAME, "button")

# Fill in the input
input_field.send_keys("Your input here")

# Submit the form
submit_button.click()

# Wait for the new page to load
WebDriverWait(driver, 10).until(
    EC.url_changes(initial_url)
)

# Get the new URL
new_url = driver.current_url

# Now scrape the new website
page_source = driver.page_source
soup = BeautifulSoup(page_source, 'html.parser')

# Example scraping (modify according to your needs)
# Find all paragraph elements
paragraphs = soup.find_all('p')

# Print the text content of each paragraph
for p in paragraphs:
    print(p.text)

# Close the browser
driver.quit()
