#'''AvgL' -> match, 'AvgW'->match, 'Pointsdiff'->rank, 'LPts'->rank, 'WPts'->rank, 'LRank'->rank, 'WRank'->rank, 'B365L'->match, 'B365W'->match, 'Date'->match, 'Round_encoded'->match, 'Series_encoded'->match, 'Info'->match, 'Surface_Hard'->match, 'Surface_Clay->match'''
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time

# Path to your ChromeDriver (make sure this path is correct)

try:

# Step 2: Launch the browser using the service
    driver = webdriver.Chrome()

    # Step 2: Open the website
    driver.get("https://www.tennisexplorer.com/")

    # Step 3: Wait for the input field to be present and locate it
    wait = WebDriverWait(driver, 10)
    input_field = wait.until(EC.presence_of_element_located((By.ID, "search-text-plr")))
    time.sleep(4)
    # Step 4: Enter the player's name
    input_field.send_keys("Roger Federer")

    # Step 5: Press the Enter key
    input_field.send_keys(Keys.ENTER)
    time.sleep(4)

    page_html = driver.page_source

# Step 3: Print or process the HTML
    print(len(page_html))
    print(type(page_html))
    print(page_html.find("Roger Federer"))

    # Step 6: Wait for the search results to load (adjust timeout as needed)

    wait.until(EC.presence_of_element_located((By.CLASS_NAME, "player")))
    print("Search completed successfully.")



except NoSuchElementException as e:
    print(f"Element not found: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Close the browser
    if 'driver' in locals():
        driver.quit()
