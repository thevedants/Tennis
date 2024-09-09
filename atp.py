from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
# Use Playwright to open the page and get the HTML
def get_html_from_pokernow(url):
    with sync_playwright() as p:
        
        # Launch a browser (Chromium is used by default)
        chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

        browser = p.chromium.launch(executable_path=chrome_path, headless=True)
        page = browser.new_page()

        # Go to the page
        page.goto(url)

        # Wait for the page to fully load
        page.wait_for_timeout(5000)  # 5 seconds, adjust based on load time

        # Get the HTML content
        html = page.content()
        
        # Close the browser
        browser.close()

        return html

# Example usage
url = 'https://www.tennisexplorer.com/match-detail/?id=2717899'
html_content = get_html_from_pokernow(url)
soup = BeautifulSoup(html_content, 'lxml')

div_class1 = soup.find('div', id='oddsMenu-1-data')

# Step 4: Find the table inside this div
table = div_class1.find('table', class_='result balance ')

# Step 5: Find the first 'tr' element with class "one" within this table
tr_one = table.find('tr', class_='one')
tr_avg = table.find('tr', class_='average')

# Step 6: Extract data (for example, text) from the first 'tr' with class "one"
if tr_one:
    print("First row with class 'one':")
    print(tr_one.get_text(strip=True))
if tr_avg:
    print("First row with class 'one':")
    print(tr_avg.get_text(strip=True))
