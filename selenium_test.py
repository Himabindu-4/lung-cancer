from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import time

# Initialize WebDriver (use the path to your ChromeDriver)
driver = webdriver.Chrome(executable_path='path/to/chromedriver')

# Open the locally running Flask app
driver.get("http://127.0.0.1:5000")  # Assuming Flask is running locally

# Check that the home page is loaded
assert "Lung Cancer Prediction API" in driver.page_source

# Navigate to the prediction route (if it's a form-based interface)
driver.get("http://127.0.0.1:5000/predict")

# Find the input fields for prediction (assuming you have a form)
input_element = driver.find_element(By.NAME, 'features')  # Example input field name

# Enter data into the form field
input_element.send_keys("45, Male, Non-smoker")  # Sample data

# Submit the form (assuming there's a submit button)
submit_button = driver.find_element(By.ID, 'submit')
submit_button.click()

# Wait for the response and print it
time.sleep(2)  # Adjust as necessary for loading time
response = driver.page_source
print(response)

# Validate if the prediction response appears on the page
assert "prediction" in response

# Close the browser
driver.quit()