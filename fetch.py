#!/usr/bin/env python2

from __future__ import print_function

import os
import subprocess
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

DUO_GEN = os.path.expanduser("~/Software/duo-cli/duo_gen.py")

SEARCH_URL = "https://floorplans.mit.edu/SearchPDF.Asp"
LIST_URL = "https://floorplans.mit.edu/ListPDF.Asp?Bldg="

options = Options()
#options.add_argument("--headless")
options.add_experimental_option("prefs", {
#    "download.default_directory" : DOWNLOAD_DIR,
    'plugins.always_open_pdf_externally': True,
    'profile.default_content_setting_values.automatic_downloads': 2,
    'profile.managed_auto_select_certificate_for_urls': ['{"pattern":"https://idp.mit.edu:446","filter":{"ISSUER":{"OU":"Client CA v1"}}}'],
})

driver = webdriver.Chrome(executable_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "chromedriver")),
                          service_args=["--verbose", "--log-path=/tmp/chromedriver.log"],
                          options=options)  
wait = WebDriverWait(driver, 10)
driver.get(SEARCH_URL)

# TODO: Skip if already logged in or don't need WAYF
wait.until(EC.visibility_of_element_located((By.NAME, 'user_idp')))

# TODO: Skip if already logged in
if driver.find_element_by_name("user_idp"):
    driver.find_element_by_id("Select").click()

wait.until(EC.element_to_be_clickable((By.NAME, "login_certificate"))).click()

wait.until(EC.frame_to_be_available_and_switch_to_it((By.ID, 'duo_iframe')))
pc = driver.find_element_by_id("passcode")
if pc:
    pc.click()
    otpgen = subprocess.Popen([DUO_GEN], cwd=os.path.dirname(DUO_GEN), stdout=subprocess.PIPE)
    otp, _ = otpgen.communicate()
    otp = otp.strip()
    driver.find_element_by_name("passcode").send_keys(otp)
    pc.click()

# Wait until logged in
wait.until(EC.visibility_of_element_located((By.NAME, "Bldg")))

def get_building_list(driver):
    building_select = driver.find_element_by_name("Bldg")
    building_options = building_select.find_elements_by_tag_name("option")
    return [building_option.get_attribute("value") for building_option in building_options]

wget_args = ['wget', '-N']

for cookie in driver.get_cookies():
    if cookie['name'].startswith('_shibsession'):
        wget_args.extend(("--header", "Cookie: %s=%s" % (cookie['name'], cookie['value'])))

pdf_urls = []

for building in get_building_list(driver):
    driver.get(LIST_URL + building)

    wait.until(EC.visibility_of_element_located((By.ID, 'maincontent')))

    for floor in driver.find_elements_by_xpath('//a[contains(@href,"/pdfs/")]'):
        pdf_urls.append(floor.get_property('href'))

# TODO: Figure out what has changed since last run

try:
    subprocess.check_call(wget_args + pdf_urls)
finally:
    driver.quit()
