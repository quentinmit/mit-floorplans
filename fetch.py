#!/usr/bin/env python2

import os
import subprocess
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

DUO_GEN = os.path.expanduser("~/Software/duo-cli/duo_gen.py")

options = Options()
#options.add_argument("--ppapi=0")
#options.add_argument("--headless")
options.add_argument("--enable-logging=stderr")
options.add_argument("--vmodule=content_settings_policy_provider=2")
options.add_experimental_option("prefs", {
#    "download.default_directory" : "/data/books/chrome/",
    'plugins.always_open_pdf_externally': True,
    'profile.default_content_setting_values.automatic_downloads': 2,
#    'profile.default_content_setting_values.auto_select_certificate': 1,
    'profile.default_content_setting_values.auto_select_certificate': {"filters":[{}]},
    'profile.managed_auto_select_certificate_for_urls': ['{"pattern":"https://idp.mit.edu:446","filter":{"ISSUER":{"OU":"Client CA v1"}}}'],
#    'profile.default_content_setting_values.auto_select_certificate': ['{"pattern":"https://idp.mit.edu:446","filter":{}}'],
#    'profile.default_content_setting_values.auto_select_certificate': ['{"pattern":"https://[*.]mit.edu:446","filter":{"ISSUER":{"O":"Massachusetts Institute of Technology"}}}'],
})

driver = webdriver.Chrome(executable_path=os.path.abspath("chromedriver"),
                          service_args=["--verbose", "--log-path=/tmp/chromedriver.log"],
                          options=options)  
wait = WebDriverWait(driver, 10)
driver.get("https://floorplans.mit.edu/searchPDF.asp")

# TODO: Skip if already logged in or don't need WAYF
wait.until(EC.visibility_of_element_located((By.NAME, 'user_idp')))

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
