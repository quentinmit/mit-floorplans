#!/usr/bin/env python3

from __future__ import print_function

import argparse
import datetime
import git
import os
import subprocess
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

parser = argparse.ArgumentParser(description='Download floorplans.')
parser.add_argument('--duo-gen', type=str,
                    default=os.path.expanduser("~/Software/duo-cli/duo_gen.py"),
                    help='path to duo_gen.py')
parser.add_argument('--git', action='store_true',
                    help='commit new floorplans to Git')
args = parser.parse_args()

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

def commit_files_by_date(current_files=None):
    repo = git.Repo('.')
    if current_files:
        current_files = set(current_files)
        old_files = set(blob.name for blob in repo.commit().tree)
        deleted_files = old_files - current_files
        if deleted_files:
            repo.index.remove(deleted_files)
            repo.index.commit("Remove files that no longer exist")
            for f in deleted_files:
                os.remove(f)
    changed_files = repo.git.status(porcelain=True)
    changed_files = set(x.split(None, 1)[1] for x in changed_files.splitlines())
    # Group files by date
    dates = dict()
    for f in changed_files:
        t = os.stat(f).st_mtime
        date = time.strftime("%Y-%m-%d", time.localtime(t))
        dates[date] = dates.get(date, []) + [(f, t)]
    for date in sorted(dates.keys()):
        # Commit files from oldest to newest.
        commit_time = max(f[1] for f in dates[date])
        repo.index.add([f[0] for f in dates[date]])
        commit_message = "%s\n\nModified files:\n%s" % (
            date,
            "\n".join("- %s (%s)" % (f[0], time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(f[1]))) for f in sorted(dates[date])),
        )
        commit_date = time.strftime("%s %z", time.localtime(commit_time))
        print("Commit at %s: %s" % (commit_date, dates[date]))
        repo.index.commit(commit_message, author_date=commit_date, commit_date=commit_date)

if args.git:
    commit_files_by_date()

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
    otpgen = subprocess.Popen([args.duo_gen], cwd=os.path.dirname(args.duo_gen), stdout=subprocess.PIPE)
    otp, _ = otpgen.communicate()
    otp = otp.decode('ascii').strip()
    driver.find_element_by_name("passcode").send_keys(otp)
    pc.click()

# Wait until logged in
wait.until(EC.visibility_of_element_located((By.NAME, "Bldg")))

def get_building_list(driver):
    building_select = driver.find_element_by_name("Bldg")
    building_options = building_select.find_elements_by_tag_name("option")
    return [building_option.get_attribute("value") for building_option in building_options]

wget_args = 'wget -p -nH --cut-dirs=10'.split()

for cookie in driver.get_cookies():
    if cookie['name'].startswith('_shibsession'):
        wget_args.extend(("--header", "Cookie: %s=%s" % (cookie['name'], cookie['value'])))

pdf_urls = []

for building in get_building_list(driver):
    driver.get(LIST_URL + building)

    wait.until(EC.visibility_of_element_located((By.ID, 'maincontent')))

    for floor in driver.find_elements_by_xpath('//a[contains(@href,"/pdfs/")]'):
        pdf_urls.append(floor.get_property('href'))

driver.quit()

# TODO: Figure out what has changed since last run

try:
    subprocess.check_call(wget_args + pdf_urls)
except subprocess.CalledProcessError as e:
    print("Some downloads failed: %s" % (e,))

if args.git:
    commit_files_by_date(os.path.basename(f) for f in pdf_urls)
