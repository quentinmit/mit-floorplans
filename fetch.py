#!/usr/bin/env python3

from __future__ import print_function

import argparse
import datetime
import git
import os
import subprocess
import time
import mechanize
import sys
import logging

parser = argparse.ArgumentParser(description='Download floorplans.')
parser.add_argument('--tc-username', type=str, required=True)
parser.add_argument('--tc-password', type=str, default=os.environ.get("TC_PASSWORD"))
parser.add_argument('--verbose', '-v', action='store_true')
parser.add_argument('--git', action='store_true',
                    help='commit new floorplans to Git')
args = parser.parse_args()

SEARCH_URL = "https://floorplans.mit.edu/SearchPDF.Asp"
LIST_URL = "https://floorplans.mit.edu/ListPDF.Asp?Bldg="

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
        logging.info("Commit at %s: %s", commit_date, dates[date])
        repo.index.commit(commit_message, author_date=commit_date, commit_date=commit_date)

if args.git:
    commit_files_by_date()

br = mechanize.Browser()
br.set_handle_robots(False)
if args.verbose:
    logging.basicConfig(level=logging.DEBUG)
    br.set_debug_http(True)
    br.set_debug_responses(True)
    br.set_debug_redirects(True)
else:
    logging.basicConfig(level=logging.INFO)
br.open(SEARCH_URL)

br.select_form(name='IdPList')
br.form["user_idp"] = ["https://idp.touchstonenetwork.net/shibboleth-idp"]
br.submit()

br.select_form(name="loginform")
br.form["j_username"] = args.tc_username
br.form["j_password"] = args.tc_password
br.submit()

# Shibboleth returns a form with a "Continue" button we have to press since we
# don't have JavaScript.
br.select_form(nr=0)
br.submit()

br.open(SEARCH_URL)

def get_building_list(br):
    br.select_form(name="frmSearchPDF")
    # item.name is the VALUE attribute
    return [item.name for item in br.find_control("Bldg").items]

wget_args = 'wget -nv -p -nH --cut-dirs=10'.split()

for cookie in br.cookiejar:
    if cookie.name.startswith('_shibsession'):
        wget_args.extend(("--header", "Cookie: %s=%s" % (cookie.name, cookie.value)))

pdf_urls = []

for building in get_building_list(br):
    br.open(LIST_URL + building)
    for floor in br.links(url_regex=r'/pdfs/'):
        pdf_urls.append(floor.absolute_url)

# TODO: Figure out what has changed since last run

try:
    subprocess.check_call(wget_args + pdf_urls)
except subprocess.CalledProcessError as e:
    print("Some downloads failed: %s" % (e,))

if args.git:
    commit_files_by_date(os.path.basename(f) for f in pdf_urls)
