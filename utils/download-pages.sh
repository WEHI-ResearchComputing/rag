#!/bin/bash

sp_pages=$(m365 spo page list --webUrl https://wehieduau.sharepoint.com/sites/rc2 --query '[*].Name' | jq -r '.|join(" ")')

for page in $sp_pages; 
do
    # Need to get pages' JSON output from Office 365 CLI and preprocess
    ## get the json content 
    temp="$(m365 spo page get --webUrl https://wehieduau.sharepoint.com/sites/rc2 --name "$page" --query canvasContentJson)"
    ## remove quotation at back
    temp="${temp%\"}"
    ## remove quotation at front
    temp="${temp#\"}"
    ## replace replace \" with " and \\" with \"
    ## then extract content with jq (ignoring null)
    echo "$temp" | sed 's/\\"/"/g' | sed 's/\\\\"/\\"/g' | jq -r .[].innerHTML | grep -v '^null' > data/$page.html

done
