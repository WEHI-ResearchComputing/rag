#!/bin/bash
# Script to download tickets from support.wehi.edu.au.
# Downloads latest 10 pages of tickets and their exchanges.
# Requires FRESHSERVICE_API_KEY to be set to an appropriate key.

set -eu

# get first 10 pages of tickets and save into variable
ticket_ids=$(\
    for i in `seq 1 10`
    do 
        curl -u $FRESHSERVICE_API_KEY:X -H "Content-Type: application/json" -X GET https://support.wehi.edu.au/helpdesk/tickets/filter/all_tickets'?'format=json'&'page=$i
    done | jq .[].display_id
)

# download ticket content and save into data/ticket*.txt
for id in $ticket_ids
do
    curl -u $FRESHSERVICE_API_KEY:X -H "Content-Type: application/json" -X GET https://support.wehi.edu.au/helpdesk/tickets/$id.json | jq .helpdesk_ticket.notes[].note.body > data/ticket$id.txt
done
