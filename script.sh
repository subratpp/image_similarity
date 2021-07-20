#!/bin/sh

inotifywait -qm ./test -e create -e moved_to | 
	while read path action file; do
		echo "The file '$file' appeared in directory '$path' via '$action'"
		python check_similarity.py "${file}"
		# do something with the file
	done

