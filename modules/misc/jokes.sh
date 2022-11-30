#!/bin/sh
# generates a random joke using the jq joke api and feeds it into cowsay.
# adapted from https://viruchith.com/Bash-script-to-display-a-random-joke-20-04-2021

msg="# ${1} #"
edge=$(echo "${msg}" | sed "s/./#/g")
tput setaf 2
tput bold

echo "********************"
echo "Welcome to VirtuaLab"
echo "********************"

cowsay "${msg}"

echo
tput setaf 4
echo "******************************"
echo "Message recived and understood"
echo "Test Sucessfull!!"
echo "******************************"
tput sgr 0
echo
# NOTE: CENSORED VERSION..........
# EXPLICIT JOKES EXCLUDED
#
CENSORED_ENDPT="https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,religious,political,racist,sexist,explicit"


DATA=`curl -s $CENSORED_ENDPT | tr '\r\n' ' '`

TYPE=`echo $DATA | jq -r ".type" `

ART=$(ls /usr/share/cowsay/cows/ | shuf -n1)

case $TYPE in

	single)
		JOKE=`echo $DATA | jq -r ".joke"`

		cowsay $JOKE

		;;

	twopart)

		SETUP=`echo $DATA | jq -r ".setup"`

		DELIVERY=`echo $DATA | jq -r ".delivery"`

		

	        cowsay -f $ART $SETUP

		sleep 1

		cowsay -f $ART $DELIVERY
		;;

	*)
		cowsay "Running simulation of 1000 monkeys pressing random buttons."
		cowsay -f $ART "There’s no more faith in thee than in a stewed prune."
		echo "< Henry IV Part 1 (Act 3, Scene 3) >"
		cowsay "See, I told you it would work."
		echo $DATA
		;;
esac
