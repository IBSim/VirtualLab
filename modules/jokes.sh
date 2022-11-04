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
echo "${edge}"
echo "${msg}"
echo "${edge}"
echo
tput setaf 4
echo "******************************"
echo "Message recived and understood"
echo "Test Sucessfull!!"
echo "Here is a joke to celebrate"
echo "******************************"
tput sgr 0
echo
# NOTE: CENSORED VERSION..........
# EXPLICIT JOKES EXCLUDED
#
CENSORED_ENDPT="https://v2.jokeapi.dev/joke/Any?blacklistFlags=nsfw,religious,political,racist,sexist,explicit"


DATA=`curl -s $CENSORED_ENDPT | tr '\r\n' ' '`

TYPE=`echo $DATA | jq -r ".type" `

ART=$(ls /usr/share/cowsay/cows/gnu.cow | shuf -n1)

case $TYPE in

	single)
		JOKE=`echo $DATA | jq -r ".joke"`

		cowsay -f $ART $JOKE

		;;

	twopart)

		SETUP=`echo $DATA | jq -r ".setup"`

		DELIVERY=`echo $DATA | jq -r ".delivery"`

		

	        cowsay -f $ART $SETUP

		sleep 1

		cowsay -f $ART $DELIVERY
		;;

	*)
		cowsay Sorry I Failed to entertain you!
		;;
esac
