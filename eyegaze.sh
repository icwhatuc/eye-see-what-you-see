Use set -e

make icwhatuc_display_piped 
make eyegaze_factored 
./eyegaze_factored $1 | ./icwhatuc_display_piped Posters/
