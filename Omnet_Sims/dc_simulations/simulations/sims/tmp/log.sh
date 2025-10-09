grep -i 'occupancy:' opp_runall.log | awk '{print $6}' | sort -nr | head -n 5
