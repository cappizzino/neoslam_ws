# source: https://github.com/ctu-mrs/uav_core/blob/6117a228c0d6708b1275115c6e4b60091ed3a6f8/miscellaneous/shell_additions/shell_additions.sh

# #{ killp()

# allows killing process with all its children
killp() {

  if [ $# -eq 0 ]; then
    echo "The command killp() needs an argument, but none was provided!"
    return
  else
    pes=$1
  fi

  for child in $(ps -o pid,ppid -ax | \
    awk "{ if ( \$2 == $pes ) { print \$1 }}")
    do
      # echo "Killing child process $child because ppid = $pes"
      killp $child
    done

# echo "killing $1"
kill -9 "$1" > /dev/null 2> /dev/null
}

## --------------------------------------------------------------
## |                       waitFor* macros                      |
## --------------------------------------------------------------

# #{ waitForRos()

waitForRos() {
  echo "waiting for ROS"
  until timeout 6s rosparam get /run_id > /dev/null 2>&1; do
    echo "waiting for /run_id"
    sleep 1;
  done
  sleep 1;
}

# #{ waitForSimulation()

waitForSimulation() {
  until timeout 6s rostopic echo /gazebo/model_states -n 1 --noarr > /dev/null 2>&1; do
    echo "waiting for simulation"
    sleep 1;
  done
  sleep 1;
}

## --------------------------------------------------------------
## |                       check* macros                      |
## --------------------------------------------------------------

# #{ checkRos()

checkRos() {
  echo "checking ROS"
  return $(rosparam get /run_id > /dev/null 2>&1)
}