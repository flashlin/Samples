#!/bin/bash
set -e

if [ $# -lt 1 ]; then
   ps aux
   echo "Usage: $0 <arg1>"
   echo "lp   :list ports"
   exit 0
fi

action=$1

if [ "lp" == "$action" ]; then
   echo "netstat -tulpn"
   # netstat -tulpn
   # lsof -i :8888 
   # ps -p 2462 -o pid,cmd
   netstat -tulpn | grep LISTEN | awk '{print $1, $4}' | while read line; do
      protocol=$(echo $line | cut -d ':' -f1)
      port=$(echo $line | cut -d ':' -f2)

      if [[ $port =~ ^[0-9]+$ ]]; then
         # 查找特定端口的进程ID（PID）
         pids_result=$(lsof -i :$port -t)
         if [ -n "$pids_result" ]; then
            read -ra pids <<< "$pids_result"
            for pid in "${pids[@]}"; do
               app_info=$(ps -p $pid -o pid,cmd --no-headers)
               echo "PID: $pid, Port: $port, $protocol, cmd: $app_info"
            done
         fi
      fi
   done
   exit
fi

echo "unknown action"