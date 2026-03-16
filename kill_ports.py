import os
import sys
import psutil

def kill_processes_on_ports(ports):
    """Find and kill processes listening on specified ports."""
    for port in ports:
        print(f"Checking for processes on port {port}...")
        found = False
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                for conns in proc.connections(kind='inet'):
                    if conns.laddr.port == port:
                        print(f"Killing process {proc.info['name']} (PID: {proc.info['pid']}) on port {port}")
                        proc.terminate()
                        proc.wait(timeout=3)
                        found = True
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except psutil.TimeoutExpired:
                print(f"Process {proc.info['pid']} did not terminate in time, killing forcefully...")
                proc.kill()
        
        if not found:
            print(f"No process found on port {port}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python kill_ports.py <port1> <port2> ...")
        sys.exit(1)
    
    target_ports = [int(p) for p in sys.argv[1:]]
    kill_processes_on_ports(target_ports)
