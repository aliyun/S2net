import os, sys, inspect, platform, subprocess

def robust_exec(cmds):
    src_file_name = os.path.basename(inspect.getfile(inspect.currentframe()))
    if platform.system() == 'Linux':
        os.environ['LD_LIBRARY_PATH'] = os.path.dirname(os.path.abspath(os.sys.argv[0]))
    if type(cmds) == str:
        p = subprocess.Popen(shlex.split(cmds.encode('string-escape')), env=os.environ
            , cwd=os.path.dirname(os.path.abspath(os.sys.argv[0])))
        try: p.wait()
        except:
            src_file_name = os.path.basename(inspect.getfile(inspect.currentframe()))
            print('[{0}] exception in robust_exec()'.format(src_file_name))
            if platform.system() == 'Linux':
                p.send_signal(9)
            else:
                p.kill()
            while p.poll() == None:
                print('[{0}] waiting the child exit'.format(src_file_name))
                time.sleep(0.05)
                if platform.system() == 'Linux':
                    p.send_signal(9)
                else:
                    p.kill()
                pass
            exit(1)
    else:
        p = subprocess.Popen(map(str, cmds), env=os.environ
            , cwd=os.path.dirname(os.path.abspath(os.sys.argv[0])))
        try: p.wait()
        except:
            print('[{0}] exception in robust_exec()'.format(src_file_name))
            if platform.system() == 'Linux':
                p.send_signal(9)
            else:
                p.kill()
            while p.poll() == None:
                print('[{0}] waiting the child exit'.format(src_file_name))
                time.sleep(0.05)
                if platform.system() == 'Linux':
                    p.send_signal(9)
                else:
                    p.kill()
                pass
            exit(1)
