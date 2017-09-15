import os



if __name__ == '__main__':
    dir = os.path.dirname(os.path.realpath(__file__))
    from subprocess import check_output

    stdout = check_output('"%s" html' % os.path.join(dir, 'make.bat'), shell=True).decode()
    print(stdout)