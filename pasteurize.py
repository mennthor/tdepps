# coding: utf-8

"""Usage: pasteurize.py [-nd DIR]

Add python2 backwards contability to python3 code.

Options:
  -h --help   Show this help message.
  -d DIR      Top directory to search for python files. [default: ./]
  -n          Dry run, not modifiyng the file system.

"""

import os
import subprocess
from docopt import docopt

if __name__ == "__main__":
    args = docopt(__doc__)

    topdir = args.pop("-d", os.path.curdir).rstrip("/")
    DRYMODE = args.pop("-n", False)

    if DRYMODE:
        print("#####################################")
        print("## In DRYMODE. Not modifying any file")
        print("#####################################")

    if not os.path.exists(topdir):
        raise ValueError("Directory '{} not found.".format(topdir))
    if not os.path.isdir(topdir):
        raise ValueError("Given dir '{}' is not a directory.".format(topdir))

    # Walk through package and collect all python files
    pyfiles = []
    for dirpath, dirnames, filenames in os.walk(topdir):
        pyfiles += [os.path.join(dirpath, f) for f in filenames
                    if f.endswith(".py")]

    print("Checking found files:")
    for f in pyfiles:
        print("  - {}".format(f))

    # `pasteurize` all py3 files to make them work with py2 too.
    # `pasteurize` will also make a backup.
    for f in pyfiles:
        if DRYMODE:
            cmd = ["pasteurize", f]
        else:
            cmd = ["pasteurize", "-w", f]
        subprocess.call(cmd)

    # We need to remove `unicode_literals` import because it breaks#
    # np.record.arrays dtypes. Best make sure to avoid unicodes in py3 code.
    remove_str = "from __future__ import unicode_literals\n"
    print("Removing 'from __future__ import unicode_literals' from:")
    for f in pyfiles:
        with open(f, "r") as fi:
            data = fi.readlines()

        if remove_str in data:
            # Don't forget '\n' as readline doesn't remove it
            print("  - {}".format(f))
            data.remove(remove_str)

        if not DRYMODE:
            with open(f, "w") as fi:
                fi.writelines(data)

    # At last, prepend every file with a utf-8 coding string
    add_coding = "# coding: utf-8\n"
    print("Prepending '# coding: utf-8' to files:")
    for f in pyfiles:
        with open(f, "r") as fi:
            data = fi.readlines()

        # Only append if not already there
        if add_coding not in data:
            print("  - {}".format(f))
            data = [add_coding + "\n"] + data
            if not DRYMODE:
                with open(f, "w") as fi:
                    fi.writelines(data)
