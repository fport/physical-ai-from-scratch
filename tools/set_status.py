"""Set the status of lessons in curriculum.yaml.  Usage: set_status.py done 1 2 3"""
import re
import sys

status, nums = sys.argv[1], {int(x) for x in sys.argv[2:]}
path = __file__.rsplit("/tools/", 1)[0] + "/curriculum.yaml"
s = open(path).read()
s = re.sub(r"(\{n: (\d+),[^}]*?status: )(\w+)", lambda m: m.group(1) + (status if int(m.group(2)) in nums else m.group(3)), s)
open(path, "w").write(s)
