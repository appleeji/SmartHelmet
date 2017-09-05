import re

infile = open("dataNoAccident.txt","r")
s = infile.read()
numbers = re.split("['\n' ]",s)
print(numbers)
