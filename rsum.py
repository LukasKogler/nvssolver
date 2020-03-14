row = list(float(x) for x in "  -0.164       0       0       0 0.117017 0.115175 0.046983 -0.115175       0       0       0       0       0       0       0       0       0       0       0       0       0       0       0".split())
odd = row[1::2]
even = row[0::2]
print(row)
print(odd)
print(even)

print(sum(row))
print(sum(odd))
print(sum(even))
# print(sum(row[0:2:]))
# print(sum(row[1:2:]))

