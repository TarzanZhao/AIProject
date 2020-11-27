a = {1:"zbl",2:"zcl",3:'ddl'}
print(a)
key = list(a.keys())
for i in key:
    if i != 1:
        del a[i]
print(a)