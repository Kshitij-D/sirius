import re,sys
def fails(p):
    cur=None; S=set()
    for l in open(p,errors='replace'):
        m=re.match(r'^\[\d+/\d+\] \(\d+%\): (.+)$',l)
        if m: cur=m.group(1).strip(); continue
        if cur and re.search(r': failed:|FAILED|: failed ', l): S.add(cur)
    return S
A,B,C=[fails(f'tests_{t}.log') for t in ('base_default','final_default','final_grant')]
print(len(A),len(B),len(C))
print('final-default minus base:',sorted(B-A)); print('base minus final-default:',sorted(A-B))
print('grant minus final-default:',sorted(C-B)); print('final-default minus grant:',sorted(B-C))
